# Kronos probing v2 — Fast Track 구현 명세 (spec.md)

> 상위 문서: `outline.md` (v2). 이 spec은 그중 **가장 빨리 결론을 내는 데 필요한 부분만** 구현한다.
> 대상 레포: `jinyoung924/Kronos-latent-space` (기본 브랜치 `main`). 1차 실험 코드(`jinyoung924/Kronos_probing`)를 옮겨 와 §3의 변경을 얹었다.
> RunPod 방식: `jinyoung924/Kronos-investing`의 `RunPod/` 디렉토리를 이식한다 (§4).
> 작성: 2026-10-04 · 개정: 2026-10-05 (구현에 맞춰 갱신. 상태: 코드와 로컬 검증 완료, pod 실행 전)

코딩 에이전트는 작업 전에 다음을 읽는다: 이 문서 → `outline.md` → `CLAUDE.md` §5(이미 밟은 지뢰) → `RunPod/README.md`.

---

## 0. 범위

Fast track = **pod 1대, RUN_ID 1개(`v2_fast`), OU → RW 순차 실행**으로 아래 다섯 질문에 답한다. outline의 나머지는 Tier 2(§10)로 미룬다.

| ID | 질문 | outline 근거 | 판정 재료 |
|---|---|---|---|
| Q0 | v1(Colab) OU 결과가 재현되는가 | §2, §6 | `v1_baseline.json` vs v2 |
| Q1 | 선형 분리가 사전학습으로 생긴 것인가 | §5.3, §8.1 | 입력 기준선 / 무작위 초기화 / 사전학습 LDR |
| Q2 | steering 효과가 S 방향에 특이적인가 | §5.1, §8.2 | A vs J(랜덤 방향, seed 3개) |
| Q3 | 역방향(−λS)도 작동하는가 | §5.2 | L |
| Q4 | RW에서도 v1 OU의 결론이 유지되는가 | §0 완결 | RW의 A, J, L, REF |

속도를 위한 설계 원칙
- v1 코드 경로를 그대로 쓴다. 새 코드는 §3에 적힌 것만 만든다.
- 데이터 생성, S 산출, 평가 절차는 v1과 동일하게 유지해 비교 가능성을 지킨다.
- n_eval만 32 → 64로 늘린다. 앞 32개는 v1과 입력·배치·seed가 같으므로 **subset32 지표**를 따로 보고해 v1과 직접 비교한다.
- OU가 끝나면 요약을 중간 push해서 RW가 도는 동안 OU 결과를 먼저 볼 수 있게 한다.

---

## 1. 고정값

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | `NeoQuasar/Kronos-base` @ `2b554741eca47781b64468546e77fef3e85130e6` | Kronos-investing `configs/base.yaml`과 같은 핀 → 볼륨의 HF 캐시 재사용 |
| 토크나이저 | `NeoQuasar/Kronos-Tokenizer-base` @ `0e0117387f39004a9016484a186a908917e22426` | 동일 |
| 로컬 프록시 | Kronos-small (코드 검증 전용, revision 미핀 허용) | 과학적 판단에 쓰지 않는다 |
| Kronos 코드 | 레포에 포함된 `Kronos/` (vendored. upstream `shiyu-coder/Kronos@67b630e`의 `model/`) | `git log -1 -- Kronos` 해시를 manifest의 `kronos_dir_commit`에 기록 |
| DataCfg | v1 그대로 (T 512, n_micro 15, sigma 0.01, ou_theta 0.01, snr U(3,6), 클래스당 2048, seed 20250812, V/A=0) | **변경 금지** |
| ProbeCfg | v1 그대로 (fp16 저장, shrinkage 1e-3, test_ratio 0.3) | |
| SteerCfg | pred_len 64, sample_count 5, temperature 1.0, top_k 0, top_p 0.9 | v1과 동일 |
| λ_rel 격자 (fast) | {0, 0.1, 0.15, 0.25, 0.5} | v1 OU 포화 0.25, RW 안전 상한 0.15~0.25 |
| n_eval | 64 (base held-out `idx_te[:64]`) | 앞 32개 = v1 평가 표본 |
| eval_batch | 32 입력 / generate 호출 (= 160 시퀀스) | RUN_ID 안에서 고정. 배치가 바뀌면 샘플 값이 바뀜 (Kronos-investing README §6.2) |
| 정밀도 | float32 추론, TF32 off | Kronos-investing에서 TF32가 결과를 바꾸는 것을 확인함 |
| tag | `v2` | v1 산출물과 분리 |
| GPU | RTX 4090 × 1, Secure Cloud, On-Demand | 한 RUN_ID 안에서 GPU 종류를 섞지 않는다 |

---

## 2. 신규 팔과 판정 규칙

### 2.1 Stage 5 fast 팔

| 팔 | 입력 | payload | 나머지 |
|---|---|---|---|
| `A_paper_median_matrix_all` | base held-out | v1 A 그대로: `λ_i · S_median_matrix` | all-tokens, 전체 12층, matrix |
| `J_ctrl_random_s0`, `_s1`, `_s2` | base held-out | 레이어 i·위치 t마다 `g ~ N(0, I_D)`, `R_i^(t) = g/‖g‖ · ‖S_i^(t)‖`, `λ_i · R` | A와 같은 층·범위·λ_i |
| `L_ctrl_reverse` | base held-out | `−λ_i · S_median_matrix` | A와 동일 |
| `REF_trend_unsteered` | trend held-out | 없음 | v1 `--reference-trend` 그대로 |

- λ=0 기준선은 한 번만 계산해 모든 팔이 공유한다 (v1 코드가 이미 그렇게 동작).
- J의 λ_i는 A와 같은 값을 쓴다. 위치별 노름을 S와 맞췄으므로 변위 크기가 A와 같다.
- 직교화 대조군(K)은 fast에서 뺀다. D=832에서 가우시안 랜덤 방향과 S의 코사인은 기대값이 약 1/√832 ≈ 0.035라 J가 사실상 직교 대조군 역할을 한다. 실측 코사인을 로그로 남긴다.

### 2.2 판정 규칙 (사전 고정, 결과를 본 뒤 바꾸지 않는다)

`stage6_summary.json["fast"]`에 기록한다. 이항검정은 v1과 같은 결과 부호 이항검정이다.

| 판정 | 조건 |
|---|---|
| `A_up` | A가 어떤 λ>0에서 양의 기울기 비율 ≥ 0.9 이고 p < 0.01. 이를 만족하는 최소 λ를 λ*로 둔다 |
| `direction_specific` | `A_up` 이고, λ*에서 J 세 seed의 양의 기울기 비율 평균이 [0.3, 0.7] 안 |
| `generic_perturbation` | `A_up` 이고, λ*에서 J 평균이 ≥ 0.9 |
| `no_effect` | `A_up` 아님 |
| (위 셋에 안 맞으면) `ambiguous` | 수치 그대로 보고 |
| `collapse_generic` | λ*에서 J의 분산비(std 비) 평균 ≤ 0.2 → 분산 붕괴가 섭동 일반에서도 일어남 |
| `reversible` | L이 어떤 λ>0에서 음의 기울기 비율 ≥ 0.9 이고 p < 0.01 |
| `ref_match` | λ*에서 A 기울기 분포 vs REF의 KS p ≥ 0.05 |

`fast_summary.md`의 해석 문장은 outline §8 표의 문장 중에서 **선택**만 한다. 새 해석 문장을 생성하지 않는다.

선택 규칙 (`stages/stage7_fast_summary.py`)

- outline §8.2 (개입): 위 판정값을 그대로 쓴다. `generic_perturbation` → 1행, `direction_specific`이고 `ref_match` 거짓 → 2행, `direction_specific`이고 `ref_match` 참 → 3행, `no_effect` → 4행. `ambiguous`이거나 REF가 없으면 "해당 행 없음"으로 적고 수치만 보고한다.
- outline §8.1 (표현): 마지막 층·t=511의 LDR 비율로 고른다. `사전학습/무작위`는 Kronos 자체 초기화 대조군의 값이다 (torch 기본 초기화는 참고 열이고 판정에 쓰지 않는다). 비율이 [0.5, 2] 안이면 "비슷", 2를 넘으면 "더 큼"으로 본다 (`Q1_RATIO = 2`). `사전학습/무작위`가 비슷 → 1행, 더 크고 `사전학습/입력`이 비슷 → 2행, 둘 다 더 큼 → 3행. `사전학습/무작위`가 0.5 미만이거나, 2를 넘는데 `사전학습/입력`이 0.5 미만이면 "해당 행 없음". 이 임계는 위 표의 규칙과 달리 근거가 있는 값이 아니다. 표의 비율을 직접 본다.

---

## 3. 코드 변경 명세

경로는 `experiment/code/` 기준이다. 1차 원본에서 출발해 아래 변경만 얹었다. `stage4`, `kexp/synth.py`, `kexp/ldr.py`, `kexp/steering_vec.py`, `kexp/hooks.py`는 1차 파일과 동일하다.

### 3.1 `kexp/paths.py`
- 루트 결정 순서: 환경변수 `KEXP_OUT` / `KEXP_SCRATCH` → 없으면 로컬 기본값(`experiment/_out`, `experiment/_scratch`).
- `in_colab()` 분기를 제거한다. `drive_root()`는 `out_root()`로 이름을 바꾸고 호출부를 일괄 수정한다.
- 하위 경로 규칙은 유지: `data/<tag>/<noise>`, `results/<tag>/<noise>_<model>`, `figs/<tag>`, `activations/<tag>/<variant>`.

### 3.2 `kexp/config.py`
- `Config.tag = "v2"`.
- `ModelCfg`에 `revision`, `tokenizer_revision` 필드 추가. `MODEL_ZOO["base"]`에 §1의 핀을 넣는다. base를 실행하는데 핀이 None이면 실패시킨다 (`load_kronos()`가 검사한다). small·mini는 핀 없이 최신을 받는다.
- `SteerCfg`에 `lambdas_rel_fast = (0.0, 0.1, 0.15, 0.25, 0.5)`, `n_eval_fast = 64`, `eval_batch = 32` 추가.
- `ControlCfg` 신설 (`Config.control`): `random_seeds = (0, 1, 2)`, `random_init_seed = 0`, `band_positions = (0, 32, 64, …, 480, 511)` (17개).
- revision 필드 추가로 `activation_hash()` 값이 바뀌는 것은 의도된 동작이다 (v2는 새로 추출).

### 3.3 `kexp/kronos_loader.py`
- `load_kronos()`: `from_pretrained(..., revision=...)`로 핀을 전달한다. Kronos-base인데 핀이 None이면 예외를 낸다.
- `set_determinism()`: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`. 모델이나 GPU 행렬 연산을 쓰는 stage(0, 2, 3, 5)가 시작할 때 호출한다. stage 4는 변경 없음(§3.8)이고 stage 1·6·7은 torch 연산이 없다.
- `randomize_model(model, seed, tokenizer=None, init="kronos")`: 무작위 초기화 대조군용. 토크나이저는 건드리지 않는다.
  - `torch.manual_seed(seed)` 후 `reset_parameters()`가 있는 모든 하위 모듈에 호출한다. 난수열이 장치에 따라 달라지지 않도록 CPU에서 초기화한 뒤 원래 장치로 돌려보낸다.
  - `reset_parameters()`가 없는 모듈이 직접 가진 파라미터는 1차원 `weight` = 1, `bias` = 0, 2차원 이상 = N(0, 0.02²)로 초기화한다.
  - `init="torch"`는 여기서 끝낸다 (PyTorch 기본 초기화). `init="kronos"`(기본)는 이어서 `model.apply(model._init_weights)`를 적용한다. Kronos가 학습을 시작할 때 쓰는 초기화다 (Linear는 xavier normal, Embedding은 std = d_model^-0.5).
  - 호출 전후 모델 파라미터 해시(`param_hash()`)가 바뀌었는지, 토크나이저 해시가 그대로인지 출력한다. 모델 해시가 그대로이거나 토크나이저 해시가 바뀌면 예외를 낸다.
  - **두 방식을 모두 돌리고, 판정에는 `kronos`만 쓴다 (2026-10-05 결정).** `torch` 방식은 임베딩이 N(0, 1)이라 Kronos 자체 초기화보다 약 29배 크고, base 구조에서 블록이 residual에 층당 약 1%만 기여한다 (마지막 층 출력과 입력 임베딩의 코사인 0.999, 실측). 스모크에서 이 대조군의 LDR은 층에 따라 거의 변하지 않았다. "학습되지 않은 트랜스포머"보다 "위치별 토큰 임베딩"에 가까운 대조군이라 참고 열로만 둔다. `kronos` 방식은 블록 기여가 층당 30~50%로 사전학습 모델과 비슷하다.

### 3.4 `kexp/intervene.py`
- `build_random_payload(S_mat, layers, lam_by_layer, seed)` 추가. 레이어 i마다 `np.random.default_rng([seed, i])`로 `[T, D]` 가우시안을 뽑고, 행별로 `‖S_i^(t)‖`에 노름을 맞춘다. 반환 형식은 `build_payload(..., form="matrix")`와 같다. `cos(R_i^(t), S_i^(t))`의 레이어별 평균 절대값을 함께 반환한다.
- 역방향은 별도 함수 없이 `lam_by_layer`에 −1을 곱해 `build_payload`를 재사용한다.
- `Steerer`는 변경하지 않는다 (AR 롤링 위치 정합 그대로).

### 3.5 `stages/stage1_dataset.py`
- `--expect-fingerprint`: 생성 후 `data_fingerprint`를 `experiment/expected/fingerprints.json`의 해당 noise 값과 대조한다. 불일치면 exit 2. 키가 없으면 경고 후 계속.
- `--write-expected`: 현재 생성 결과의 fingerprint를 위 파일에 쓴다 (로컬에서 한 번 실행해 커밋).
- 이미 생성된 데이터를 재사용할 때(`--force` 없음)도 저장된 `meta.json`의 지문으로 같은 대조를 한다.

### 3.6 `stages/stage2_activations.py`
- `--random-init`: `randomize_model()` 적용 후 추출한다 (seed는 `ControlCfg.random_init_seed`). 출력은 `activations/<tag>/<noise>_randinit/<model>[_smoke]/` 아래에 따로 쓴다. 1차의 `<noise>/<model>[_smoke]`와 같은 구조다.
- `--init {kronos,torch}` (기본 `kronos`): `--random-init`의 초기화 방식. `torch`의 출력은 `<noise>_randinit_torch/` 아래에 쓴다.
- `--positions bands`: `ControlCfg.band_positions` 위치만 저장한다 (`[N, 17, D]`). `--random-init`와 함께 쓴다.
- 기존 전체 추출 경로는 변경하지 않는다. `meta.json`에 `positions`와 `random_init`을 기록하고 재개 판정에도 쓴다.

### 3.7 `stages/stage3_ldr.py`
기존 계산은 그대로 두고 `--controls` 플래그로 아래를 추가한다. 출력은 `stage3_controls.json`, `figs/v2/stage3_controls_<noise>.png`.

- (a) **무작위 초기화 LDR**: band 위치 × 12층, held-out LDR과 null(라벨 셔플). split·shrinkage는 기존과 같다. 활성화가 없거나 위치·표본 수가 맞지 않으면 (a)만 건너뛰고 경고한다. Kronos 자체 초기화(`random_init`, 판정 기준)와 PyTorch 기본 초기화(`random_init_torch`, 참고)를 각각 잰다.
- (b) **사전학습 LDR**: 같은 band 위치 값을 같은 실행에서 계산한 전 위치 LDR(`ldr.npz`에 저장되는 값)에서 발췌한다.
- (c) **입력 기준선**: 위치 t에서 정규화 close 경로 `x_norm[:, :t+1, close]`(차원 t+1)에 같은 LDA·held-out·null 절차를 적용한다. 위치 t에서 모델이 볼 수 있는 정보와 같은 범위다.
- 요약 키 (`summary`): 마지막 층(base는 layer 11)·t=511의 `pretrained`, `random_init`, `input`과 비율 `pretrained_over_random`, `pretrained_over_input`, 그리고 참고용 `random_init_torch`와 `pretrained_over_random_torch`. 같은 값들의 band 평균(17개 위치 평균)은 `band_mean`에 넣는다.
- `--smoke`에서는 null 대비 판정과 무관하게 exit 0이다. 표본 8개로는 판정이 의미가 없고, 러너가 스모크를 끝까지 돌려야 하기 때문이다. 본 실행의 판정과 종료 코드는 1차와 같다.

### 3.8 `stages/stage4_steering_vector.py`
변경 없음. 활성화 삭제는 러너가 Stage 4 뒤에 한다 (§3.11).

### 3.9 `stages/stage5_intervene.py`
- `--arms fast`: §2.1의 A, J×3, L.
- `--lambdas fast`: §1의 격자.
- `--eval-batch N`: 입력을 N개씩 나눠 generate한다. chunk k의 seed는 `args.seed + k`. 기준선(λ=0)도 같은 분할·seed를 쓴다. 결과를 이어 붙여 기존 `evaluate()`에 넘긴다.
  - chunk 0은 v1 평가 호출과 입력·배치 구성·seed가 같다. 이것이 subset32 비교의 근거다.
  - `--eval-batch`를 생략하면 분할 없는 기존 경로다 (기록되는 `eval_batch`는 `n_eval`).
- 결과 항목에 추가: `run_settings`에 `eval_batch`, `lambdas`, `arms` 포함. `subset32` 지표(앞 32개 입력으로 계산한 `frac_positive`, `slope_mean`, `slope_std`). J 팔은 `cos_random_vs_S`(레이어 평균).
  - 이어받기 판정(조건이 다른 기존 항목 폐기)에는 `n_eval`, `pred_len`, `sample_count`, `eval_batch`만 쓴다. `lambdas`와 `arms`까지 비교하면 `--reference-trend` 실행이 앞 실행의 결과를 지운다.
  - 개념축 이동(`readout_shift_sigma`)은 L에서는 부호가 뒤집히고, J에서는 마지막 위치의 payload를 LDA 축에 투영해 잰다 (0 근처가 정상).
- `--smoke --arms fast`는 A, J_s0, L만 돌린다 (λ {0, 0.25}, n_eval 4, 결과는 `steer/results_smoke.json`).
- `--reference-trend`는 기존 동작에 같은 `n_eval`·`eval_batch`를 적용한다. 같은 `steer/results.json`에 `REF_trend_unsteered|0.0`로 저장한다.

### 3.10 `stages/stage6_report.py`
- `--smoke`: `steer/results_smoke.json`을 읽는다.
- J 세 seed를 묶은 `J_ctrl_random(pooled)` 행을 추가한다 (seed별 값과 평균).
- L은 음의 기울기 비율 이항검정으로 판정한다.
- §2.2 판정을 계산해 `stage6_summary.json["fast"]`에 기록한다. 판정값과 함께 λ별 표(`table`: A, A의 subset32, J의 seed별 값과 평균, L, A-REF KS p)와 기준선·REF의 부호 통계(전체, subset32)를 넣는다. `stage7`은 Stage 5 결과를 이 요약으로만 읽는다.
- 기하 그림의 OOD 패널은 Stage 4가 저장한 λ 격자(`steering.npz`의 `lambdas_rel`)로 그린다. 1차 코드는 Stage 5의 격자와 같다고 가정해서 fast 격자(5개)로는 그림을 그리지 못했다.

### 3.11 `run_plan.py` (신규, `experiment/code/run_plan.py`)
`colab_control.py`를 대체하는 파이프라인 러너다.

- CLI: `--plan {smoke,fast} --run-id ID --model {base,small} [--noise ou,rw]`. `--model`은 stage1을 뺀 모든 stage에 전달한다. 시작할 때 러너 버전(`run_plan v2-fast`)과 코드 커밋을 출력한다.
- `KEXP_OUT`가 없으면 `experiment/runs/<run-id>`로 설정한다. `KEXP_SCRATCH`가 없으면 pod에서는 `/root/kexp_scratch/<run-id>`, 로컬에서는 `experiment/_scratch/<run-id>`로 설정한다.
- 각 단계는 stage 스크립트를 subprocess로 실행하고, 로그를 `<OUT>/logs/<step>.log`에 tee한다. 실패하면 즉시 중단하고 exit code를 전달한다.
- 재개: 단계가 끝나면 `<OUT>/.done/<step>` 마커를 남긴다. 같은 RUN_ID로 다시 실행하면 마커가 있는 단계는 건너뛴다. 단, Stage 2 단계는 마커가 있어도 그 노이즈의 scratch 정리가 아직 안 끝났고 활성화가 없으면 다시 실행한다 (컨테이너 디스크는 pod가 바뀌면 사라진다).
- 노이즈 하나가 끝날 때마다 `stage7_fast_summary.py`를 실행한 뒤 `bash RunPod/push_meta.sh <OUT>`를 호출한다. 현재 브랜치가 `results/*`일 때만 호출하고, 실패해도 계속 진행한다. push는 `--plan fast`에서만 한다.
- 단계가 끝날 때마다 `manifest.json`(단계별 소요 시간, 상태)을 갱신한다.
- 단계 이름(로그·마커): `00_stage0`, `<X>_01_stage1`, `<X>_02_stage2`, `<X>_03_stage2_randinit`, `<X>_03b_stage2_randinit_torch`, `<X>_04_stage3`, `<X>_05_stage4`, `<X>_06_clean_scratch`, `<X>_07_stage5_fast`, `<X>_08_stage5_ref`, `<X>_09_stage6`, `<X>_10_summary`.

**fast plan** (노이즈 X ∈ [ou, rw] 순서)

| # | 단계 | 비고 |
|---|---|---|
| 0 | `stage0` | 1회. L=12, d=832 확인 |
| 1 | `stage1 --noise X --expect-fingerprint` | |
| 2 | `stage2 --noise X` | 전체 활성화 → scratch |
| 3 | `stage2 --noise X --random-init --positions bands` | Kronos 자체 초기화 (판정 기준) |
| 3b | `stage2 --noise X --random-init --init torch --positions bands` | PyTorch 기본 초기화 (참고) |
| 4 | `stage3 --noise X --controls` | |
| 5 | `stage4 --noise X` | |
| 6 | scratch 정리 | X의 전체 활성화와 randinit 활성화(둘 다) 삭제 |
| 7 | `stage5 --noise X --arms fast --lambdas fast --n-eval 64 --eval-batch 32` | |
| 8 | `stage5 --noise X --reference-trend --n-eval 64 --eval-batch 32` | |
| 9 | `stage6 --noise X` | |
| 10 | `stage7_fast_summary` + 중간 push | OU 결과가 여기서 먼저 GitHub에 올라간다 |

**smoke plan**: `--noise ou`만, 각 stage의 `--smoke` 경로를 사용한다. Stage 5는 A, J_s0, L만, λ {0, 0.25}, n_eval 4. 단계 구성은 fast와 같고(`--n-eval`·`--eval-batch`만 빠진다) `--reference-trend`와 요약까지 돌리며, push는 하지 않는다. 목적은 환경·경로·훅 오류를 몇 분 안에 잡는 것이다.

### 3.12 `stages/stage7_fast_summary.py` (신규)
- CLI: `--model {base,small} --noise ou,rw`
- 입력: 노이즈별 `stage3_summary.json`, `stage3_controls.json`, `stage4_summary.json`, `stage6_summary.json`, `data/<tag>/<noise>/meta.json`(지문, 입력단 LDR), `manifest.json`(실행 메타), 그리고 `experiment/expected/{v1_baseline,fingerprints}.json`. `stage6_summary.json`이 있는 노이즈만 반영한다.
- 출력: `<OUT>/fast_summary.md`, `<OUT>/fast_summary.json`
- 구성
  1. **Q0 재현표 (OU)**: 항목, v1 값, v2 값, outline §6 기준 판정. Stage 5 지표는 subset32를 쓴다.
     - `data_fingerprint` 완전 일치 / 입력단 LDR 소수 둘째 자리까지 / held-out LDR 네 항목 상대오차 5% 이내 / null 대비 배율 같은 자릿수(|log10 비| < 0.5) / 평활 정점 위치의 레이어 중앙값이 v1 범위 안 / 코사인 두 항목 차이 0.01 이내
     - A 팔(subset32): λ=0.25에서 양의 비율 > 0.5이고 p < 0.01, 상향되는 최소 λ가 0.25, 최소 분산비가 같은 자릿수 / REF 양의 기울기 수 ±2/32 / 기준선 양의 비율은 판정 없이 참고 (v1도 GPU에 따라 0.594 / 0.469로 갈렸다)
  2. **Q1 표 (노이즈별)**: 입력 기준선 / 무작위 초기화(Kronos 초기화, 그리고 참고 열로 torch 기본) / 사전학습 LDR (layer 11·t=511, band 평균), 비율, outline §8.1에서 선택한 해석 행.
  3. **Q2·Q3 표 (노이즈별)**: λ별 A, J(seed별, pooled), L의 양(음)의 기울기 비율 / 분산비 / p.
  4. **Q4**: λ*에서 A vs REF의 KS, REF 양의 기울기 비율.
  5. §2.2 판정값과 outline §8.2에서 선택한 해석 행.
  6. 실행 메타: code commit, GPU, torch/CUDA, 단계별 소요 시간.

### 3.13 `kexp/runmeta.py` (신규)
Kronos-investing의 `checksum.py`, `verify_run.py`, `env_info.py` 역할을 하나로 합친다.
CLI: `python -m kexp.runmeta {write-manifest,checksum-write,checksum-verify,verify-run} --run-id ID [--plan fast]` (`PYTHONPATH=experiment/code`)

- `manifest.json`: run_id, plan, model, noises, status, code_commit, Kronos 디렉토리 커밋(`kronos_dir_commit`), 모델·토크나이저 id와 revision, `Config.to_dict()`(`config`)와 해시, `env`(torch/CUDA/GPU/드라이버/python/pod_id), 시작·갱신·종료 시각, 단계별 소요 시간(`steps`). 러너가 단계마다 갱신한다.
- `checksums.json`: `<OUT>` 아래 모든 파일(`logs/`, `.done/`, `checksums.json` 자신 제외)의 이름·바이트·sha256. verify는 Kronos-investing과 같은 MISSING / SIZE / SHA256 / UNLISTED / COUNT 검사를 한다. `cloud_run.json`도 대상이므로 `runpod.sh`는 최종 상태를 기록한 뒤(wrap-up)에 `checksum-write`를 실행한다.
- `verify-run --plan fast`: 노이즈별 필수 파일이 있는지 확인한다. 모델 이름과 노이즈 목록은 `manifest.json`에서 읽는다 (없으면 base, ou·rw).
  - `data/v2/<n>/{base,trend}.npz`
  - `results/v2/<n>_<model>/{ldr.npz, stage3_summary.json, stage3_controls.json, steering.npz, stage4_summary.json, steer/results.json, stage6_summary.json}`
  - `fast_summary.{md,json}`
  - `steer/results.json`의 키가 5팔 × 5λ + REF 1 = 26개인지, slopes에 NaN이 없는지
- `verify-run --plan smoke`: 같은 검사를 ou만, `steer/results_smoke.json`과 키 3팔 × 2λ + REF 1 = 7개로 한다.

### 3.14 구현 중에 더한 것 (위 절에 없던 변경)

로컬 검증에서 필요가 드러나 넣었다. 실험 로직은 바꾸지 않는다.

| 위치 | 변경 | 이유 |
|---|---|---|
| `stage0` | `--model` 인자 | 러너가 small로 스모크를 돌릴 수 있게 |
| `stage3` | `--smoke`에서 exit 0 (§3.7) | 표본 8개의 null 대비 판정이 스모크 계획을 멈추지 않게 |
| `stage6` | OOD 패널의 λ 격자 (§3.10) | 1차 코드가 fast 격자에서 그림 단계에서 죽는다 |
| `run_plan.py` | 활성화가 없으면 Stage 2 마커 무시 (§3.11) | pod가 바뀌면 컨테이너 디스크의 활성화가 사라진다 |
| `runpod.sh` | `checksum-write`를 wrap-up으로 (§3.13) | 체크섬 대상인 `cloud_run.json`이 끝에 한 번 더 바뀐다 |
| `local.sh` | 커밋이 없는 저장소에서도 `push-code`·`status` 동작 | 이 레포의 첫 push |
| `stage7` | outline §8.1 문장의 선택 임계 `Q1_RATIO` (§2.2) | spec에 수치 규칙이 없었다 |
| `randomize_model`, `stage2`, `stage3`, 러너 | 초기화 방식 둘(`kronos` 판정 기준, `torch` 참고) (§3.3) | 원래 명세의 `reset_parameters()`만으로는 대조군이 토큰 임베딩에 가까웠다 |

---

## 4. RunPod 이식 (Kronos-investing `RunPod/` 방식)

### 4.1 가져올 파일과 변경점

| 파일 | 변경 |
|---|---|
| `RunPod/README.md` | 이 레포용으로 다시 쓴다. §0 실행 루프, 사양, 환경변수, 문제 해결 표 구조는 유지한다. "Claude 세션은 이 문서를 먼저 읽는다" 문구도 유지한다 |
| `RunPod/runpod.sh` | ① `main` → `$MAIN_BRANCH`(기본 `main`) ② `pred_dir()` → `experiment/runs/$RUN_ID` ③ 입력 업로드·sha256 검사 단계 삭제 (합성 데이터는 pod에서 생성하고 Stage 1 fingerprint가 검사를 대신함) ④ GPU 스모크 → `experiment/runs/smoke_gpu` 삭제 후 `python experiment/code/run_plan.py --plan smoke --run-id smoke_gpu --model base` ⑤ `RUN_CMD` 기본값 → `python experiment/code/run_plan.py --plan ${PLAN:-fast} --run-id $RUN_ID --model base` ⑥ `verify_run`·`checksum` → `kexp.runmeta` (`checksum-write`는 wrap-up에서 cloud_run.json의 최종 상태를 쓴 뒤 실행하고, 콘솔 로그 사본은 `<OUT>/logs/runpod_console.log`에 둔다) ⑦ 수정 파일 검사와 `CODE_COMMIT` 계산의 제외 경로 `data` → `experiment/runs` ⑧ `KEXP_OUT`, `KEXP_SCRATCH=/root/kexp_scratch/$RUN_ID`, `PYTHONPATH=experiment/code` export ⑨ `PROFILE`·`INFER_ARGS`·`INPUTS_DIR` knob 삭제, `PLAN` knob 추가 |
| `RunPod/setup_runpod.sh` | ① Kronos 코드 checkout 단계 삭제 (vendored), 대신 `Kronos/model/kronos.py` 존재 확인 ② `requirements-infer.txt` → `requirements-probe.txt` ③ 모델 다운로드 핀을 `kexp.config`에서 읽음 ④ torch 핀·CUDA 확인 유지 ⑤ `VENV_DIR` 기본값 `/workspace/venv-probe` |
| `RunPod/push_meta.sh` | 고정 `META_FILES` 4개 → §4.4 화이트리스트 글롭. 파일당 1MB 초과는 제외 |
| `RunPod/local.sh` | ① `main` → `$MAIN_BRANCH` ② `upload` 서브커맨드 삭제 ③ `pred_rel` → `experiment/runs/<RUN_ID>` ④ checksum verify → `PYTHONPATH=experiment/code python -m kexp.runmeta checksum-verify` ⑤ `REMOTE_REPO` 기본값 `/workspace/<origin 레포 이름>` (= `/workspace/Kronos-latent-space`) ⑥ 나머지(push-code, list, fetch, same, merge, drop, terminate, watch, ssh, status)는 그대로. 단 `push-code`·`status`는 커밋이 하나도 없는 저장소에서도 동작하게 했다 (첫 push) |
| `RunPod/inputs.sha256.json` | 가져오지 않는다 |
| `requirements-probe.txt` (신규, 루트) | Kronos-investing 핀(torch 2.14.1, huggingface_hub 2.1.1, einops 0.8.2, safetensors 0.8.0, tqdm 4.66.4, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, matplotlib 3.10.0) + scikit-learn 1.9.0, seaborn 0.13.2 (로컬 venv 버전으로 핀) |
| `colab_control.py` | `legacy/`로 이동 |
| `.gitignore` | `experiment/runs/` 추가 (push_meta는 화이트리스트만 `-f`로 추가) |

Kronos-investing의 규칙은 그대로 따른다: 코드는 로컬 `main`에서만 고치고 `push-code`로 올린다, pod에서는 코드를 고치지 않는다, RUN_ID는 덮어쓰지 않는다, 로컬 checksum verify 전에는 pod를 종료하지 않는다.

### 4.2 저장 배치

| 위치 | 내용 | 크기 |
|---|---|---|
| 네트워크 볼륨 `/workspace` (Kronos-investing 볼륨 재사용) | `/workspace/Kronos-latent-space` (레포 + `experiment/runs/<RUN_ID>`), `/workspace/venv-probe`, `/workspace/hf` (HF 캐시 공유. 같은 revision의 Kronos-base·토크나이저가 이미 있음) | 증가분 약 10GB (venv 약 8GB + runs 수백 MB). 배포 전에 볼륨 여유를 확인하고, 부족하면 새 볼륨을 만든다 |
| 컨테이너 디스크 `/root/kexp_scratch` | 활성화 scratch | 노이즈당 약 39GB + randinit band 약 1.4GB. 노이즈별로 Stage 4 뒤 삭제하므로 최대 약 42GB |

- venv는 Kronos-investing의 `/workspace/venv`와 분리한다. 그쪽 핀을 건드리지 않기 위해서다. 첫 설치는 약 13분 걸리고 이후에는 볼륨에 남는다.

### 4.3 Pod 사양

- 템플릿 `kronos-probe` = 기존 `kronos-infer` 복제. **Container Disk 20 → 60GB**만 바꾼다. 이미지, `/workspace` 마운트, TCP 22, 환경변수(`GITHUB_TOKEN`, `PUBLIC_KEY`)는 같다.
- `GITHUB_TOKEN`(fine-grained PAT)의 Repository access에 `jinyoung924/Kronos-latent-space`를 추가한다 (Contents: Read and write).
- 배포: Secure Cloud, RTX 4090 × 1, On-Demand, 네트워크 볼륨 연결.

### 4.4 결과 브랜치에 올라가는 파일 (`push_meta.sh` 화이트리스트)

`experiment/runs/<RUN_ID>/` 기준, 각 1MB 이하만 올린다.

```
cloud_run.json  manifest.json  checksums.json  requirements.lock.txt
fast_summary.md  fast_summary.json
results/v2/*/stage3_summary.json   results/v2/*/stage3_controls.json
results/v2/*/stage4_summary.json   results/v2/*/stage6_summary.json
results/v2/*/stage6_summary.csv    results/v2/*/steer/results.json
figs/v2/*.png
```

npz, `data/`, `logs/`는 GitHub에 올리지 않고 `fetch`(rsync)로 회수한다.
따라서 **`fetch` 전에도 GitHub의 `results/v2_fast` 브랜치에서 `fast_summary.md`와 그림을 바로 볼 수 있다.**

### 4.5 실행 루프

```
로컬 (main)                                       pod (/workspace = 네트워크 볼륨)
① bash RunPod/local.sh push-code "v2 fast track"
② (웹) kronos-probe 템플릿으로 배포
                                                  ③ 첫 pod 한 번: clone (README §1.3 방식, 토큰은 .git/config에 남기지 않음), tmux
                                                  ④ RUN_ID=v2_fast bash RunPod/runpod.sh
                                                     setup → GPU 스모크 → OU → 중간 push → RW → verify → checksum → push
⑤ caffeinate -i bash RunPod/local.sh watch v2_fast
   (OU 요약은 중간 push 시점에 GitHub results/v2_fast 브랜치에서 먼저 확인)
⑥ bash RunPod/local.sh merge v2_fast
```

---

## 5. v1 기준값 (`experiment/expected/v1_baseline.json`)

출처: `0812_1차실험_중간보고_stage0-4.md`, `0812_1차실험_구현단계.md`, `CLAUDE.md`. 에이전트가 이 파일을 작성해 커밋한다.

```json
{
  "source": "v1 (0812, Colab) reports",
  "ou": {
    "input_ldr": 19.57,
    "ldr_heldout": {"layer0_mid": 22.99, "layer1_mid": 29.27, "layer11_late": 23.76, "layer11_last": 23.52},
    "null_ratio": 987,
    "peak_t_range": [259, 350],
    "cos_mean_median": 0.978,
    "cos_lda_vs_median": 0.040,
    "steer_A": {"lambda_up_min": 0.25, "frac_positive_at_up": 1.0, "var_ratio": 0.046},
    "baseline_frac_positive": 0.469,
    "ref_frac_positive": 0.0
  },
  "rw": {
    "input_ldr": 2.22,
    "null_ratio": 632,
    "peak_t": 511,
    "cos_mean_median": 0.986,
    "cos_lda_vs_median": 0.022,
    "ref_frac_positive": 0.531
  }
}
```

Q0 재현 판정은 outline §6 기준을 따른다. Stage 5 항목은 subset32로 비교하고, "팔별 상향/역방향/무효 분류와 유의성이 같은가"를 본다.

---

## 6. 로컬 검증 (push-code 전 필수)

환경: 기존 로컬 venv(`~/.venvs/kronos`, Python 3.11, torch 2.13.0) + pytest, Kronos-small. 로컬 venv의 버전은 pod의 `requirements-probe.txt` 핀과 다르다. 로컬은 버그 검출용이고, pod가 같은 데이터를 만드는지는 Stage 1의 지문 대조가 확인한다.

### 6.1 단위 테스트 (`experiment/code/tests/`, pytest)

`test_kexp.py`(항목 1, 2, 3, 4, 5, 7)와 `test_stages.py`(항목 6, 8)에 12개 테스트로 구현했다. 실행: `~/.venvs/kronos/bin/python -m pytest experiment/code/tests -q`

1. `paths`: 환경변수 설정/미설정 시 루트가 올바른지
2. 랜덤 payload: 위치별 노름 = S 노름 (상대오차 < 1e-5), 같은 seed는 동일하고 다른 seed는 다름, `|cos(R, S)|` 평균 < 0.1
3. 역방향 payload = −(A payload)
4. `Steerer` + 랜덤 payload가 AR 롤링에서 기존 위치 정합 규칙을 따르는지 (기존 로컬 검증 방식 재사용. 훅 등록 순서 주의, `CLAUDE.md` §5)
5. `randomize_model`: 모델 파라미터 해시가 바뀌고 토크나이저 해시는 그대로인지, 1차원 weight = 1인지
6. eval chunking: `eval_batch = n_eval`일 때 분할 없는 기존 경로와 결과가 같은지 (같은 seed)
7. `runmeta`: checksum write → verify 통과, 파일 1바이트 변경 시 실패
8. Stage 6 fast 판정: 합성 `results.json` 픽스처로 `direction_specific` / `generic_perturbation` / `no_effect` / `reversible` 경로가 각각 나오는지

### 6.2 로컬 end-to-end 스모크
```bash
PYTHONPATH=experiment/code python experiment/code/run_plan.py --plan smoke --run-id local_smoke --model small
PYTHONPATH=experiment/code python -m kexp.runmeta verify-run --run-id local_smoke --plan smoke
```
`fast_summary.md`가 생성되고 verify-run이 통과해야 한다.

### 6.3 커밋할 기대값
- `python experiment/code/stages/stage1_dataset.py --noise both --write-expected` (전체 n=2048, CPU) → `experiment/expected/fingerprints.json`. 현재 값은 ou `d8e4765c83adf538`, rw `6bdc0a9a4f572cd5`이고, 1차 코드로 생성한 값과 같다
- `experiment/expected/v1_baseline.json` (§5)

### 6.4 스크립트 점검
- `bash -n RunPod/*.sh`
- `bash RunPod/local.sh status` 동작 확인

**small 결과로 base 실험 설계를 바꾸지 않는다** (`CLAUDE.md` 원칙). 로컬은 버그 검출용이다.

---

## 7. 실행 순서와 예상 시간

추정 근거는 v1 실측(Stage 2: T4 약 8.6분 / 노이즈, Stage 5: L4 기준 n=32에 약 1.6분 / 조합)뿐이다. 4090 실측이 없으므로 러너가 출력하는 ETA로 보정하고, 실측값은 `RunPod/README.md`에 기록한다.

| 구간 | 예상 |
|---|---|
| 첫 환경 설치 (venv-probe) | 약 13분 (1회) |
| GPU 스모크 | 수 분 |
| 노이즈당 Stage 1~4 + 대조군 | 약 20~30분 |
| 노이즈당 Stage 5 (generate 22회 × n=64) | 약 45~65분 |
| 노이즈당 합계 | 약 1~1.5시간 |
| **전체 (OU → RW)** | **약 2.5~3.5시간, $0.74/시간 기준 약 $2~3** |
| OU 요약 확인 가능 시점 | 시작 후 약 1.5시간 |

더 빠르게 하려면 `v2_fast_ou`, `v2_fast_rw`를 두 pod에서 `--noise`를 나눠 병렬로 돌릴 수 있다. 같은 GPU 종류를 쓰고, 같은 네트워크 볼륨을 두 pod에 동시에 연결할 수 있는지는 배포 전에 확인한다. 안 되면 순차로 돌린다.

---

## 8. 산출물 레이아웃

```
experiment/runs/v2_fast/                 # KEXP_OUT (pod: 네트워크 볼륨 위)
├── cloud_run.json  manifest.json  checksums.json  requirements.lock.txt
├── fast_summary.md  fast_summary.json
├── data/v2/{ou,rw}/{base,trend}.npz, timestamps.npy, meta.json
├── results/v2/{ou,rw}_base/
│   ├── ldr.npz  class_stats_layer*.npz  stage3_summary.json  stage3_controls.json
│   ├── steering.npz  pca_subset.npz  stage4_summary.json
│   ├── steer/results.json
│   └── stage6_summary.json  stage6_summary.csv
├── figs/v2/*.png
├── logs/<step>.log, logs/runpod_console.log
└── .done/<step>
/root/kexp_scratch/v2_fast/activations/v2/{ou,rw,ou_randinit,rw_randinit}/base/{base,trend}/layer*.npy   # 컨테이너 디스크, 노이즈별 삭제
```

---

## 9. 완료 기준

- [x] §6 단위 테스트와 로컬 스모크 통과, 기대값 파일 커밋 (2026-10-05)
- [ ] pod 실행 `status: ok`, 로컬 `fetch` checksum verify 통과, `merge` 완료
- [ ] `fast_summary.md`가 Q0~Q4에 답하고, 해석 문장이 outline §8 표에서 선택돼 있음
- [x] `CLAUDE.md` 갱신: RunPod 실행법, v2 진행 상황, 새로 밟은 지뢰 (pod 실행 뒤 다시 갱신)
- [ ] `RunPod/README.md`에 4090 실측 시간·비용 기록

---

## 10. Tier 2 (이 spec 범위 밖)

fast 결과를 보고 우선순위를 정한다. 상세 설계는 outline §5.

- v1 나머지 팔 B, C, D~I의 v2 재실행
- K(직교화 랜덤), M(trend 입력 + 역방향)
- 샘플링 민감도 (sample_count, temperature), S 재추정 안정성, 부트스트랩 CI
- 저SNR OU, V/A 비영 변형
- 금융 stylized fact(변동성 군집) 확장, KRX 실데이터 probe