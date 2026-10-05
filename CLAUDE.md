# Kronos 개념 표현·조정 재실험 (v2 fast track) — 작업 인수인계

새 세션은 이 파일을 먼저 읽는다. 코드를 고치기 전에 §5 "이미 밟은 지뢰" 표를 반드시 읽는다.

---

## 1. 무엇을 하는가

TSFM인 **Kronos**에 **Moment Steering 논문**(Wiliński et al., ICML 2025, arXiv:2409.12915)의 방법론을 적용해, "선형 증가 모멘텀" 개념이 (1) latent space에 선형 표현으로 존재하는지, (2) `h_i ← h_i + λS_i` 개입으로 출력을 유도할 수 있는지 검증한다.

이 저장소는 1차 실험(0812, Colab, 저장소 `Kronos_probing`)의 **재실험**이다. 지금 범위는 `docs/spec.md`의 **fast track**: pod 1대, RUN_ID 1개(`v2_fast`), OU → RW 순차 실행으로 다섯 질문에 답한다.

| ID | 질문 | 판정 재료 |
|---|---|---|
| Q0 | v1(Colab) OU 결과가 재현되는가 | `experiment/expected/v1_baseline.json` vs v2 (Stage 5는 subset32) |
| Q1 | 선형 분리가 사전학습으로 생긴 것인가 | 입력 기준선 / 무작위 초기화 / 사전학습 LDR |
| Q2 | steering 효과가 S 방향에 특이적인가 | A vs J(랜덤 방향, seed 3개) |
| Q3 | 역방향(−λS)도 작동하는가 | L |
| Q4 | RW에서도 v1 OU의 결론이 유지되는가 | RW의 A, J, L, REF |

### 읽어야 할 문서 (우선순위 순)

| 파일 | 내용 |
|---|---|
| `docs/spec.md` | **fast track 구현 명세.** 고정값, 팔과 판정 규칙(§2.2, 사전 고정), 코드 변경, RunPod 이식 |
| `docs/outline.md` | 재실험 전체 설계. v1 결과 요약, 보강 실험, 결과별 허용 결론(§8). spec 밖의 것은 Tier 2 |
| `RunPod/README.md` | RunPod 실행 절차. pod와 결과 브랜치는 이 문서의 명령으로만 다룬다 |
| `experiment/0812_*.md` | 1차 실험 문서 (결과 보고서, 구현 단계, 원 명세·설계) |

`Kronos/`(모델, upstream `shiyu-coder/Kronos@67b630e`의 `model/`)와 `representations-in-tsfms/`(논문 공식 구현)는 읽기 전용 참조다.

---

## 2. 지금 어디까지 왔는가

- **코드: fast track 구현 완료, 로컬 검증 통과, origin `main`에 push됨 (2026-10-05).** pod에서는 아직 아무것도 돌리지 않았다.
- `docs/spec.md`와 `docs/outline.md`는 2026-10-05에 현재 코드에 맞춰 갱신했다. outline의 **[fast]** / **[Tier 2]** 표시가 구현 여부다.
- 로컬 검증 (macOS, Kronos-small — **코드 버그 검출용이며 과학적 결론이 아니다**)
  - 단위 테스트 12개 통과 (`experiment/code/tests`, spec §6.1의 8항목)
  - `run_plan.py --plan smoke --run-id local_smoke --model small` 끝까지 + `verify-run` 통과, `fast_summary.md` 생성
  - 비스모크 fast 경로(5팔 × 5λ + REF = 26키, REF 이어 쓰기, 재개)를 small·축소 설정으로 확인
  - `push_meta.sh` 화이트리스트, `local.sh`의 list·merge·verify를 임시 bare 저장소로 확인
- `experiment/expected/fingerprints.json`: ou `d8e4765c83adf538`, rw `6bdc0a9a4f572cd5`. 1차 코드로 뽑은 값과 같고, 입력단 LDR 19.569 / 2.220도 1차 보고서와 일치한다.
- **`RunPod/runpod.sh`, `setup_runpod.sh`는 실제 pod에서 검증되지 않았다.**

1차 실험의 결과(재현 기준값)는 `docs/outline.md` §2에 있다. 핵심: 개념의 선형 표현은 존재하고(null 대비 OU 987배), A 팔은 λ=0.25에서 32/32 양의 기울기를 만들지만 분포를 붕괴시키며(std 0.023 → 0.001), 진짜 trend 입력에 모델은 추세를 외삽하지 않는다(REF: OU 0/32, RW 53.1%).

---

## 3. 코드 구조

```
RunPod/                          runpod.sh, setup_runpod.sh, push_meta.sh (pod) / local.sh (로컬) / README.md
legacy/colab_control.py          1차의 Colab 제어 셀 (보존용)
requirements-probe.txt           pod 환경 (Kronos-investing 핀 + scikit-learn, seaborn)
experiment/expected/             fingerprints.json, v1_baseline.json (커밋된 기대값)
experiment/runs/<run_id>/        실행별 산출물 = KEXP_OUT (gitignore, 요약만 결과 브랜치로)
experiment/code/
  run_plan.py                    파이프라인 러너 (fast / smoke 계획, .done 마커, 노이즈별 요약·중간 push)
  kexp/
    config.py                    하이퍼파라미터 단일 소스. tag "v2", 모델 revision 핀, ControlCfg, fast 격자
    paths.py                     KEXP_OUT / KEXP_SCRATCH 기반 경로
    kronos_loader.py             로드(핀 전달), set_determinism, randomize_model, 전처리 재현
    intervene.py                 Steerer (AR 롤링 위치 정합), build_payload, build_random_payload
    runmeta.py                   manifest / checksum / verify-run (CLI: python -m kexp.runmeta)
    hooks.py synth.py activations.py ldr.py steering_vec.py     1차 그대로
  stages/stage{0..6}_*.py        독립 엔트리포인트 (1차 + spec §3 의 변경)
  stages/stage7_fast_summary.py  fast_summary.md / .json
  tests/                         pytest (spec §6.1)
```

**패키지 이름이 `kexp`인 이유**: `experiment/code`를 `sys.path`에 올리므로 하위 패키지가 `model`이면 Kronos의 `model`과 충돌한다. `code`도 표준 라이브러리 모듈명이다.

### 1차 대비 바뀐 것 (spec §3)

| 파일 | 변경 |
|---|---|
| `paths.py` | `in_colab()` 제거, `drive_root()` → `out_root()`, 환경변수 `KEXP_OUT` / `KEXP_SCRATCH` |
| `config.py` | tag `v2`, `ModelCfg.revision`·`tokenizer_revision`, `SteerCfg.lambdas_rel_fast`·`n_eval_fast`·`eval_batch`, `ControlCfg` |
| `kronos_loader.py` | revision 전달(base인데 핀 없으면 실패), `set_determinism()`, `randomize_model()` |
| `intervene.py` | `build_random_payload()`. 역방향은 `build_payload(..., -lam, ...)` |
| `stage1` | `--expect-fingerprint`(불일치 exit 2), `--write-expected` |
| `stage2` | `--random-init`(출력 `<noise>_randinit`), `--init {kronos,torch}`, `--positions bands`(17개 위치만 저장) |
| `stage3` | `--controls` → `stage3_controls.json`, `figs/v2/stage3_controls_<noise>.png` |
| `stage4` | 변경 없음 (1차 파일과 동일) |
| `stage5` | `--arms fast`(A, J×3, L), `--lambdas fast`, `--eval-batch`, 결과에 `subset32`·`cos_random_vs_S` |
| `stage6` | `--smoke`, `J_ctrl_random(pooled)` 행, `stage6_summary.json["fast"]`(spec §2.2 판정) |

spec에 적히지 않았지만 필요해서 넣은 것 넷: (1) stage0에 `--model`(러너가 small로 스모크를 돌릴 수 있게), (2) stage3가 `--smoke`에서는 null 대비 판정과 무관하게 exit 0, (3) stage6의 OOD 패널이 Stage 4의 λ 격자를 쓰도록 수정(§5), (4) 러너가 활성화가 사라진 경우 Stage 2 마커를 무시.

### Stage 5의 fast 팔

| 팔 | 입력 | payload |
|---|---|---|
| `A_paper_median_matrix_all` | base held-out | `λ_i · S_median_matrix` (1차 A 그대로) |
| `J_ctrl_random_s0~2` | base held-out | 레이어·위치별 `g/‖g‖ · ‖S_i^(t)‖`, `g ~ N(0, I)`, 같은 `λ_i` |
| `L_ctrl_reverse` | base held-out | `−λ_i · S_median_matrix` |
| `REF_trend_unsteered` | trend held-out | 없음 (`--reference-trend`) |

λ_rel 격자 {0, 0.1, 0.15, 0.25, 0.5}, n_eval 64, eval_batch 32. 1차의 다른 팔(B~I)은 코드에 남아 있지만 fast 계획은 돌리지 않는다(Tier 2).

### 산출물 (`experiment/runs/<run_id>/`)

```
cloud_run.json  manifest.json  checksums.json  requirements.lock.txt  fast_summary.md  fast_summary.json
data/v2/<noise>/{base,trend}.npz, timestamps.npy, meta.json
results/v2/<noise>_<model>/
  ldr.npz  class_stats_layer*.npz  stage3_summary.json  stage3_controls.json
  steering.npz  pca_subset.npz  stage4_summary.json
  steer/results.json (스모크는 results_smoke.json)
  stage6_summary.json  stage6_summary.csv
figs/v2/*.png   logs/<step>.log   .done/<step>
<KEXP_SCRATCH>/activations/v2/{<noise>,<noise>_randinit}/<model>[_smoke]/<class>/layer*.npy   (Stage 4 뒤 삭제)
```

---

## 4. 어떻게 실행하는가

### 역할 분담 (1차의 합의 유지)

- **Claude**: 코드 작성 + 로컬 검증. 커밋과 푸시(`push-code`)는 사용자가 요청할 때만 한다. **pod 조작은 하지 않는다** — 명령어만 제시한다.
- **사용자**: pod 배포와 `runpod.sh`, `watch`/`fetch`/`merge`.
- **결과 확인**: OU 요약은 중간 push 시점에 GitHub `results/v2_fast` 브랜치의 `fast_summary.md`에서 먼저 볼 수 있다. `fetch` 뒤에는 `experiment/runs/<RUN_ID>/`의 요약·그림·로그를 Claude가 직접 Read로 본다.

### RunPod (절차 전체는 `RunPod/README.md`)

```bash
bash RunPod/local.sh push-code "v2 fast track"          # 로컬
RUN_ID=v2_fast bash RunPod/runpod.sh                    # pod (tmux 안에서)
caffeinate -i bash RunPod/local.sh watch v2_fast        # 로컬: 끝나면 fetch + verify + terminate
bash RunPod/local.sh merge v2_fast                      # 로컬: 요약을 main 으로
```

push하지 않은 코드는 pod에 존재하지 않는다. `run_plan.py`는 첫 줄에 `[run_plan v2-fast]`와 코드 커밋을 찍는다.

### 로컬 검증 (push-code 전 필수, spec §6)

전용 venv `~/.venvs/kronos` (Python 3.11, torch 2.13.0, pytest 설치됨). Kronos-mini/small/base 가중치가 HF 캐시에 있다.

```bash
~/.venvs/kronos/bin/python -m pytest experiment/code/tests -q
PYTHONPATH=experiment/code ~/.venvs/kronos/bin/python experiment/code/run_plan.py --plan smoke --run-id local_smoke --model small
PYTHONPATH=experiment/code ~/.venvs/kronos/bin/python -m kexp.runmeta verify-run --run-id local_smoke --plan smoke
bash -n RunPod/*.sh && bash RunPod/local.sh status
```

스모크는 약 1.5분 걸린다. 산출물은 `experiment/runs/local_smoke`, `experiment/_scratch/local_smoke`(둘 다 gitignore). 다시 처음부터 돌리려면 둘을 지운다.
로컬 프록시는 **Kronos-small**이다(base와 토크나이저·max_context가 같아 AR 롤링 경로가 동일). mini는 `max_context=2048`이라 다른 분기를 탄다.

**주의**: Bash 도구가 2분에 타임아웃된다. 긴 실행은 `run_in_background: true`를 쓴다.

---

## 5. 이미 밟은 지뢰 (재발 방지)

위 10개는 1차 실험, 아래는 재실험 코드 작업에서 나왔다.

| 사고 | 원인 | 대응 (코드에 반영됨) |
|---|---|---|
| 본 실행이 통째로 건너뛰어짐 | 스모크와 본 실행이 같은 출력 디렉토리 사용 | 활성화는 `_smoke` 접미사, Stage 5는 `results_smoke.json`. v2는 실행마다 `experiment/runs/<run_id>`도 다르다 |
| 활성화 39GB 불필요 재계산 | 재개 판정에 **전체** config 해시 사용 | `activation_hash()`로 모델·데이터·dtype만 해싱 |
| 낡은 데이터로 만든 활성화 재사용 위험 | 생성 로직이 바뀌어도 config 값은 그대로 | Stage 1이 `data_fingerprint`를 남기고 Stage 2가 대조 |
| Stage 5 결과에 스모크 값 혼입 | 스모크가 `results.json`에 썼다 | 항목마다 `run_settings` 기록 + 이어받을 때 불일치 폐기 |
| 첫 토큰이 최고 분리도로 나옴 | OU를 영 초기조건으로 시작 | burn-in `10/θ` 스텝 폐기 |
| coherence 위반 96.88% (기준선에서) | 정규화 공간에서 검사 | 역정규화 후 검사 |
| 위치별 argmax가 노이즈 스파이크를 고름 | 512개 위치에서 raw argmax | 25점 이동평균 + 구간 평균 |
| steering 훅 테스트가 "효과 없음" | recorder 훅을 Steerer보다 **먼저** 등록 | 훅 등록 순서 주의 (테스트 4가 지킨다) |
| Stage 6이 "모든 팔 효과 없음"으로 오판 | 대응표본 평균 비교를 주 검정으로 썼다 | 주 검정을 **결과 부호의 이항검정**으로 |
| 목표 분포 없이 인과 결론을 낼 뻔함 | steering 작동만으로는 구분 불가 | `--reference-trend` 대조군 |
| Stage 6 기하 그림이 fast 격자에서 죽음 | OOD 패널이 "Stage 5의 λ 격자 = Stage 4의 λ 격자(8개)"를 가정. fast는 5개 | OOD 패널은 `steering.npz`의 `lambdas_rel`로 그린다 |
| `--reference-trend`가 앞 실행의 결과를 지울 뻔함 | `run_settings`에 `arms`·`lambdas`를 넣고 전체 일치를 요구하면 팔만 다른 실행이 "불일치"가 된다 | 이어받기 판정은 n_eval·pred_len·sample_count·eval_batch만 본다 |
| 랜덤 payload 테스트에서 cos = 1.0 | numpy는 `default_rng([0, 0])`과 `default_rng(0)`을 같은 난수열로 만든다(뒤쪽 0 무시). 테스트의 S를 seed 0으로 만들었더니 (seed 0, layer 0) 방향과 같았다 | 테스트의 S는 다른 seed. 실제 S와는 무관하다 |
| 스모크 Stage 3가 exit 1 | 표본 8개로는 null 대비 판정이 항상 불안정 | `--smoke`에서는 판정과 무관하게 exit 0 |
| 3.12 문법의 f-string이 로컬에서 SyntaxError | 로컬 venv는 **Python 3.11**, pod는 3.12 | f-string 안에서 같은 따옴표를 중첩하지 않는다. `python -m py_compile` |

### 수치·해석상 주의

- **기울기는 정규화 공간, coherence는 역정규화 공간**에서 잰다. LDR의 분산 분모는 **ddof=0**.
- λ는 **상대 강도** `λ_rel = λ_i‖S_i‖/‖h_i‖`. `pred_len`이 다르면 기울기를 비교할 수 없다.
- **`eval_batch`(32)와 `seed`를 바꾸면 샘플 값이 달라진다.** chunk k의 seed는 `seed + k`이고, chunk 0은 1차의 평가 호출과 입력·배치·seed가 같다. 그래서 `subset32`(앞 32 입력)를 v1과 직접 비교한다.
- **"분산비"(`var_ratio`)는 표준편차의 비**다(개입 후 / 기준선).
- **판정 규칙(spec §2.2)은 사전 고정이다. 결과를 본 뒤 바꾸지 않는다.** `fast_summary.md`의 해석 문장은 outline §8 표에서 선택만 한다.
- Q1의 문장 선택 기준(`stage7`의 `Q1_RATIO = 2`, 마지막 층·t=511의 비율)은 spec에 없는 값이다. 표의 비율을 직접 본다.
- **무작위 초기화 대조군은 둘이다.** 판정 기준은 Kronos 자체 초기화(`randomize_model(init="kronos")`, `<noise>_randinit`)이고, PyTorch 기본 초기화(`init="torch"`, `<noise>_randinit_torch`)는 참고 열이다. torch 방식은 임베딩이 N(0, 1)이라(Kronos 자체는 std 0.035) base 구조에서 블록이 residual에 층당 약 1%만 기여해 "위치별 토큰 임베딩"에 가까운 대조군이 된다. 원래 spec은 torch 방식만 적었고, 2026-10-05에 사용자가 둘 다 돌리되 Kronos 방식으로 판정하기로 정했다.
- **small 결과로 base 실험을 바꾸지 않는다.** 로컬은 버그 검출용이다.
- "모델이 개념을 이해한다/못한다"는 표현은 쓰지 않는다 (outline §8).

---

## 6. 다음 할 일

2. 템플릿 `kronos-probe` 배포: `kronos-infer` 복제, Container Disk 60GB, PAT에 이 레포 추가 (`RunPod/README.md` §1)
3. pod: `RUN_ID=v2_fast bash RunPod/runpod.sh` → 로컬 `watch` → `merge`
4. `fast_summary.md`로 Q0~Q4 확인. 실측 시간·비용을 `RunPod/README.md` §5에 기록
5. Tier 2(spec §10)는 fast 결과를 보고 정한다: B~I 재실행, K·M, 샘플링 민감도, S 안정성, 부트스트랩 CI, 저SNR·V/A 변형, GARCH·KRX

---

## 7. 사용자 작업 방식

- 한국어로 소통한다.
- **결정이 필요하면 필요한 정보를 먼저 요구하고 각 정보를 본 뒤 논리적으로 판단**하기를 원한다.
- 원논문의 방법을 **유지**하면서 개선안·대조군을 **추가**하는 방식을 선호한다.
- 근거 없는 추론을 경계한다. 그림·수치는 직접 확인시켜 주기를 원한다.
- spec의 구현 범위를 넘지 않는 컴팩트한 구현을 원한다. Tier 2 기능을 미리 넣지 않는다.
