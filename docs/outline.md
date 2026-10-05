# Kronos-base 내부 표현 분석 재실험 Outline (v2)

> 방법론 논문: Wiliński et al., *Exploring Representations and Interventions in Time Series Foundation Models* (ICML 2025, arXiv:2409.12915)
> 대상 모델: `NeoQuasar/Kronos-base` + `NeoQuasar/Kronos-Tokenizer-base`
> 기존 실험 코드: https://github.com/jinyoung924/Kronos_probing (1차 실험 "0812", Colab 실행)
> 재실험 코드: https://github.com/jinyoung924/Kronos-latent-space (이 저장소. 1차 코드를 옮겨 와 수정)
> 작성: 2026-10-04 · 개정: 2026-10-05 (현재 코드에 맞춰 갱신)
>
> **구현 범위.** 지금 코드가 구현한 것은 이 설계의 일부인 **fast track**(`docs/spec.md`)이다. RUN_ID 하나로 OU → RW를 순차 실행해 재현(A 팔), 표현 대조군, 랜덤 방향(J), 역방향(L), REF를 본다. 아래에서 **[fast]** 는 구현된 항목이고 **[Tier 2]** 는 설계만 있고 코드가 없는 항목이다. pod에서는 아직 실행하지 않았다.

---

## 0. 재실험의 목적과 범위

1차 실험(0812)은 OU 데이터셋에서 Stage 0~6, RW 데이터셋에서 Stage 1~4와 목표 분포 대조군(REF)까지 완료했다. 다만 Colab 세션과 Google Drive에 의존해 실행 로그와 중간 산출물을 다시 확인하기 어렵다. 재실험은 실행 환경을 RunPod로 옮기고 아래 세 가지를 수행한다.

1. **재현** — 1차 실험의 핵심 수치(§2)를 기준값으로 삼아 같은 코드·같은 시드로 다시 산출하고, 결론 수준에서 일치하는지 확인한다.
2. **완결** — 1차에서 미실행으로 남은 RW 개입 팔(Stage 5·6)을 실행한다.
3. **보강** — 1차 설계에 없던 대조군(랜덤 방향, 역방향, 무작위 초기화 모델 등, §5)을 추가해 인과 해석을 확정한다.

작업 원칙 (1차 실험 `CLAUDE.md` §7의 합의 유지)
- 원논문 방법은 유지하고, 데이터에서 발견한 개선안·대조군은 **추가**한다. 대체하지 않는다.
- Kronos-small(로컬)은 코드 버그 검출용이다. **과학적 결정은 base 실측으로만** 한다.
- 결정이 필요한 지점에서는 필요한 정보를 먼저 출력해 확인한 뒤 판단한다.
- 코딩 에이전트는 작업 전 `CLAUDE.md` §5 "이미 밟은 지뢰" 표를 반드시 먼저 읽는다.

---

## 1. 이미 확정된 Kronos 구현 사항 (재조사 불필요)

1차 실험 Stage 0 실측과 코드 분석으로 해소된 항목이다. 출처: `experiment/0812_1차실험_중간보고_stage0-4.md` §1, `CLAUDE.md` §5.

| 항목 | 확인 결과 |
|---|---|
| 구조 | Kronos-base: 12층, d_model 832, d_ff 2048, 16 heads, max_context 512, 102.3M |
| hook 지점 | `model.transformer[i]` forward hook 출력 = FFN 이후 residual stream. 모델 개조 불필요 |
| 위치 임베딩 | RoPE만 사용. 절대 위치 임베딩 없음 → max_context는 런타임 인자 |
| 토큰화 | BSQ 토크나이저가 coarse/fine 두 서브토큰 생성 → 한 시퀀스 위치로 융합 |
| 전처리 | 채널별 z-score(`np.std`, ddof=0) 후 [-5, 5] clip. V/A=0이면 0 유지(NaN 없음) |
| 시간 특징 | TemporalEmbedding 사용 → 두 클래스 timestamp를 동일하게 고정 |
| AR 생성 | 매 디코딩 스텝 `decode_s1` 재호출 → hook이 스텝마다 발화. T = max_context = 512이면 첫 스텝부터 롤링 버퍼(위치가 한 칸씩 이동) |
| sample_count | `auto_regressive_inference` 내부에서 sample_count개 예측을 **평균**해 반환 |
| eval 모드 | TransformerBlock dropout 때문에 필수. 매 stage 결정성 검증 |
| 측정 공간 | 기울기는 정규화 공간, OHLC coherence는 역정규화 공간에서 측정 |

---

## 2. 1차 실험 결과 요약 = 재현 기준값

출처: `0812_1차실험_중간보고_stage0-4.md`, `0812_1차실험_구현단계.md`, `CLAUDE.md`. 모두 Colab 실행분이므로 재확인 대상이다.

### 2.1 표현 (Stage 1~4)

| 항목 | OU (주) | RW (robustness) |
|---|---|---|
| 입력단 LDR (close 기울기 1차원) | 19.57 | 2.22 |
| 층 held-out LDR (layer 11, 마지막 토큰 부근) | 약 23.5 | 약 15~16 |
| null(라벨 셔플) 대비 배율 | 987배 | 632배 |
| 평활 LDR 정점 위치 | t ≈ 259~350 | t = 511 |
| cos(mean, median) | 0.978 | 0.986 |
| cos(LDA 방향, 평균차 방향) | 0.040 | 0.022 |
| 상위 2 PC 안의 LDA 방향 비중 | 0.01% | 0.01~0.02% |
| ‖S‖/‖h‖ | 0.44~0.60 | 0.66~0.78 |
| layer 0 OOD 취약성 | λ_rel ≈ 0.06에서 이탈 | 동일 경향 |

### 2.2 개입 (Stage 5·6, OU만 실행; n_eval=32, pred_len=64, sample_count=5)

- A/B/C(논문 방법): λ_rel = 0.25에서 **32/32 양의 기울기** (이항검정 p = 4.7e-10), coherence 위반 최대 0.098%
- 기울기 분포 붕괴: std 0.023 → 0.001. 개입이 drift를 "더하는" 것이 아니라 출력을 "덮어쓴다"
- C(single-token) ≈ A(all-tokens) — 논문(인코더 모델)과 반대 결과
- F(layer {1,2}, 전역 LDR 최대 지점): λ_rel = 0.35에서 **0/32** — 역방향 조정
- G/H(LDA 방향): 효과 없음
- I(median vector 형태): λ_rel = 0.05에서 이미 65.6%

### 2.3 목표 분포 대조군 (REF)

| | slope | 양의 기울기 비율 |
|---|---|---|
| OU REF (진짜 trend 입력, 개입 없음) | −0.0656 | 0/32 |
| RW REF (진짜 trend 입력, 개입 없음) | −0.0113 | 53.1% |
| 기준선 (base 입력, 개입 없음) | −0.0011 | 46.9% |
| steered base (OU, λ_rel = 0.25) | +0.0038 | 100% |

KS 검정: steered 출력이 REF 분포와 닮은 (팔, λ) 조합이 없음 (모두 p ≈ 1e-18).

### 2.4 1차 실험의 잠정 결론

- 추세와 상수를 구분하는 **선형 방향은 존재**한다.
- 평균차 방향 개입으로 출력 기울기를 **양수로 만들 수 있다**.
- 그러나 모델은 실제 추세 입력에 대해 추세를 외삽하지 않으며(RW 53.1%), steered 출력은 목표 분포와 다르다.
- 따라서 "steering이 작동한다 ⇒ 모델이 그 개념을 기저 기작으로 쓴다"는 추론은 성립하지 않는다. RW 개입 팔이 확정의 관건이다.

### 2.5 기준값에서 새로 읽어야 할 점

- OU는 입력단 LDR(19.57)과 층 LDR(약 23.5)이 비슷하다 → OU의 "선형 표현"은 입력 특성이 거의 그대로 통과한 것일 수 있다.
- RW는 2.22 → 약 15로 **약 7배 증폭**된다 → 모델이 문맥 누적으로 분리를 만들어낸다는 해석이 가능하다.
- 단, 입력단 LDR은 1차원(close 기울기) 기준이고 층 LDR은 832차원 LDA 기준이라 직접 비교할 수 없다. §5.3에서 같은 절차로 다시 잰다.

---

## 3. 기존 코드 재사용 지도

### 3.1 그대로 가져갈 것

| 파일 | 역할 | 비고 |
|---|---|---|
| `kexp/config.py` | 하이퍼파라미터 단일 소스, 부분 해시(`activation_hash`, `hash_for`) | 재실험 tag를 `v2`로 분리해 v1 산출물과 섞이지 않게 함. 모델 revision 핀, fast λ 격자, `ControlCfg` 추가 |
| `kexp/kronos_loader.py` | 로드, 스펙 실측, 전처리 재현(`normalize_batch`, `to_tokens`, `make_stamps`) | Stage 2와 5가 같은 전처리를 공유. 핀 전달, `set_determinism`, `randomize_model` 추가 |
| `kexp/synth.py` | micro-path → OHLC 집계, 노이즈 공유, OU burn-in, SNR 파라미터화 | v1 outline의 합성 데이터 설계를 대체(더 정교함). coherence 자동 충족 |
| `kexp/activations.py` | 레이어별 memmap 입출력 | 저장 위치 선택(`positions`) 추가 |
| `kexp/hooks.py` | `ActivationRecorder` | `steering()` 컨텍스트는 `Steerer`로 대체된 레거시 |
| `kexp/ldr.py` | 배치화 closed-form LDA, ddof=0, shrinkage, held-out, null | steertool과 수치 대조 완료 |
| `kexp/steering_vec.py` | S(median/mean/lda × matrix/vector), λ_rel 캘리브레이션, OOD 비율 | 기본 집계 구간 `late` 유지 |
| `kexp/intervene.py` | `Steerer`: AR 롤링 위치 정합, matrix/vector, all/single | 신규 대조군 payload만 추가 (`build_random_payload`) |
| `stages/stage0~6_*.py` | 독립 엔트리포인트, `--smoke`/`--force`, 증분 저장·재개, `run_settings` 검증 | |
| `stage6_report.py` 판정 | 결과 부호 이항검정, 분산비, REF 대비 KS | 대응표본 검정이 무력했던 이유(분포 붕괴)가 docstring에 기록됨 |
| `representations-in-tsfms/` | 논문 공식 구현(steertool) | 읽기 전용. 수치 대조용 |

### 3.2 교체할 것 (Colab 의존부)

- `colab_control.py` → `experiment/code/run_plan.py`(파이프라인 러너) + `RunPod/runpod.sh`(pod 진입점). 기존 파일은 `legacy/colab_control.py`로 옮겼다
- `kexp/paths.py`: `in_colab()`을 없애고 환경변수 `KEXP_OUT` / `KEXP_SCRATCH`로 보존/scratch 루트를 결정한다. 없으면 로컬 경로(`experiment/_out`, `experiment/_scratch`)를 쓴다. `drive_root()`는 `out_root()`가 됐다
- 문서의 Drive 경로·Colab 실행법: `CLAUDE.md`를 새로 썼고, `0812_1차실험_구현단계.md` 상단에 새 이름과의 대응을 적었다
- 그림 확인 흐름(`experiment/confirm_data/`): 요약과 그림은 결과 브랜치(`results/<RUN_ID>`)에서 바로 보고, 전체는 `RunPod/local.sh fetch`로 `experiment/runs/<RUN_ID>/`에 회수한다

### 3.3 확장할 것 (§5 보강 실험용)

- `stage5_intervene.py`: **[fast]** 신규 팔 J(seed 3개)·L, `--arms fast`, `--lambdas fast`, `--eval-batch` / **[Tier 2]** K·M, 팔별 n_eval 지정
- `stage2_activations.py`: **[fast]** `--random-init`, 축소 저장 `--positions bands`(17개 위치)
- `stage3_ldr.py`: **[fast]** `--controls` — 입력단 기준선, 무작위 초기화, 사전학습 LDR을 같은 위치·같은 절차로 비교 (`stage3_controls.json`)
- `stage6_report.py`: **[fast]** J pooled 행, L 판정, 사전 고정 판정(`stage6_summary.json["fast"]`) / **[Tier 2]** 부트스트랩 CI
- `stage7_fast_summary.py` (신규): **[fast]** Q0~Q4 요약. 재현 대조표와 입력·무작위 대비 비율 표가 여기에 있다
- `config.py`: **[fast]** tag `v2`, 모델 revision 핀, `ControlCfg`(랜덤 seed, band 위치), fast λ 격자 / **[Tier 2]** 샘플링 민감도 격자

---

## 4. 실행 계획

각 단계는 1차와 같은 순서를 유지한다: **로컬(Kronos-small, 축소 표본) → RunPod 스모크(`--smoke`) → RunPod 본 실행**.

| Phase | 내용 | 진행 조건 |
|---|---|---|
| P0 이식 검증 | Stage 0 → §1 스펙 일치 / Stage 1 → `data_fingerprint`가 v1과 동일(같은 seed) / Stage 2 결정성 True | 셋 다 통과해야 P1 진행 |
| P1 OU 재현 | Stage 1~6 (OU) 전체 재실행 | §6 기준으로 §2와 결론 수준 일치 |
| P2 RW 완결 | Stage 5 `--arms paper` → `--arms ours` → Stage 6 (RW) | REF는 재실행 후 재사용 |
| P3 보강 대조군 | §5.1~5.5 | P1·P2 결과 확인 후 |
| P4 강건성 | §5.6 저SNR OU, V/A 비영 변형 | P3 결과 확인 후 |
| P5 (선택) 확장 | §5.6 금융 stylized fact, 실데이터 probe | — |

**현재 구현된 실행 (fast track).** 위 Phase를 따로 돌리지 않고 RUN_ID 하나(`v2_fast`)로 OU → RW를 순차 실행한다 (`run_plan.py --plan fast`, `docs/spec.md` §3.11). Phase와의 대응은 다음과 같다.

| Phase | fast track에서 하는 것 | 남는 것 (Tier 2) |
|---|---|---|
| P0 | stage 0(스펙·결정성), stage 1 `--expect-fingerprint`, 본 실행 전의 GPU 스모크(smoke 계획 전체) | — |
| P1 | OU stage 1~6을 A 팔로. v1과의 비교는 앞 32표본(subset32) | B~I 팔 |
| P2 | RW stage 1~6을 A·J·L·REF로 | B~I 팔 |
| P3 | §5.1의 J(seed 3개), §5.2의 L, §5.3 전부, §5.4 중 n_eval 64 | K, M, n_eval 128, 부트스트랩 CI, S 추정 안정성, §5.5 |
| P4, P5 | — | 전부 |

---

## 5. 보강 실험 설계

### 5.1 랜덤 방향 대조군 (최우선)

**왜 필요한가.** 1차 결과에서 개입은 기울기 분포를 붕괴시켜 "전부 작은 양수"로 덮어썼다. 같은 크기의 임의 섭동도 비슷한 정형화된 출력을 만든다면, 효과는 추세 방향이 아니라 "큰 섭동 일반"의 성질이다. 이 대조군 없이는 개념 특이성을 주장할 수 없다.

- **J** (`J_ctrl_random_s{seed}`) **[fast]**: 레이어·위치별로 `S_i^(t)`와 같은 L2 노름을 갖는 가우시안 랜덤 방향. A와 같은 λ_rel 격자, all-tokens·전체 층. fast track은 seed 3개(0, 1, 2)이고, 5개 이상은 Tier 2다
- **K_ctrl_orthogonal** **[Tier 2]**: 랜덤 방향에서 S 성분을 제거(직교화)한 뒤 같은 노름으로 맞춘 방향. D=832에서 랜덤 방향과 S의 코사인은 약 0.035라 J가 사실상 직교 대조군 역할을 하므로 fast track에서는 뺐다. 실측 코사인은 결과의 `cos_random_vs_S`에 남긴다
- 판정 규칙은 결과를 보기 전에 고정했다 (spec §2.2: `direction_specific` / `generic_perturbation` / `collapse_generic`)
- 보고 지표를 둘로 분리한다: (a) 양의 기울기 비율, (b) 분산비(붕괴 여부)
- 해석 예: 붕괴는 J/K에서도 일어나는데 부호가 무작위라면 → "붕괴는 섭동 일반의 성질, 양의 부호는 S 방향 특이적"

### 5.2 역방향 개입

논문 Appendix D는 −λS가 개입 방향을 반전시킨다고 보고한다.

- **L** (`L_ctrl_reverse`) **[fast]**: base 입력 + (−λS) → 음의 기울기 비율. 판정은 spec §2.2의 `reversible`
- **M_reverse_trend** **[Tier 2]**: trend 입력 + (−λS) → REF 대비 기울기 감소. OU REF는 이미 음수(0/32)라 주로 RW에서 해석
- 대칭성이 깨지면(상향만 되고 하향은 안 됨 등) 그 자체를 결과로 보고

### 5.3 표현 대조군 (Stage 2·3)

- **무작위 초기화 Kronos-base** **[fast]**(같은 구조, 같은 사전학습 토크나이저)로 Stage 2·3 → 층별 LDR. 사전학습 모델이 이보다 높아야 "학습된 표현"이라 할 수 있다
  - 구현(`randomize_model`)은 PyTorch 기본 `reset_parameters()`를 쓴다. 이 초기화에서는 임베딩이 블록 출력보다 훨씬 커서 residual이 층을 지나도 거의 변하지 않는다(base 구조 실측: 블록 기여가 층당 약 1%). 대조군이 "학습되지 않은 트랜스포머"보다 "위치별 토큰 임베딩"에 가깝다는 점을 감안해 읽는다 (spec §3.3의 알려진 한계)
- **입력단 기준선 통일** **[fast]**: 위치 t까지의 정규화 close 경로(차원 t+1)에 층 LDR과 같은 LDA·held-out·null 절차를 적용 → 데이터셋별 **입력 대비 비율**(`pretrained_over_input`) 산출
- 저장 축소 **[fast]**: 대조군은 17개 band 위치(0, 32, …, 480, 511)만 저장해 39GB 재생성을 피한다

### 5.4 통계 강화

- n_eval **[fast]**: 전 팔 64. 입력 32개씩 두 번에 나눠 생성하고 k번째 묶음의 seed는 seed + k다. 앞 32개는 v1과 입력·배치·seed가 같아 `subset32` 지표로 v1과 직접 비교한다
- n_eval **[Tier 2]**: 주요 팔(A, I, J, L, REF)은 **128 이상**. 실행 시간은 표본 수에 대략 비례한다고 보고 fast 실측으로 사전 추산
- **[Tier 2]** 양의 기울기 비율과 LDR에 부트스트랩 95% CI
- **[Tier 2]** S 추정 안정성: train 분할을 바꿔 S를 3회 추정하고 A 팔 결과가 유지되는지 확인

### 5.5 샘플링 민감도 [Tier 2]

Kronos는 sample_count개 예측을 평균해 반환하므로, 분산 붕괴 지표가 샘플링 설정의 영향을 받을 수 있다.

- 주요 팔(A, J, REF)에서 sample_count {1, 5} × temperature {1.0, 0.5}
- 붕괴가 샘플링 설정과 무관하게 나타나야 모델 내부 현상으로 해석 가능

### 5.6 강건성과 확장 [Tier 2]

- **저SNR OU** (예: snr ~ U(1, 3)): "OU trend 클래스가 OOD였다"는 해석 검증 (`CLAUDE.md` §6 2순위 후보)
- **V/A 비영**: 상수 거래량 / 가격 변동폭에 비례하는 노이즈 공유 거래량 (`0812 _1차실험_구현명세.md` 2절 Robustness Check)
- (선택) **금융 stylized fact**: 무조건부 분산이 같은 i.i.d. 수익률 vs GARCH(1,1) 수익률(변동성 군집). Stage 1~6 파이프라인 재사용, 출력 평가는 예측 경로의 |수익률| 자기상관
- (선택) **실데이터 probe**: KRX 일별 OHLCVA 구간을 실현 drift 또는 실현변동성 기준으로 라벨링해 probing만 적용

---

## 6. 재현 판정 기준 (P1)

GPU·드라이버가 바뀌면 fp16 연산과 샘플링 결과가 미세하게 달라질 수 있으므로 비트 단위 일치는 요구하지 않는다. 아래는 제안값이며 P0·P1 첫 실행 후 조정한다.

| 항목 | 일치로 보는 기준 |
|---|---|
| `data_fingerprint` | 완전 일치 (CPU numpy 생성) |
| 모델 스펙, 결정성 | 완전 일치 |
| 층별 held-out LDR | 상대오차 수 % 이내, 정점 위치·구간 패턴 동일 |
| null 대비 배율 | 같은 자릿수 |
| 코사인 지표 | 소수 둘째 자리까지 일치 |
| Stage 5·6 판정 | 각 팔의 상향/역방향/무효 분류 동일, 이항검정 유의성 동일 |
| REF | 양의 기울기 비율 ±2/32 이내 |

구현: `stages/stage7_fast_summary.py`의 Q0 재현표가 이 기준을 다음 값으로 적용한다 — LDR 상대오차 5% 이내, 코사인 차이 0.01 이내, null 대비 배율은 |log10 비| < 0.5, REF는 ±2/32. Stage 5 항목은 A 팔만, 앞 32표본(subset32)으로 비교한다. 기준값은 `experiment/expected/v1_baseline.json`에 있다.

불일치 시 원인 추적 순서: 코드 차이 → 실행 환경 차이 → 원 결과 오류. 원 결과가 틀린 것으로 판명되면 v1 보고서에 정정 기록을 남긴다.

---

## 7. RunPod 실행 환경

> **RunPod 사용 방식(Pod 생성·접속, 코드 동기화, 실행, 결과 회수)은 사용자가 로컬에 보유한 기존 RunPod 사용 코드를 레퍼런스로 제공한다. 코딩 에이전트는 그 레퍼런스의 패턴을 따라 구현하며, 이 문서는 RunPod 쪽 구체 방식을 정하지 않는다.**
> 아래는 레퍼런스 방식이 무엇이든 실험 쪽에서 충족해야 하는 요구사항이다.
>
> 구현: Kronos-investing의 `RunPod/`를 이식했다. 절차는 `RunPod/README.md`, 명세는 `docs/spec.md` §4에 있다. 아래 각 요구사항 뒤의 "→"는 그것을 충족하는 구현이다.

### 7.1 대체 진입점이 제공해야 할 기능 (`colab_control.py`가 하던 일)

- 실행할 stage 목록과 stage별 인자 지정, 순차 실행, 실패 시 이후 stage 중단, stage별 종료 코드 요약 → `run_plan.py`의 `fast` / `smoke` 계획. 단계와 인자는 코드에 고정돼 있고, 실패하면 그 종료 코드로 멈춘다. 끝난 단계는 `.done` 마커로 건너뛴다
- 실행 시작 시 진입점 버전과 커밋 해시 출력 (옛 진입점·옛 코드 사용 사고 방지) → `[run_plan v2-fast] … commit …`
- 의존성 설치 → `RunPod/setup_runpod.sh`가 `requirements-probe.txt`를 `/workspace/venv-probe`에 설치한다. Kronos-investing이 pod에서 검증한 핀에 scikit-learn, seaborn을 더한 것이고, `Kronos/requirements.txt`의 핀은 따르지 않는다
- 환경변수로 보존/scratch 루트 전달 (§3.2의 `paths.py` 수정과 짝) → `KEXP_OUT`, `KEXP_SCRATCH`

### 7.2 저장 요구사항

| 계층 | 내용 | 크기 (1차 실측) | 요구 |
|---|---|---|---|
| 보존 | 합성 데이터셋, `results/`, `figs/`, 실행 로그 | 수 GB 이하 (`pca_subset.npz` 36MB 등) | Pod 종료 후에도 남고 로컬로 회수 가능 |
| scratch | 레이어별 활성화 | 노이즈 유형당 약 39GB (OU+RW 동시 보관 시 약 78GB) | 재계산 가능(Stage 2). Stage 3·4만 읽음 |
| 모델 캐시 | HuggingFace 가중치 | 소량 | 재다운로드 허용 |

Stage 5·6은 보존 계층만 읽으므로 활성화 없이 실행할 수 있다(1차와 동일).

구현된 배치
- 보존: `experiment/runs/<RUN_ID>/` (네트워크 볼륨 위의 레포 안, `KEXP_OUT`). 요약 JSON·`fast_summary.md`·그림은 결과 브랜치로 push하고, 전체는 `RunPod/local.sh fetch`로 회수한다
- scratch: 컨테이너 디스크 `/root/kexp_scratch/<RUN_ID>` (`KEXP_SCRATCH`). 노이즈별로 Stage 4 뒤에 지우므로 OU와 RW를 동시에 보관하지 않는다. 최대 약 42GB
- 모델 캐시: `/workspace/hf`. revision을 고정했다 (`kexp/config.py`)

### 7.3 재현성 기록 (Colab에서 부족했던 부분)

- 각 stage 실행 로그 전문을 보존 계층에 파일로 저장 (Colab 출력 붙여넣기 대체) → `<OUT>/logs/<step>.log`
- results 메타에 커밋 해시, config 해시, GPU 이름, torch/CUDA 버전, 실행 시각 기록 → 실행 단위의 `manifest.json`에 기록한다. stage별 메타는 1차처럼 config 해시만 갖는다
- 산출물 경로의 tag(`v2`)·`<noise>_<model>` 구분 유지, 스모크는 `_smoke` 접미사 유지 → 1차와 같이 활성화는 `_smoke` 접미사, Stage 5는 `results_smoke.json`을 쓴다. 스모크는 실행 폴더(run_id)도 본 실행과 다르다
- 커밋되지 않은 코드로 본 실행하지 않는다 (실행 시점 코드 = 커밋) → `runpod.sh`는 추적 파일에 수정이 있으면 시작을 거부하고, pod는 origin의 커밋만 받는다

### 7.4 GPU·시간 참고 (1차 실측)

- Stage 2: T4 약 8.6분 / 노이즈 유형
- Stage 5 RW 논문 팔(22조합, n_eval=32): L4 약 35분 추정
- 1차는 Stage 5 도중 T4 → L4로 교체했다. 재실험은 **한 가지 GPU 종류로 고정**하고 기록한다 → RTX 4090 (spec §1). GPU 이름은 `manifest.json`과 `cloud_run.json`에 남는다

---

## 8. 결과별 허용 결론 (사전 정의)

### 8.1 표현

| 결과 | 말할 수 있는 것 |
|---|---|
| 사전학습 LDR ≈ 무작위 초기화 LDR | 분리는 입력·구조에서 오며, 학습된 표현이라는 근거 없음 |
| 사전학습 > 무작위, 입력단과 비슷 | 입력 특성이 보존·전달됨. 개념 "형성"의 근거는 약함 |
| 사전학습 > 무작위, 입력단 대비 증폭 | 모델이 문맥 누적으로 추세 구분을 만들어냄 |

### 8.2 개입

| S 개입 | 랜덤 방향(J/K) | REF와 비교 | 말할 수 있는 것 |
|---|---|---|---|
| 양수화 | 양수화 | — | 효과는 섭동 일반의 성질. 개념 특이적 조정이 아님 |
| 양수화 | 무작위 부호 | 다름 | 방향 특이적 조정은 가능하나, 모델이 실제 추세 입력에 하는 계산과 다른 출력을 주입 (1차 OU 잠정 결론) |
| 양수화 | 무작위 부호 | 닮음 | 개념 방향이 모델의 자연 기작과 일치 |
| 효과 없음 | — | — | 해당 설정에서 인과적 사용의 증거 없음 |

문장 선택은 기계적으로 한다 (`stage7_fast_summary.py`). 8.2는 spec §2.2의 사전 고정 판정으로, 8.1은 마지막 층·t=511의 LDR 비율로 고른다 (규칙은 spec §2.2의 "선택 규칙"). `fast_summary.md`는 위 두 표의 문장을 그대로 인용하고, 맞는 행이 없으면 수치만 보고한다.

> "모델이 개념을 이해한다/이해하지 못한다"는 표현은 쓰지 않는다. 말할 수 있는 것은 (a) 선형 표현의 존재, (b) 그 방향 개입의 출력 효과, (c) 그 효과와 모델 자연 행동의 일치 여부 세 가지다.

---

## 9. 코드 구조 (현재)

```
Kronos-latent-space/
├── CLAUDE.md                      # 인수인계: 범위, 진행 상황, 실행법, 이미 밟은 지뢰
├── docs/
│   ├── outline.md                 # 이 문서 (전체 설계)
│   └── spec.md                    # fast track 구현 명세
├── RunPod/                        # runpod.sh, setup_runpod.sh, push_meta.sh (pod) / local.sh (로컬) / README.md
├── requirements-probe.txt         # pod 환경
├── legacy/colab_control.py        # 1차의 Colab 제어 셀 (보존)
├── Kronos/                        # 모델 구현 (읽기 전용, model/ 만)
├── representations-in-tsfms/      # 논문 공식 구현 (읽기 전용)
└── experiment/
    ├── 0812_*.md                  # 1차 실험 문서 (보존)
    ├── expected/                  # fingerprints.json, v1_baseline.json (커밋된 기대값)
    ├── runs/<run_id>/             # 실행별 산출물 = KEXP_OUT (gitignore. 요약만 결과 브랜치로)
    └── code/
        ├── run_plan.py            # 신규: 파이프라인 러너 (fast / smoke)
        ├── kexp/
        │   ├── paths.py           # 수정: 환경변수 기반 루트
        │   ├── config.py          # 수정: tag v2, revision 핀, ControlCfg, fast 격자
        │   ├── kronos_loader.py   # 수정: 핀 전달, set_determinism, randomize_model
        │   ├── intervene.py       # 확장: build_random_payload
        │   ├── activations.py     # 확장: 저장 위치 선택
        │   ├── runmeta.py         # 신규: manifest, checksum, verify-run
        │   └── synth.py, ldr.py, steering_vec.py, hooks.py    # 1차 그대로
        ├── stages/
        │   ├── stage0_smoke.py            # 수정: --model
        │   ├── stage1_dataset.py          # 확장: --expect-fingerprint, --write-expected
        │   ├── stage2_activations.py      # 확장: --random-init, --positions bands
        │   ├── stage3_ldr.py              # 확장: --controls
        │   ├── stage4_steering_vector.py  # 1차 그대로
        │   ├── stage5_intervene.py        # 확장: --arms fast, --lambdas fast, --eval-batch
        │   ├── stage6_report.py           # 확장: --smoke, J pooled 행, fast 판정
        │   └── stage7_fast_summary.py     # 신규: fast_summary.md / .json
        └── tests/                 # 신규: pytest 단위 테스트
```

---

## 10. 산출물 체크리스트

"→" 뒤는 fast track 실행(`v2_fast`)에서 그 산출물이 나오는 곳이다. 전부 pod 실행 뒤에 채워진다.

- [ ] P0 이식 검증 로그 (스펙, fingerprint, 결정성) → `logs/00_stage0.log`, `logs/<noise>_01_stage1.log`, `logs/<noise>_02_stage2.log`
- [ ] P1 OU 재현 대조표 (§2 기준값 vs v2 실측) → `fast_summary.md`의 Q0 (A 팔, subset32)
- [ ] P2 RW Stage 6 요약 (논문 팔 + 개선안 팔) → `results/v2/rw_base/stage6_summary.json`. fast는 A·J·L·REF만이고 B~I는 Tier 2
- [ ] 랜덤/직교 방향 대조 결과 표 (부호 비율, 분산비 분리) → `fast_summary.md`의 Q2 표 (J). 직교(K)는 Tier 2
- [ ] 역방향 개입 결과 → `fast_summary.md`의 Q3 (L). M은 Tier 2
- [ ] 입력단 vs 무작위 초기화 vs 사전학습 층별 LDR 곡선 (OU, RW) → `figs/v2/stage3_controls_<noise>.png`, `fast_summary.md`의 Q1 표
- [ ] 샘플링 민감도 표 → Tier 2
- [ ] 갱신된 보고서: §8 표에 따른 최종 결론 → fast 결과를 본 뒤 작성
- [ ] 자소서용 한 문단 요약 (§8 허용 표현 안에서) → fast 결과를 본 뒤 작성