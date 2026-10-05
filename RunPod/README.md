# RunPod에서 fast track 돌리기 (docs/spec.md §4)

> **Claude 세션은 이 문서를 먼저 읽고 §0의 명령으로만 pod와 결과 브랜치를 다룹니다.**

`docs/spec.md`의 fast track(`v2_fast`: OU → RW, pod 1대)을 RunPod Secure Cloud의 GPU pod에서 실행하고 로컬로 가져오는 절차다. Kronos-investing의 `RunPod/`를 이식했다.
무엇이 어느 길로 움직이는지가 전부다.

| 대상 | 크기 | 경로 |
| --- | --- | --- |
| 코드 | 작음 | GitHub: 로컬 `push-code` → pod `git clone` / 새 RUN_ID는 최신 origin `main`에서 시작 |
| 합성 데이터셋 | 노이즈당 약 45 MB | 옮기지 않는다. pod의 Stage 1이 생성하고 `experiment/expected/fingerprints.json`과 대조한다 |
| 실행 폴더 `experiment/runs/<RUN_ID>/` (data, results, figs, logs) | 수백 MB | pod의 **네트워크 볼륨** 위 레포 안에 쓰고, SSH(rsync)로 로컬 회수 → `checksum verify` |
| 요약 JSON·`fast_summary.md`·그림·메타데이터 (§3 화이트리스트, 각 1 MB 이하) | 작음 | GitHub: pod가 `results/<RUN_ID>` 브랜치로 push → 로컬 `merge`가 `main`에 병합. **`fetch` 전에도 GitHub에서 바로 볼 수 있다** |
| 활성화 (scratch) | 노이즈당 약 39 GB | **컨테이너 디스크** `/root/kexp_scratch/<RUN_ID>`. Stage 4 뒤에 지운다. 회수하지 않는다 |

## 0. 실행 한 번의 루프

```
로컬 (main)                                       pod (/workspace = 네트워크 볼륨)
① bash RunPod/local.sh push-code "v2 fast track"
② (웹) kronos-probe 템플릿으로 배포
                                                  ③ 첫 pod 한 번: clone (§1.3), tmux
                                                  ④ RUN_ID=v2_fast bash RunPod/runpod.sh
                                                     setup → GPU 스모크 → OU → 중간 push → RW → verify → checksum → push
⑤ caffeinate -i bash RunPod/local.sh watch v2_fast
   (OU 요약은 중간 push 시점에 GitHub results/v2_fast 브랜치에서 먼저 확인)
⑥ bash RunPod/local.sh merge v2_fast
```

| 단계 | 어디서 | 명령 | 하는 일 |
| --- | --- | --- | --- |
| ① | 로컬 | `bash RunPod/local.sh push-code "메시지"` | `main`에 커밋하고 push (`pull --rebase` 먼저) |
| ④ | pod | `RUN_ID=v2_fast bash RunPod/runpod.sh` | 아래 §1.3. **pod에서는 코드를 고치지 않음** |
| ⑤ | 로컬 | `bash RunPod/local.sh watch v2_fast` | pod가 `status: ok`를 push할 때까지 2분마다 확인 → `fetch`(rsync + verify) → `terminate`. 실패한 실행과 verify 실패는 종료하지 않는다. `WATCH_TERMINATE=0`이면 pod를 남긴다 |
| ⑤' | 로컬 | `fetch <RUN_ID>` / `terminate <RUN_ID>` | watch를 나눠서 할 때. `terminate`는 로컬 verify가 통과해야 한다 |
| ⑥a | 로컬 | `bash RunPod/local.sh merge v2_fast` | 남길 실행: 요약을 `main`에 병합·push, 원격 결과 브랜치 삭제 |
| ⑥b | 로컬 | `bash RunPod/local.sh drop <RUN_ID>` | 버릴 실행: 원격 결과 브랜치만 삭제 |
| 확인 | 로컬 | `status` / `list` / `ssh <RUN_ID>` / `same <A> <B>` | 상태, 결과 브랜치 목록, pod 셸, 두 실행의 파일이 같은지 |

결과는 `experiment/runs/<RUN_ID>/fast_summary.md`(Q0~Q4의 답), `figs/v2/`, `logs/<step>.log`에서 본다.

**규칙** (Kronos-investing과 같다)

1. 코드는 로컬 `main`에서만 고치고 `push-code`로 올린다. pod는 새 RUN_ID를 시작할 때 최신 origin `main`을 받는다.
2. RUN_ID는 덮어쓰지 않는다. 같은 RUN_ID로 다시 실행하면 "끊긴 실행 이어 하기"다(코드는 그 브랜치의 것 그대로, `.done` 마커가 있는 단계는 건너뜀). 코드를 고쳤으면 새 RUN_ID를 쓴다.
3. pod의 커밋은 `results/<RUN_ID>` 브랜치에만 가고 §3의 화이트리스트 파일만 담는다.
4. 로컬 `checksum verify`가 통과하기 전에는 pod를 종료하지 않는다(`terminate`가 강제한다).
5. 한 RUN_ID 안에서 GPU 종류를 섞지 않는다.

## 1. Pod 배포

### 1.1 사양

| 항목 | 값 | 비고 |
| --- | --- | --- |
| 클라우드 | **Secure Cloud**, On-Demand | pod에 GitHub 토큰을 둔다 |
| GPU | RTX 4090 24 GB × 1 | |
| 템플릿 | `kronos-probe` = Kronos-investing의 `kronos-infer` 복제 | **Container Disk 20 → 60 GB**만 바꾼다 (활성화 최대 약 42 GB) |
| 네트워크 볼륨 | Kronos-investing 볼륨 재사용, `/workspace` | 증가분 약 10 GB (venv 약 8 GB + 실행 폴더 수백 MB). 배포 전에 여유를 확인한다 |
| venv | `/workspace/venv-probe`, Python 3.12, `requirements-probe.txt` | Kronos-investing의 `/workspace/venv`와 분리. 첫 설치 약 13분 |
| 모델 | `NeoQuasar/Kronos-base` @ `2b554741…`, 토크나이저 @ `0e011738…` (`kexp/config.py`) | Kronos-investing과 같은 핀 → `/workspace/hf` 캐시 재사용 |
| Kronos 코드 | 레포의 `Kronos/model/` (vendored) | clone하지 않는다 |
| 정밀도 | float32, TF32 off (`kl.set_determinism()`) | |

### 1.2 배포 화면에 넣는 환경변수

| 이름 | 값 | 없으면 |
| --- | --- | --- |
| `GITHUB_TOKEN` | **이 레포**에 Contents: Read and write 권한이 있는 fine-grained PAT (Kronos-investing 토큰의 Repository access에 이 레포를 추가) | `runpod.sh`가 시작을 거부 (`ALLOW_NO_PUSH=1`이면 진행) |
| `PUBLIC_KEY` | 내 SSH 공개키 한 줄 | 결과를 옮길 수 없다 |
| `RUNPOD_USER_API_KEY` | RunPod API 키 (선택) | 로컬 `terminate`가 로컬 키 또는 pod의 키를 쓴다. 둘 다 없으면 웹에서 종료 |

`RUNPOD_POD_ID`, `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`는 RunPod이 넣는다(TCP 22를 expose해야 한다). SSH 세션은 컨테이너 환경변수를 상속하지 않으므로 스크립트가 PID 1의 환경에서 읽는다.

### 1.3 pod 안에서 실행

```bash
# 첫 pod 한 번. 토큰은 PID 1 환경에서 꺼내고 .git/config에 남기지 않는다
TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^GITHUB_TOKEN=//p')
git clone https://x-access-token:${TOKEN}@github.com/jinyoung924/Kronos-latent-space.git /workspace/Kronos-latent-space
git -C /workspace/Kronos-latent-space remote set-url origin https://github.com/jinyoung924/Kronos-latent-space.git

cd /workspace/Kronos-latent-space
apt-get install -y -qq tmux >/dev/null; tmux new -s probe        # SSH가 끊겨도 실행이 유지된다
RUN_ID=v2_fast bash RunPod/runpod.sh
```

`runpod.sh`가 하는 일: 환경변수 읽기 → 수정된 추적 파일이 있으면 거부 → `results/<RUN_ID>` 브랜치(새 RUN_ID면 최신 origin `main`에서) → push 권한 사전 검사 → `setup_runpod.sh` → cloud_run.json 기록·push → **GPU 스모크**(`run_plan.py --plan smoke --run-id smoke_gpu --model base` + `verify-run`) → `RUN_CMD`(`run_plan.py --plan fast`) → `verify-run` → `checksum-write` → push.
어떤 경로로 끝나도 상태를 cloud_run.json에 적고 push한다. 실패하면 pod는 그대로 남으니 `logs/runpod_<RUN_ID>.log`와 `experiment/runs/<RUN_ID>/logs/`를 본다.

`run_plan.py`의 fast 계획 (노이즈 X ∈ [ou, rw] 순서, `docs/spec.md` §3.11):
stage0 → [stage1 `--expect-fingerprint` → stage2 → stage2 `--random-init --positions bands` → stage3 `--controls` → stage4 → scratch 정리 → stage5 `--arms fast --lambdas fast --n-eval 64 --eval-batch 32` → stage5 `--reference-trend` → stage6 → stage7 요약 + 중간 push] × 노이즈.

### 1.4 환경변수 (스크립트 knob)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RUN_ID` | (필수) | 실행 폴더와 결과 브랜치 이름. `v2_fast` |
| `PLAN` | `fast` | `run_plan.py`의 계획 |
| `RUN_CMD` | `python experiment/code/run_plan.py --plan $PLAN --run-id $RUN_ID --model base` | 실제로 돌릴 명령. 노이즈를 나누려면 `--noise ou`를 붙여 넘긴다 |
| `SKIP_SMOKE` | `0` | `1`이면 GPU 스모크 생략 |
| `MAIN_BRANCH` | `main` | 코드 브랜치 |
| `KEXP_SCRATCH` | `/root/kexp_scratch/$RUN_ID` | 활성화 위치 (컨테이너 디스크) |
| `TORCH_INDEX_URL` | 빈 값 | torch 휠을 다른 인덱스에서 받아야 할 때 (§4) |
| `ALLOW_NO_PUSH` | `0` | `1`이면 `GITHUB_TOKEN` 없이 실행 (테스트용) |
| `VENV_DIR`, `HF_HOME` | `/workspace/venv-probe`, `/workspace/hf` | 볼륨에 남는 위치 |

`KEXP_OUT`은 `runpod.sh`가 `experiment/runs/$RUN_ID`로 고정한다.

## 2. 파일

| 파일 | 어디서 실행 | 역할 |
| --- | --- | --- |
| `local.sh` | 로컬 | §0의 동작 전부 |
| `runpod.sh` | pod | 진입점 |
| `setup_runpod.sh` | pod | Python 3.12 venv, `requirements-probe.txt`, 핀된 모델·토크나이저 다운로드, torch 버전·CUDA 확인. 재실행 안전 |
| `push_meta.sh` | pod | 실행 폴더의 화이트리스트 파일만 커밋해 `results/<RUN_ID>`로 push |
| `.env` (gitignore) | 로컬 | 선택: `RUNPOD_USER_API_KEY=...`, `RUNPOD_SSH_KEY=~/.ssh/id_ed25519_runpod`, `PYTHON=python`, `MAIN_BRANCH=main` |

관련 코드: `experiment/code/run_plan.py`(러너), `kexp/runmeta.py`(manifest, checksum, verify-run), `stages/stage7_fast_summary.py`(요약).

## 3. 브랜치 규칙과 안전장치

```
main                ── push-code의 코드·문서 커밋 + merge가 만드는 "merge results/<RUN_ID>" 커밋
results/<RUN_ID>    ── pod가 만드는 "results(<RUN_ID>): ... summaries" 커밋만
```

결과 브랜치에 올라가는 파일 (`experiment/runs/<RUN_ID>/` 기준, 각 1 MB 이하):

```
cloud_run.json  manifest.json  checksums.json  requirements.lock.txt
fast_summary.md  fast_summary.json
results/v2/*/stage3_summary.json   results/v2/*/stage3_controls.json
results/v2/*/stage4_summary.json   results/v2/*/stage6_summary.json
results/v2/*/stage6_summary.csv    results/v2/*/steer/results.json
figs/v2/*.png
```

| 상황 | 막는 장치 |
| --- | --- |
| 결과 커밋에 코드나 npz가 섞임 | `push_meta.sh`: `results/*` 브랜치가 아니면 거부, 화이트리스트만 stage. `runpod.sh`: 수정된 추적 파일이 있으면 실행 거부 |
| 데이터가 기대값과 다름 | Stage 1 `--expect-fingerprint` (불일치면 exit 2) |
| 가중치가 바뀜 | `kexp/config.py`의 revision 핀. base인데 핀이 없으면 로드 실패 |
| 토큰에 쓰기 권한이 없음 | `runpod.sh`가 시작할 때 `push --dry-run` |
| 환경·경로·훅 오류 | GPU 스모크(smoke 계획 전체)가 본 실행 전에 몇 분 안에 잡는다 |
| 실행이 불완전함 | `kexp.runmeta verify-run`: 필수 파일, `steer/results.json`의 키 수(5팔 × 5λ + REF = 26), NaN |
| 회수 중 파일 손상·누락 | `fetch`의 `checksum verify`. 실패하면 `merge`와 `terminate`가 거부 |
| pod가 중간에 죽음 | 실행 폴더와 `.done` 마커는 볼륨에 남는다. 새 pod에서 같은 명령을 재실행. 활성화(컨테이너 디스크)가 사라졌으면 Stage 2는 마커가 있어도 다시 돈다 |
| 두 pod가 같은 브랜치에 push | 거절 → fetch+rebase 재시도 5회 → 예비 브랜치 `results/<RUN_ID>-<host>-<시각>` |

## 4. 문제 해결

| 증상 | 원인 / 조치 |
| --- | --- |
| `FATAL: GITHUB_TOKEN cannot push` / 403 | 토큰에 이 레포의 Contents write가 없음. `GITHUB_TOKEN=<새 토큰> RUN_ID=... bash RunPod/runpod.sh` |
| `CUDA not available inside torch` | torch 휠의 CUDA가 호스트 드라이버보다 새것. 배포 화면 CUDA 필터를 올려 재배포하거나 `TORCH_INDEX_URL` 지정 후 `/workspace/venv-probe`를 지우고 재실행 |
| `torch X installed but requirements-probe.txt pins Y` | `/workspace/venv-probe` 삭제 후 재실행 (**`/workspace/venv`는 Kronos-investing 것이다. 지우지 않는다**) |
| stage1 `지문 불일치` (exit 2) | 생성 코드나 numpy/scipy가 로컬과 다르다. 입력단 LDR이 19.57 / 2.22로 같은지 로그에서 확인하고 원인을 정한다 |
| `tracked files are modified` (pod) | pod에서 코드를 고쳤음. `git checkout -- .` 후 로컬에서 고쳐 `push-code` |
| 컨테이너 디스크 부족 (stage2) | Container Disk가 60 GB인지 확인. 앞 노이즈의 활성화가 남았으면 `rm -rf /root/kexp_scratch/<RUN_ID>/activations/v2/<noise>*` |
| GPU 메모리 부족 (stage5) | `--eval-batch`를 줄이면 **샘플 값이 달라져** v1과의 subset32 비교가 깨진다. 먼저 원인을 본다 |
| `fetch`: `no ssh endpoint` | 결과 브랜치가 아직 없거나 TCP 22가 expose되지 않음. `fetch <RUN_ID> <host> <port>` |
| `fetch`: `checksum verify FAILED` | 실행이 아직 안 끝났거나(checksums.json 없음) 복사 중 끊김. 다시 `fetch` |
| 특정 단계부터 다시 | `experiment/runs/<RUN_ID>/.done/<step>`을 지우고 같은 명령 재실행 (같은 코드). 코드를 고쳤으면 새 RUN_ID |

## 5. 실측 기록 (첫 실행 후 기입)

추정은 v1 실측(Stage 2: T4 약 8.6분/노이즈, Stage 5: L4에서 n=32 조합당 약 1.6분)뿐이다. `runpod.sh`와 `setup_runpod.sh`는 아직 실제 pod에서 돌려 보지 않았다.

| 항목 | 예상 (spec §7) | 실측 (RTX 4090) |
| --- | --- | --- |
| 첫 환경 설치 | 약 13분 | |
| GPU 스모크 | 수 분 | |
| 노이즈당 Stage 1~4 + 대조군 | 약 20~30분 | |
| 노이즈당 Stage 5 (generate 22회 × n=64) | 약 45~65분 | |
| 전체 (OU → RW), 비용 | 약 2.5~3.5시간, 약 $2~3 | |
