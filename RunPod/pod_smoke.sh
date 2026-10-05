#!/usr/bin/env bash
# Pod smoke test, run from the LOCAL machine right after deploying a pod (before the 3-hour fast run):
#
#   bash RunPod/pod_smoke.sh <host> <port>        # pod page -> Connect -> SSH over exposed TCP
#
# It checks, on the pod: GPU, disk space, deploy-page env vars, GitHub clone + push permission, the venv install
# (RunPod/setup_runpod.sh, ~13 min the first time), and then runs the smoke plan on the real Kronos-base
# (run_plan.py --plan smoke, a few minutes) with verify-run. Nothing is pushed and no results branch is created.
# Safe to re-run: the clone, the venv and the HF cache are reused. The pod-side log is /workspace/pod_smoke.log.
# The remote part travels over SSH, so this script itself does not have to be on the pod.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f RunPod/.env ]]; then set -a; source RunPod/.env; set +a; fi
host="${1:-}"; port="${2:-}"
[[ -n "$host" && -n "$port" ]] || { echo "usage: $0 <host> <port>" >&2; exit 1; }
key="${RUNPOD_SSH_KEY:-}"
REPO_SLUG="$(git remote get-url origin | sed -E 's#.*github\.com[:/]##; s#\.git$##')"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
LOCAL_HEAD="$(git rev-parse HEAD)"
[[ "$(git rev-parse "origin/$MAIN_BRANCH" 2>/dev/null)" == "$LOCAL_HEAD" ]] \
  || echo "WARNING: local HEAD is not origin/$MAIN_BRANCH -> the pod tests what is pushed, not what is here (push-code first)"

# shellcheck disable=SC2086
ssh -p "$port" ${key:+-i ${key/#\~/$HOME}} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  -o ServerAliveInterval=30 "root@$host" \
  "REPO_SLUG='$REPO_SLUG' MAIN_BRANCH='$MAIN_BRANCH' EXPECT_COMMIT='$LOCAL_HEAD' bash -s" <<'REMOTE' 2>&1 | tee "/tmp/pod_smoke_$host.log"
set -euo pipefail
exec > >(tee /workspace/pod_smoke.log) 2>&1
T0=$(date +%s)
FAIL=0
ok()   { echo "  [OK]   $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }
note() { echo "  [NOTE] $*"; }
pid1() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1; }
REPO="/workspace/$(basename "$REPO_SLUG")"

echo "== 1. pod =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || bad "nvidia-smi"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "4090" && ok "GPU is an RTX 4090" || note "GPU is not an RTX 4090 (spec §1 fixes 4090)"
[[ -d /workspace ]] && ok "/workspace is mounted" || bad "/workspace is missing (network volume not attached)"
ROOT_FREE=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9); WS_FREE=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
(( ROOT_FREE >= 45 )) && ok "container disk: ${ROOT_FREE} GB free (activations need ~42 GB)" || bad "container disk: only ${ROOT_FREE} GB free, need 45+ (Container Disk 60 GB)"
if [[ -x /workspace/venv-probe/bin/python ]]; then NEED=2; else NEED=10; fi
(( WS_FREE >= NEED )) && ok "/workspace: ${WS_FREE} GB free (need ~${NEED})" || bad "/workspace: only ${WS_FREE} GB free, need ~${NEED}"

echo "== 2. deploy-page env vars =="
TOKEN="$(pid1 GITHUB_TOKEN)"
[[ -n "$TOKEN" ]] && ok "GITHUB_TOKEN is set" || bad "GITHUB_TOKEN is not set"
[[ -n "$(pid1 PUBLIC_KEY)" ]] && ok "PUBLIC_KEY is set" || note "PUBLIC_KEY not in the pod env (ssh works anyway)"
[[ -n "$(pid1 RUNPOD_PUBLIC_IP)" && -n "$(pid1 RUNPOD_TCP_PORT_22)" ]] && ok "public IP + TCP 22: $(pid1 RUNPOD_PUBLIC_IP):$(pid1 RUNPOD_TCP_PORT_22)" \
  || bad "RUNPOD_PUBLIC_IP / RUNPOD_TCP_PORT_22 missing (expose TCP 22) -> local.sh fetch/watch cannot find the pod"
[[ -n "$(pid1 RUNPOD_USER_API_KEY)" ]] && ok "RUNPOD_USER_API_KEY is set (terminate can use it)" || note "no RUNPOD_USER_API_KEY on the pod (terminate needs a local key or the web)"

echo "== 3. repo =="
URL="https://x-access-token:${TOKEN}@github.com/${REPO_SLUG}.git"
if [[ ! -d "$REPO/.git" ]]; then
  git clone -q "$URL" "$REPO" && git -C "$REPO" remote set-url origin "https://github.com/${REPO_SLUG}.git" && ok "cloned $REPO_SLUG" || bad "clone failed (token has no access to $REPO_SLUG?)"
fi
cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=no -- . ':!experiment/runs')" ]]; then
  bad "tracked files are modified on the pod: git -C $REPO checkout -- ."
elif git fetch -q "$URL" "$MAIN_BRANCH" && git checkout -q -B "$MAIN_BRANCH" FETCH_HEAD; then
  HEAD_NOW=$(git rev-parse HEAD)
  [[ "$HEAD_NOW" == "$EXPECT_COMMIT" ]] && ok "on $MAIN_BRANCH @ ${HEAD_NOW:0:7} (= local HEAD)" || note "pod is on ${HEAD_NOW:0:7}, local HEAD is ${EXPECT_COMMIT:0:7}"
else
  bad "could not fetch $MAIN_BRANCH"
fi
git push -q --dry-run "$URL" "HEAD:refs/heads/results/_smoke_probe" 2>/dev/null && ok "token can push (Contents: write)" || bad "token cannot push to $REPO_SLUG (PAT: Contents Read and write)"
[[ $FAIL == 0 ]] || { echo; echo "POD SMOKE: FAILED before setup (fix the items above and re-run)"; exit 1; }

echo "== 4. environment (RunPod/setup_runpod.sh) =="
T1=$(date +%s)
bash RunPod/setup_runpod.sh && ok "setup done in $(( ($(date +%s) - T1) / 60 )) min" || { bad "setup_runpod.sh"; echo "POD SMOKE: FAILED in setup"; exit 1; }
source /workspace/venv-probe/bin/activate

echo "== 5. smoke plan on Kronos-base =="
export PYTHONPATH=experiment/code KEXP_OUT="$REPO/experiment/runs/smoke_gpu" KEXP_SCRATCH=/root/kexp_scratch/smoke_gpu
rm -rf "$KEXP_OUT" "$KEXP_SCRATCH"
T2=$(date +%s)
python experiment/code/run_plan.py --plan smoke --run-id smoke_gpu --model base && ok "smoke plan finished in $(( $(date +%s) - T2 )) s" || bad "smoke plan"
python -m kexp.runmeta verify-run --run-id smoke_gpu --plan smoke && ok "verify-run" || bad "verify-run"
python - <<'PY' || true
import json, os
m = json.load(open(os.environ["KEXP_OUT"] + "/manifest.json"))
e = m["env"]
print(f"  env: {e['gpu_name']} | driver {e['driver']} | torch {e['torch']} | cuda {e['cuda']} | python {e['python']}")
print("  steps (s):", ", ".join(f"{k.split('_', 1)[-1]} {v:.0f}" for k, v in m["steps"].items()))
PY
grep -h "지문 일치\|지문 불일치\|결정성" "$KEXP_OUT"/logs/*.log | sed 's/^/  /' || true
rm -rf "$KEXP_SCRATCH"

echo
if [[ $FAIL == 0 ]]; then
  echo "POD SMOKE: OK ($(( ($(date +%s) - T0) / 60 )) min). Start the run on the pod:"
  echo "  cd $REPO && tmux new -s probe"
  echo "  RUN_ID=v2_fast bash RunPod/runpod.sh"
else
  echo "POD SMOKE: FAILED. Logs: /workspace/pod_smoke.log, $KEXP_OUT/logs/"; exit 1
fi
REMOTE
