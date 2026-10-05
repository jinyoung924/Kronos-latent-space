#!/usr/bin/env bash
# Environment setup on a RunPod pod. Idempotent: safe to re-run after a pod restart.
# Called by RunPod/runpod.sh; can be run alone.
#
# The image does not matter: a Python 3.12 venv on /workspace gets requirements-probe.txt (the Kronos-investing pins,
# torch included). It is a separate venv (/workspace/venv-probe), so the Kronos-investing venv on the same volume is
# not touched. The Kronos model code is vendored in this repo; the model + tokenizer are downloaded at the revisions
# pinned in kexp/config.py into /workspace/hf (shared with Kronos-investing, which pins the same revisions).
set -euo pipefail
cd "$(dirname "$0")/.."

# /workspace is the network volume: the venv and the HF cache survive the pod.
if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then export HF_HOME=/workspace/hf; fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=' ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "== system =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace / 2>/dev/null | tail -2 || true     # volume (run dir, venv) and container disk (activations)

echo "== venv (Python 3.12) =="
if [[ -z "${VENV_DIR:-}" ]]; then
  if [[ -d /workspace ]]; then VENV_DIR=/workspace/venv-probe; else VENV_DIR="$PWD/.venv"; fi
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if command -v python3.12 >/dev/null 2>&1 && python3.12 -m venv "$VENV_DIR"; then :; else
    # Image has no usable 3.12 -> let uv download a standalone CPython 3.12 (no apt).
    rm -rf "$VENV_DIR"
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.12 --seed "$VENV_DIR"
  fi
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version | grep -q ' 3\.12\.' || { echo "FATAL: venv python is $(python --version), need 3.12 ($VENV_DIR)" >&2; exit 2; }
echo "venv: $VENV_DIR ($(python --version))"

echo "== python deps (requirements-probe.txt) =="
python -m pip install -q --upgrade pip
# TORCH_INDEX_URL: set it when the default PyPI torch wheel does not match the host driver (RunPod/README.md §4).
python -m pip install -r requirements-probe.txt ${TORCH_INDEX_URL:+--extra-index-url "$TORCH_INDEX_URL"}

echo "== Kronos code (vendored) =="
[[ -f Kronos/model/kronos.py ]] || { echo "FATAL: Kronos/model/kronos.py is missing from the repo" >&2; exit 2; }
echo "Kronos/ at $(git log -1 --format=%H -- Kronos)"

echo "== model + tokenizer (pins from kexp/config.py) =="
PYTHONPATH=experiment/code python - <<'PY'
from huggingface_hub import snapshot_download
from kexp.config import CFG
m = CFG.model
for name, rev in ((m.model_id, m.revision), (m.tokenizer_id, m.tokenizer_revision)):
    if not rev:
        raise SystemExit(f"FATAL: revision for {name} is None: pin the HF commit sha in kexp/config.py, push-code")
    print(name, "@", rev, "->", snapshot_download(name, revision=rev))
PY

echo "== verify =="
python - <<'PY'
import re, torch
pin = re.search(r"^torch==([^\s#]+)", open("requirements-probe.txt").read(), re.M).group(1)
have = torch.__version__.split("+")[0]
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), pinned {pin}")
assert have == pin, f"torch {have} installed but requirements-probe.txt pins {pin}"
assert torch.cuda.is_available(), "CUDA not available inside torch: host driver too old for this wheel? (RunPod/README.md §4)"
print("gpu:", torch.cuda.get_device_name(0))
PY
echo "setup OK"
