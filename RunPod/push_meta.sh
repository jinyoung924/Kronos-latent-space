#!/usr/bin/env bash
# Commit the small result files of ONE run and push them to the current results/<RUN_ID> branch.
#   bash RunPod/push_meta.sh experiment/runs/<RUN_ID>
# Called by RunPod/runpod.sh at the start (so the local side can find the pod) and on every exit, and by
# experiment/code/run_plan.py each time a noise finishes (so the OU summary is on GitHub while RW still runs).
# npz files, data/ and logs/ never go to GitHub: RunPod/local.sh fetch pulls them over SSH.
#
# Rules that keep code history and run history apart (RunPod/README.md §3):
#   1. refuses to run unless the checked-out branch is results/*  -> never commits to the main branch
#   2. stages only the WHITELIST below, files of at most 1 MB       -> no arrays, no code edits
#   3. push rejected? fetch + rebase + retry; still failing? push to a uniquely named fallback branch
#   4. without GITHUB_TOKEN it does nothing
set -uo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <run dir>" >&2; exit 1; }
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$REPO_DIR"
# relative to the run directory; globs are expanded by the shell
WHITELIST=(cloud_run.json manifest.json checksums.json requirements.lock.txt
           fast_summary.md fast_summary.json
           'results/v2/*/stage3_summary.json' 'results/v2/*/stage3_controls.json'
           'results/v2/*/stage4_summary.json' 'results/v2/*/stage6_summary.json'
           'results/v2/*/stage6_summary.csv' 'results/v2/*/steer/results.json'
           'figs/v2/*.png')
MAX_BYTES=1048576
REL="$(python3 -c 'import os, sys; print(os.path.relpath(os.path.realpath(sys.argv[1]), sys.argv[2]))' "$1" "$REPO_DIR")"
[[ "$REL" == ..* || "$REL" == /* ]] && { echo "push_meta: $1 is outside the repo -> refusing" >&2; exit 1; }
[[ -d "$REL" ]] || { echo "push_meta: $REL does not exist -> nothing to push"; exit 0; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != results/* ]]; then
  echo "push_meta: current branch '$BRANCH' is not a results/* branch -> refusing (see RunPod/README.md)" >&2
  exit 1
fi
RUN_ID="${RUN_ID:-${BRANCH#results/}}"
[[ "$(basename "$REL")" == "$RUN_ID" ]] || { echo "push_meta: $REL is not the run directory of $RUN_ID -> refusing" >&2; exit 1; }

pid1() { [[ -r /proc/1/environ ]] || return 0; tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1; }
TOKEN="${GITHUB_TOKEN:-$(pid1 GITHUB_TOKEN)}"
[[ -z "$TOKEN" && -z "${PUSH_URL:-}" ]] && { echo "push_meta: no GITHUB_TOKEN -> skipped (results stay on the pod)"; exit 0; }
REPO_SLUG="${REPO_SLUG:-$(git remote get-url origin | sed -E 's#.*github\.com[:/]##; s#\.git$##')}"
URL="${PUSH_URL:-https://x-access-token:${TOKEN}@github.com/${REPO_SLUG}.git}"   # PUSH_URL: non-GitHub remote / tests

git config user.name  "${GIT_AUTHOR_NAME:-runpod-bot}"
git config user.email "${GIT_AUTHOR_EMAIL:-runpod-bot@users.noreply.github.com}"

shopt -s nullglob
for pat in "${WHITELIST[@]}"; do
  for f in "$REL"/$pat; do                            # unquoted on purpose: the pattern is a glob
    [[ -f "$f" ]] || continue
    if (( $(wc -c < "$f") > MAX_BYTES )); then echo "push_meta: skipped $f (> 1 MB, fetch it over SSH)"; continue; fi
    git add -f -- "$f"                                # -f: experiment/runs/ is gitignored, the whitelist is the exception
  done
done
shopt -u nullglob
if git diff --cached --quiet; then
  echo "push_meta: nothing new under $REL (pushing unpushed commits, if any)"   # e.g. an earlier push failed
else
  git commit -q -m "results($RUN_ID): $REL summaries" \
    -m "pod: ${RUNPOD_POD_ID:-$(hostname)}  time: $(date -u +%Y-%m-%dT%H:%M:%SZ)  pushed by RunPod/push_meta.sh"
fi

for i in 1 2 3 4 5; do
  if git push -q "$URL" "HEAD:refs/heads/$BRANCH"; then
    echo "push_meta: pushed $REL summaries -> $REPO_SLUG:$BRANCH"; exit 0
  fi
  echo "push_meta: push rejected (attempt $i), rebasing onto remote $BRANCH"
  if git fetch -q "$URL" "$BRANCH" 2>/dev/null; then
    git rebase -q FETCH_HEAD || { git rebase --abort; break; }
  fi
  sleep "${PUSH_RETRY_SLEEP:-10}"
done

FALLBACK="$BRANCH-$(hostname)-$(date -u +%H%M%S)"
if git push -q "$URL" "HEAD:refs/heads/$FALLBACK"; then
  echo "push_meta: WARNING pushed to fallback branch $FALLBACK (merge it by hand)"; exit 0
fi
echo "push_meta: PUSH FAILED for $REL (commit is kept locally on $BRANCH)" >&2
exit 1
