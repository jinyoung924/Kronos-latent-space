#!/usr/bin/env bash
# Local-side helper for the RunPod loop (RunPod/README.md §0). Each subcommand is one safe, repeatable step:
#
#   bash RunPod/local.sh push-code "message"      # ① commit code/docs on the main branch and push (pull --rebase first)
#   bash RunPod/local.sh list                     #    result branches on origin, with their status
#   bash RunPod/local.sh fetch <RUN_ID> [host port]   # ④ rsync experiment/runs/<RUN_ID>/ from the pod, then checksum verify
#   bash RunPod/local.sh same  <RUN_ID> <RUN_ID>  #    do two fetched runs have bit-identical files?
#   bash RunPod/local.sh merge <RUN_ID>           # ⑤ keep: merge results/<RUN_ID> (summaries only) into the main branch, push, delete the remote branch
#   bash RunPod/local.sh drop  <RUN_ID>           # ⑤ discard: delete the remote branch (local files stay)
#   bash RunPod/local.sh terminate <RUN_ID>       # ⑥ terminate the pod of that run; refused until the local checksum verify passes
#   bash RunPod/local.sh watch <RUN_ID>           #    ④+⑥ unattended: wait until the pod reports ok, fetch + verify, then terminate (WATCH_TERMINATE=0 to keep the pod)
#   bash RunPod/local.sh ssh <RUN_ID>             #    open a shell on the pod of that run
#   bash RunPod/local.sh status                   #    where am I, what is uncommitted, what is unmerged
#
# Optional RunPod/.env (gitignored): RUNPOD_USER_API_KEY=... (terminate), RUNPOD_SSH_KEY=~/.ssh/id_ed25519_runpod, PYTHON=python, MAIN_BRANCH=main
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f RunPod/.env ]]; then set -a; source RunPod/.env; set +a; fi
cmd="${1:-status}"; arg="${2:-}"
PY="${PYTHON:-python}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/$(basename "$(git remote get-url origin 2>/dev/null || echo Kronos-latent-space)" .git)}"
die() { echo "local.sh: $*" >&2; exit 1; }
need_run() { [[ -n "$arg" ]] || die "usage: $0 $cmd <RUN_ID>"; }
cur_branch() { git symbolic-ref --short -q HEAD || git rev-parse --abbrev-ref HEAD; }   # symbolic-ref also works before the first commit
on_main() { [[ "$(cur_branch)" == "$MAIN_BRANCH" ]] || die "switch to $MAIN_BRANCH first: git checkout $MAIN_BRANCH"; }
clean_tracked() {
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "uncommitted changes to tracked files. Run: bash RunPod/local.sh push-code \"msg\"  (or git stash)"
}
pred_rel() { printf 'experiment/runs/%s' "$1"; }
verify() { env -u KEXP_OUT PYTHONPATH=experiment/code "$PY" -m kexp.runmeta checksum-verify --run-id "$1"; }
ssh_opts() {   # pods are short-lived and reuse IPs/ports, so host keys are not remembered
  local key="${RUNPOD_SSH_KEY:-}"
  printf '%s' "-p $1 ${key:+-i ${key/#\~/$HOME}} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
}
ensure_rsync() {   # $1 host, $2 port: RunPod images do not always ship rsync
  # shellcheck disable=SC2046
  ssh $(ssh_opts "$2") "root@$1" 'command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync; }'
}
# cloud_run.json of a run, from origin/results/<RUN_ID> (the pod pushes it at start and on exit)
# falls back to the fetched / merged local copy once the branch is gone
cloud_run() {
  local rel; rel="$(pred_rel "$1")"
  if git fetch -q origin "results/$1" 2>/dev/null; then git show "FETCH_HEAD:$rel/cloud_run.json" 2>/dev/null; else cat "$rel/cloud_run.json" 2>/dev/null; fi
}
field() { python3 -c 'import json, sys; print(json.load(sys.stdin).get(sys.argv[1]) or "")' "$1"; }

case "$cmd" in
  push-code)
    on_main
    git add -A -- .                                # run outputs are gitignored; their summaries arrive via merge
    if git diff --cached --quiet; then echo "nothing to commit"; else
      git commit -q -m "${arg:-update}" && echo "committed: ${arg:-update}"
    fi
    if git ls-remote --exit-code --heads origin "$MAIN_BRANCH" >/dev/null 2>&1; then   # absent on the very first push
      git pull -q --rebase origin "$MAIN_BRANCH" || die "rebase conflict with origin/$MAIN_BRANCH. Resolve, then: git rebase --continue && git push origin $MAIN_BRANCH"
    fi
    git push -q -u origin "$MAIN_BRANCH" && echo "pushed $MAIN_BRANCH -> origin ($(git rev-parse --short HEAD))"
    ;;

  list)
    git fetch -q --prune origin
    git for-each-ref --format='%(refname:short)' refs/remotes/origin/results/ | while read -r ref; do
      name="${ref#origin/results/}"
      merged=$(git merge-base --is-ancestor "$ref" "origin/$MAIN_BRANCH" 2>/dev/null && echo merged || echo unmerged)
      st=$(git show "$ref:$(pred_rel "$name")/cloud_run.json" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status","?"), d.get("gpu",""), d.get("elapsed_min",""),"min", "pod", d.get("pod_id",""))' 2>/dev/null || echo "no cloud_run.json")
      printf '%-32s %-9s %s\n' "$name" "$merged" "$st"
    done
    ;;

  fetch)
    need_run
    host="${3:-}"; port="${4:-}"; rel="$(pred_rel "$arg")"; remote=""
    if cr="$(cloud_run "$arg")" && [[ -n "$cr" ]]; then
      [[ -n "$host" ]] || host="$(field ssh_host <<<"$cr")"
      [[ -n "$port" ]] || port="$(field ssh_port <<<"$cr")"
      remote="$(field pred_dir <<<"$cr")"
      st="$(field status <<<"$cr")"
      [[ "$st" == ok ]] || echo "note: the run's status is '$st' (not finished or failed): this copies what exists so far"
    fi
    [[ -n "$host" && -n "$port" ]] || die "no ssh endpoint for $arg. Pass it: $0 fetch $arg <host> <port>"
    remote="${remote:-$REMOTE_REPO/$rel}"
    mkdir -p "$rel"
    ensure_rsync "$host" "$port"
    rsync -rtv --partial -e "ssh $(ssh_opts "$port")" "root@$host:$remote/" "$rel/"
    verify "$arg" || die "checksum verify FAILED for $arg (run again to re-copy; do not terminate the pod)"
    echo "$rel/ is complete and verified (summary: $rel/fast_summary.md). Next: bash RunPod/local.sh merge $arg, then terminate $arg"
    ;;

  same)
    need_run; other="${3:-}"; [[ -n "$other" ]] || die "usage: $0 same <RUN_ID> <RUN_ID>"
    python3 - "$(pred_rel "$arg")/checksums.json" "$(pred_rel "$other")/checksums.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["files"] for p in sys.argv[1:3])
diff = sorted(n for n in set(a) | set(b) if a.get(n, {}).get("sha256") != b.get(n, {}).get("sha256"))
print(f"{len(a)} vs {len(b)} files, {len(diff)} differ" + (f": {diff[:5]}" if diff else " -> bit-identical"))
sys.exit(1 if diff else 0)
PY
    ;;

  merge)
    need_run; on_main; clean_tracked
    verify "$arg" || die "fetch and verify the run first: $0 fetch $arg"
    git pull -q --rebase origin "$MAIN_BRANCH" || die "rebase conflict with origin/$MAIN_BRANCH. Resolve, then run merge again"
    git fetch -q origin "results/$arg" || die "origin has no branch results/$arg"
    git merge -q --no-ff FETCH_HEAD -m "merge results/$arg" \
      || die "merge failed (unexpected: results branches only add $(pred_rel "$arg")/ summaries). git merge --abort to undo"
    git push -q origin "$MAIN_BRANCH" && echo "merged results/$arg into $MAIN_BRANCH and pushed"
    git push -q origin --delete "results/$arg" && echo "deleted origin/results/$arg"
    ;;

  drop)
    need_run
    git push -q origin --delete "results/$arg" 2>/dev/null && echo "deleted origin/results/$arg" || echo "origin/results/$arg already gone"
    echo "local files under $(pred_rel "$arg")/ are left as they are"
    ;;

  terminate)
    need_run
    verify "$arg" || die "refusing: the local copy of $arg is not verified. Run: $0 fetch $arg"
    pod="${3:-}"
    if [[ -z "$pod" ]]; then
      if cr="$(cloud_run "$arg")"; then pod="$(field pod_id <<<"$cr")"; fi
      [[ -n "$pod" ]] || pod="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pod_id") or "")' "$(pred_rel "$arg")/cloud_run.json" 2>/dev/null || true)"
    fi
    [[ -n "$pod" ]] || die "pod id unknown. Pass it: $0 terminate $arg <pod id>"
    if [[ "${YES:-0}" != "1" ]]; then
      read -r -p "terminate pod $pod? The network volume is kept. [y/N] " ans; [[ "$ans" == y || "$ans" == Y ]] || die "cancelled"
    fi
    query="{\"query\":\"mutation { podTerminate(input: {podId: \\\"$pod\\\"}) }\"}"
    if [[ -n "${RUNPOD_USER_API_KEY:-}" ]]; then
      resp="$(curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_USER_API_KEY" -H 'Content-Type: application/json' -d "$query")"
    else
      # no local key: let the pod call the API with the key of its own deploy-page env (the key never leaves the pod)
      cr="$(cloud_run "$arg")" || die "RUNPOD_USER_API_KEY is not set locally and origin has no results/$arg to find the pod: terminate pod $pod on the web"
      # shellcheck disable=SC2046,SC2029
      resp="$(ssh $(ssh_opts "$(field ssh_port <<<"$cr")") "root@$(field ssh_host <<<"$cr")" \
        "K=\$(tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^RUNPOD_USER_API_KEY=//p'); [ -n \"\$K\" ] || { echo '{\"errors\":\"no RUNPOD_USER_API_KEY on the pod\"}'; exit 0; }; curl -s -X POST \"https://api.runpod.io/graphql?api_key=\$K\" -H 'Content-Type: application/json' -d '$query'" || true)"
    fi
    echo "$resp"
    [[ -n "$resp" && "$resp" != *'"errors"'* ]] || die "the API reported an error: terminate pod $pod on the web"
    echo "pod $pod terminated. Delete the network volume on the web once a second local backup exists."
    ;;

  watch)
    # Wait for the pod to report the end of the run (cloud_run.json status on results/<RUN_ID>), then fetch + verify + terminate.
    # A failed run is never terminated: the pod stays up so the log can be read and the run resumed.
    need_run
    every="${WATCH_EVERY:-120}"
    echo "watching results/$arg every ${every}s (this machine must stay awake: run it under 'caffeinate -i' on macOS)"
    while :; do
      st=""; if cr="$(cloud_run "$arg")"; then st="$(field status <<<"$cr")"; fi
      echo "$(date '+%H:%M:%S') status=${st:-unknown}"
      case "$st" in
        ok) break ;;
        failed) die "the run failed on the pod. Not terminating. Look at it: $0 ssh $arg" ;;
      esac
      sleep "$every"
    done
    "$0" fetch "$arg" || die "fetch/verify failed. Not terminating. Run again: $0 fetch $arg"
    if [[ "${WATCH_TERMINATE:-1}" == "1" ]]; then YES=1 "$0" terminate "$arg"; else echo "WATCH_TERMINATE=0 -> the pod is left running"; fi
    echo "done. Next: bash RunPod/local.sh merge $arg"
    ;;

  ssh)
    need_run
    cr="$(cloud_run "$arg")" || die "origin has no branch results/$arg"
    # shellcheck disable=SC2046
    exec ssh $(ssh_opts "$(field ssh_port <<<"$cr")") "root@$(field ssh_host <<<"$cr")"
    ;;

  status)
    echo "branch : $(cur_branch) @ $(git rev-parse --short HEAD 2>/dev/null || echo 'no commits yet')"
    git fetch -q origin 2>/dev/null || true
    echo "vs $MAIN_BRANCH: $(git rev-list --count "origin/$MAIN_BRANCH..HEAD" 2>/dev/null || echo ?) ahead, $(git rev-list --count "HEAD..origin/$MAIN_BRANCH" 2>/dev/null || echo ?) behind"
    echo "uncommitted tracked changes:"; git status --short --untracked-files=no | sed 's/^/    /'
    echo "result branches:"; "$0" list | sed 's/^/    /'
    ;;
  *) die "unknown subcommand '$cmd'. See header of $0" ;;
esac
