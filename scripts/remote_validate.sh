#!/usr/bin/env bash
# Drive scripts/validate_interp_2026.py on a remote GPU box, one stage at a time.
#
# The box is metered, so every stage is separate and nothing repeats work:
#
#   preflight   read-only. GPU, memory, disk, python. Costs seconds, no install.
#   sync        rsync the package and scripts. No weights, no venv, no data.
#   install     create a venv and install the interp extra. Once per box.
#   run         start the validation under nohup and return. Survives a dropped
#               connection; the remote log keeps going.
#   tail        follow the remote log.
#   fetch       copy report.json (and artifacts) back into runs/validate/.
#   status      is it still running, and what has it finished.
#   all         preflight, sync, install, run, tail.
#
# The run stage is resumable by construction: validate_interp_2026.py skips any
# step already present in report.json, so re-running after a crash, a stopped
# instance or a fetched partial report costs only the steps that did not finish.
#
# Usage:
#   scripts/remote_validate.sh preflight  <host>
#   scripts/remote_validate.sh all        <host> [model] [budget]
#   scripts/remote_validate.sh fetch      <host>
#
# Defaults: model Qwen/Qwen3-8B, budget smoke.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:-}"
HOST="${2:-}"
MODEL="${3:-Qwen/Qwen3-8B}"
BUDGET="${4:-smoke}"

# Same default as remote_session.sh: the box Codex provisioned deploys to ~/aiasylum,
# and the two scripts share its venv, database and GPU lock.
REMOTE_DIR="${REMOTE_DIR:-aiasylum}"
# All remote commands enter this directory; artifact paths stay relative to it.
RUN_NAME="${RUN_NAME:-$(echo "$MODEL" | tr '/' '-')-$BUDGET}"
REMOTE_OUT="runs/validate/$RUN_NAME"
LOCAL_OUT="runs/validate/$RUN_NAME"
LOG="$REMOTE_OUT/run.log"
PIDFILE="$REMOTE_OUT/run.pid"

# Short smoke completions use non-thinking mode by default. Set --thinking
# explicitly only with a token budget suitable for reasoning completions.
EXTRA_FLAGS="${EXTRA_FLAGS:---patching}"

die() { echo "error: $*" >&2; exit 1; }
[[ -n "$STAGE" ]] || die "usage: $0 <preflight|sync|install|run|tail|status|fetch|all> <host> [model] [budget]"
[[ -n "$HOST"  ]] || die "no host given. Pass the ssh alias, e.g. $0 preflight gpu-box"
[[ "$HOST" =~ ^[A-Za-z0-9][A-Za-z0-9._@-]*$ ]] || die "invalid SSH host"
[[ "$REMOTE_DIR" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "/$REMOTE_DIR/" != *"/../"* ]] || die "REMOTE_DIR must be a plain relative path inside the remote home"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_NAME must be a plain name"
[[ "$MODEL" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || die "MODEL must be a repository ID or plain relative model path"
[[ "$BUDGET" == smoke || "$BUDGET" == small || "$BUDGET" == full ]] || die "budget must be smoke, small or full"
for flag in $EXTRA_FLAGS; do
  [[ "$flag" == --thinking || "$flag" == --patching || "$flag" == --surgery ]] || die "unsupported EXTRA_FLAGS value: $flag"
done
mkdir -p "$HOME/.ssh"

# Reuse one connection for the whole script: cheaper, and it will not trip a
# server-side ssh rate limit the way a dozen separate logins can.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$HOME/.ssh/cm-validate-%r@%h:%p" -o ControlPersist=10m
          -o ConnectTimeout=15 -o BatchMode=yes)
rsh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

# Share prompt-library export, migrations, exclusions, and CUDA verification
# with the API session workflow so a fresh remote installation is usable.
stage_preflight() { REMOTE_DIR="$REMOTE_DIR" scripts/remote_session.sh preflight "$HOST"; }
stage_sync() { REMOTE_DIR="$REMOTE_DIR" scripts/remote_session.sh sync "$HOST"; }
stage_install() { REMOTE_DIR="$REMOTE_DIR" scripts/remote_session.sh install "$HOST"; }

stage_run() {
  echo "== start validation on $HOST =="
  echo "   model  $MODEL"
  echo "   budget $BUDGET"
  echo "   out    $REMOTE_OUT"
  rsh "bash -s" <<REMOTE
set -euo pipefail
cd ~/$REMOTE_DIR
mkdir -p $REMOTE_OUT
./venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; refusing validation"'
if [ -f $PIDFILE ] && kill -0 \$(cat $PIDFILE) 2>/dev/null; then
  echo "already running as pid \$(cat $PIDFILE); use 'tail' or 'status'"; exit 0
fi
# nohup, so the run survives this ssh connection closing. Any step already in
# report.json is skipped, so restarting after a crash is cheap.
PYTHONPATH=. nohup ./venv/bin/python scripts/validate_interp_2026.py \
  --model '$MODEL' --out '$REMOTE_OUT' --device cuda --budget '$BUDGET' $EXTRA_FLAGS \
  >> $LOG 2>&1 &
echo \$! > $PIDFILE
echo "started pid \$(cat $PIDFILE); log at $LOG"
REMOTE
}

stage_tail() { echo "== tail $HOST:$LOG (ctrl-c to stop following) =="; rsh "cd ~/$REMOTE_DIR && tail -f -n 40 $LOG"; }

stage_status() {
  rsh "bash -s" <<REMOTE
set -u
cd ~/$REMOTE_DIR
if [ -f $PIDFILE ] && kill -0 \$(cat $PIDFILE) 2>/dev/null; then
  echo "running (pid \$(cat $PIDFILE))"
else
  echo "not running"
fi
if [ -f $REMOTE_OUT/report.json ]; then
  ./venv/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("$REMOTE_OUT/report.json")
d = json.loads(p.read_text())
for name, e in d.get("steps", {}).items():
    mark = "FAILED" if "error" in e else "ok"
    print(f"  {name:14s} {mark:7s} {e.get('elapsed_s', 0):8.1f}s "
          f"{e.get('error', '')}")
print("  total", round(sum(e.get("elapsed_s", 0) for e in d.get("steps", {}).values()) / 60, 1), "min")
PY
else
  echo "  no report.json yet"
fi
tail -n 5 $LOG 2>/dev/null || true
REMOTE
}

stage_fetch() {
  echo "== fetch results into $LOCAL_OUT =="
  mkdir -p "$LOCAL_OUT"
  # Report and direction artifacts only. Edited model directories stay on the
  # box: they are many GB and rebuildable from the direction.
  rsync -az --stats -e "ssh ${SSH_OPTS[*]}" \
    --exclude 'edited-*' \
    "$HOST:$REMOTE_DIR/$REMOTE_OUT/" "$LOCAL_OUT/"
  echo
  [ -f "$LOCAL_OUT/report.json" ] && python3 - "$LOCAL_OUT/report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
steps = d.get("steps", {})
print("== summary ==")
print("model", d.get("meta", {}).get("model"), "budget", d.get("meta", {}).get("budget"),
      "hardware", d.get("meta", {}).get("hardware"))
def g(name, *keys):
    cur = steps.get(name, {}).get("result")
    for k in keys:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur
print("DIM   layer", g("dim", "layer"), "AUC", g("dim", "auc"),
      "stable rank", g("dim", "extra", "stable_rank", "at_layer"), g("dim", "extra", "stable_rank", "band"))
print("RFM   layer", g("rfm", "layer"), "AUC", g("rfm", "auc"), "weights", g("rfm", "weights"))
print("curve k50 rank", g("curve", "k50_rank"), "max compliance", g("curve", "max_compliance"),
      "at rank", g("curve", "rank_at_max"))
print("sweep verdict", g("sweep", "verdict"), "ablate delta", g("sweep", "ablate_delta_points"), "pts")
errs = {k: v["error"] for k, v in steps.items() if "error" in v}
if errs:
    print("ERRORS:", json.dumps(errs, indent=2))
PY
}

case "$STAGE" in
  preflight) stage_preflight ;;
  sync)      stage_sync ;;
  install)   stage_install ;;
  run)       stage_run ;;
  tail)      stage_tail ;;
  status)    stage_status ;;
  fetch)     stage_fetch ;;
  all)       stage_preflight; stage_sync; stage_install; stage_run; stage_tail ;;
  *)         die "unknown stage '$STAGE'" ;;
esac
