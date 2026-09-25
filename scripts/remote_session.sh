#!/usr/bin/env bash
# Run the AI Asylum API on a remote GPU box so that nobody else can use it.
#
# The box is rented by the hour and holds GPUs, model weights and the prompt
# library. An exposed API would let anyone who finds the port load models,
# write multi-gigabyte edits and run up the bill. So this script enforces,
# rather than recommends, four things:
#
#   1. The API binds 127.0.0.1 on the box, never 0.0.0.0. After starting it,
#      the script checks what is actually listening and kills the API if port
#      8000 is reachable on any non-loopback address.
#   2. REQUIRE_AUTH=true with a fresh random key and HMAC secret, generated on
#      the box into a mode-600 .env. Every route then needs the key
#      (api/security.py); the API refuses to start with a weak configuration.
#   3. You reach it only through an SSH tunnel bound to 127.0.0.1 on *your*
#      machine, so nobody else on your network can ride it either.
#   4. Your local .env (provider API keys) and data/ (the whole database) are
#      never copied. Only the prompt-library rows the pipeline needs are sent.
#
# Verbs, in the order a session uses them:
#
#   preflight <host>          read-only: GPU, RAM, disk, python, open ports.
#   sync      <host>          rsync source only (no .env, data, models, runs).
#   install   <host>          venv + interp extra + migrations + prompt library.
#   up        <host>          generate secrets if absent, start the API on loopback,
#                             verify nothing is public, fetch the key to
#                             ~/.config/aiasylum/<host>.key (mode 600).
#   tunnel    <host>          foreground ssh -L 127.0.0.1:8000 -> box loopback.
#   run       <host> <batch>  run scripts/batches/<batch>.json ON the box under
#                             nohup; survives a dropped connection.
#   status    <host>          API health, batch progress, GPU use.
#   matrix-plan <host> [models]      check CUDA/resources for catalog keys or all.
#   matrix-download <host> [models]  download pinned snapshots in the background.
#   matrix-run <host> [models]       validate cached models serially in background.
#   pull      <host>          copy runs/ and manifests back (never weights).
#   down      <host>          stop the API and any batch. Keeps models/ on the box.
#   rotate    <host>          new key + secret; invalidates every session.
#
# Usage:
#   scripts/remote_session.sh preflight gpu-box
#   scripts/remote_session.sh sync gpu-box && scripts/remote_session.sh install gpu-box
#   scripts/remote_session.sh up gpu-box
#   scripts/remote_session.sh run gpu-box phase2_verify
#   scripts/remote_session.sh tunnel gpu-box        # then log in at localhost:3000
#   scripts/remote_session.sh pull gpu-box && scripts/remote_session.sh down gpu-box

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERB="${1:-}"
HOST="${2:-}"
ARG="${3:-}"

REMOTE_DIR="${REMOTE_DIR:-aiasylum}"          # relative to the remote $HOME
API_PORT=8000
LOCAL_PORT="${LOCAL_PORT:-8000}"
KEY_DIR="$HOME/.config/aiasylum"

die() { echo "error: $*" >&2; exit 1; }
[[ -n "$VERB" ]] || die "usage: $0 <preflight|sync|install|up|tunnel|run|matrix-plan|matrix-download|matrix-run|status|pull|down|rotate> <host> [batch/models]"
[[ -n "$HOST" ]] || die "no host given. Pass the ssh alias, e.g. $0 preflight gpu-box"
# The host becomes part of a local filename and a control-socket path.
[[ "$HOST" =~ ^[A-Za-z0-9][A-Za-z0-9._@-]*$ ]] || die "host '$HOST' contains characters this script will not use"
[[ "$REMOTE_DIR" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "/$REMOTE_DIR/" != *"/../"* ]] || die "REMOTE_DIR must be a plain relative path inside the remote home"
[[ "$LOCAL_PORT" =~ ^[0-9]+$ && "$LOCAL_PORT" -ge 1 && "$LOCAL_PORT" -le 65535 ]] || die "LOCAL_PORT must be 1-65535"

KEY_FILE="$KEY_DIR/$(echo "$HOST" | tr '@' '_').key"
SESSION_T0_FILE="$KEY_DIR/$(echo "$HOST" | tr '@' '_').session-start"

# One multiplexed connection for the whole script. The control socket lives in
# ~/.ssh, which only you can read. BatchMode: never fall back to a password
# prompt -- key-based auth only.
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-aiasylum-%r@%h:%p"
          -o ControlPersist=10m -o ConnectTimeout=15 -o BatchMode=yes
          -o ServerAliveInterval=30)
rsh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

# ---------------------------------------------------------------------------

stage_preflight() {
  echo "== preflight on $HOST (read-only) =="
  rsh 'bash -s' <<'REMOTE'
set -euo pipefail
echo "host:     $(hostname)  $(uname -srm)"
echo "python:   $(python3 -V 2>&1 || echo MISSING)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/gpu:      /'
else
  echo "gpu:      no nvidia-smi -- this box cannot run the weights pipeline on GPU"
  exit 2
fi
echo "ram:      $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo "disk:     $(df -h "$HOME" | awk 'NR==2{print $4" free on "$6}')"
echo "hf cache: $(du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | cut -f1 || echo empty)"
echo
echo "listening on non-loopback addresses (anything here is reachable from outside"
echo "unless a firewall says otherwise):"
ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Ev '^(127\.|\[::1\]|::1)' | sed 's/^/  /' || true
REMOTE
  cat <<'EOF'

Use matrix-plan after installation for per-model free VRAM, RAM and disk checks.
The 32B smoke preset estimates 76 GiB of free VRAM on one GPU; multiple smaller
GPUs are not combined. The API will only ever listen on loopback.
EOF
}

stage_sync() {
  echo "== sync source to $HOST:~/$REMOTE_DIR =="
  command -v rsync >/dev/null || die "rsync not installed locally"
  rsh "mkdir -p ~/$REMOTE_DIR && chmod 700 ~/$REMOTE_DIR"

  # The pipeline needs the forbidden-question prompts, not the database. Export
  # just those rows, rather than shipping data/ with every run, user and key in it.
  mkdir -p runs/.session
  venv/bin/python - <<'PY'
import json
from vivasecuris.aiasylum.database import PromptLibrary, get_session
s = get_session()
try:
    rows = s.query(PromptLibrary).filter(PromptLibrary.category == "forbidden_question").all()
    out = [{"name": r.name, "prompt_text": r.prompt_text, "category": r.category,
            "meta_data": r.meta_data or {}} for r in rows]
finally:
    s.close()
open("runs/.session/prompt_library.json", "w").write(json.dumps(out))
print(f"exported {len(out)} forbidden_question prompts")
PY

  # Source only. --delete does not remove excluded paths on the receiver, so the
  # box's own .env, venv, models and runs survive every sync.
  rsync -az --delete --stats \
    --exclude '.git' --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.env' --exclude '.env.*' --exclude '/data' --exclude '/models' \
    --exclude '/runs' --exclude '/frontend' --exclude '/docs/book' --exclude '*.key' \
    -e "ssh ${SSH_OPTS[*]}" \
    ./vivasecuris ./scripts ./config ./alembic ./alembic.ini ./setup.py ./requirements-dev.txt \
    "$HOST:$REMOTE_DIR/"
  rsync -az -e "ssh ${SSH_OPTS[*]}" runs/.session/prompt_library.json "$HOST:$REMOTE_DIR/prompt_library.json"
  rm -f runs/.session/prompt_library.json
  echo "synced (source + prompt library; no .env, no database, no weights)"
}

stage_install() {
  echo "== install on $HOST =="
  rsh "bash -s" <<REMOTE
set -euo pipefail
umask 077
cd ~/$REMOTE_DIR
python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required"'
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --quiet --upgrade pip
# pip does not select a wheel based on the installed driver. Verify CUDA below.
./venv/bin/python -c 'import torch' 2>/dev/null || ./venv/bin/pip install --quiet torch
./venv/bin/pip install --quiet -e '.[interp]'
./venv/bin/python - <<'PY'
import pathlib, shutil, sys, sysconfig
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in PyTorch. Check nvidia-smi and install the matching CUDA wheel from pytorch.org/get-started/locally/ before continuing.")
print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.get_device_name(0))
header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
if not header.is_file() or not shutil.which("cc"):
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    raise SystemExit(
        f"CUDA kernel compilation requires {header} and a C compiler. On Ubuntu/Debian install "
        f"sudo apt-get install python{version}-dev build-essential, then rerun install."
    )
# A random, tiny Qwen3 checks the installed CUDA/kernel stack before downloading
# real weights. Recent PyTorch builds can JIT-compile even an ordinary forward.
from transformers import AutoModelForCausalLM, Qwen3Config
config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, max_position_embeddings=32)
model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(device="cuda", dtype=torch.bfloat16).eval()
tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
with torch.inference_mode():
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
if not torch.isfinite(logits).all():
    raise SystemExit("Tiny Qwen3 CUDA forward produced non-finite logits")
torch.cuda.synchronize()
print("CUDA kernel smoke passed (random tiny Qwen3; no downloads)")
PY
mkdir -p data runs models
./venv/bin/python -m alembic upgrade head
./venv/bin/python - <<'PY'
import json
from vivasecuris.aiasylum.database import PromptLibrary, get_session
rows = json.load(open("prompt_library.json"))
s = get_session()
try:
    have = {n for (n,) in s.query(PromptLibrary.name).filter(PromptLibrary.category == "forbidden_question")}
    new = [PromptLibrary(**r) for r in rows if r["name"] not in have]
    s.add_all(new); s.commit()
    print(f"prompt library: {len(have) + len(new)} forbidden_question rows ({len(new)} new)")
finally:
    s.close()
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-- NO GPU VISIBLE --")
PY
REMOTE
}

stage_up() {
  echo "== start the API on $HOST (loopback only) =="
  rsh "bash -s" <<REMOTE
set -euo pipefail
umask 077
cd ~/$REMOTE_DIR
command -v ss >/dev/null || { echo "ss is required to verify the loopback listener (install iproute2)"; exit 2; }

# Secrets are generated on the box and never leave it except for the one key
# fetched below into a mode-600 file. Existing secrets are kept, so sessions
# survive a restart; use 'rotate' to replace them.
if ! grep -q '^REQUIRE_AUTH=true' .env 2>/dev/null; then
  ./venv/bin/python - <<'PY'
import secrets
lines = [
    "REQUIRE_AUTH=true",
    f"API_KEYS={secrets.token_urlsafe(32)}",
    f"API_KEY_HMAC_SECRET={secrets.token_urlsafe(48)}",
    f"API_SECRET_KEY={secrets.token_urlsafe(48)}",
    f"JWT_SECRET_KEY={secrets.token_urlsafe(48)}",
    # The browser reaches the API through the tunnel as localhost.
    "CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:13000,http://127.0.0.1:13000",
    "SESSION_COOKIE_SECURE=false",
    "DATABASE_URL=sqlite:///./data/aiasylum.db",
]
open(".env", "w").write("\n".join(lines) + "\n")
PY
  chmod 600 .env
  echo "generated a new API key and secrets in ~/$REMOTE_DIR/.env (mode 600)"
fi

if [ -f api.pid ] && kill -0 \$(cat api.pid) 2>/dev/null; then
  kill \$(cat api.pid); sleep 2
fi
# 127.0.0.1, not 0.0.0.0. No --reload: it spawns a watcher and a second process.
nohup ./venv/bin/python -m uvicorn vivasecuris.aiasylum.api.main:app \
  --host 127.0.0.1 --port $API_PORT --no-server-header >> api.log 2>&1 &
echo \$! > api.pid

for i in \$(seq 1 30); do
  curl -sf http://127.0.0.1:$API_PORT/health >/dev/null && break
  sleep 1
done
curl -sf http://127.0.0.1:$API_PORT/health >/dev/null || { echo "API did not come up:"; tail -20 api.log; exit 1; }

# Trust nothing: check what is actually bound. Anything on :$API_PORT that is not
# loopback means the API is reachable from outside, so stop it.
exposed=\$(ss -ltnH "sport = :$API_PORT" 2>/dev/null | awk '{print \$4}' | grep -Ev '^(127\.|\[::1\]|::1)' || true)
if [ -n "\$exposed" ]; then
  kill \$(cat api.pid) 2>/dev/null || true
  echo "REFUSING: port $API_PORT is listening on \$exposed -- stopped the API"; exit 3
fi

# And that auth is really on: an unauthenticated call must be refused.
code=\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$API_PORT/api/v1/weights/stages)
if [ "\$code" != "401" ]; then
  kill \$(cat api.pid) 2>/dev/null || true
  echo "REFUSING: unauthenticated request got \$code, expected 401 -- stopped the API"; exit 3
fi
echo "API up on 127.0.0.1:$API_PORT only; unauthenticated requests are refused (401)"
REMOTE

  mkdir -p "$KEY_DIR" && chmod 700 "$KEY_DIR"
  ( umask 077; rsh "grep '^API_KEYS=' ~/$REMOTE_DIR/.env | cut -d= -f2- | cut -d, -f1" > "$KEY_FILE" )
  chmod 600 "$KEY_FILE"
  [[ -s "$KEY_FILE" ]] || die "could not fetch the API key"
  [[ -f "$SESSION_T0_FILE" ]] || date +%s > "$SESSION_T0_FILE"
  echo "API key saved to $KEY_FILE (mode 600). It is not printed."
  echo "Next:  $0 run $HOST phase2_verify     or     $0 tunnel $HOST"
}

stage_tunnel() {
  [[ -s "$KEY_FILE" ]] || die "no key at $KEY_FILE; run '$0 up $HOST' first"
  cat <<EOF
== tunnel: 127.0.0.1:$LOCAL_PORT (this machine only) -> $HOST 127.0.0.1:$API_PORT ==
Bound to your loopback, so other machines on your network cannot use it.
Start the frontend with NEXT_PUBLIC_API_URL=http://localhost:$LOCAL_PORT, click
Login, and paste the key from:   $KEY_FILE
(e.g.  pbcopy < $KEY_FILE )
Ctrl-C closes the tunnel; the API keeps running on the box until '$0 down'.
EOF
  # A separate connection: the multiplexed master may exit before a long tunnel.
  exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o BatchMode=yes \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$API_PORT" "$HOST"
}

stage_run() {
  local batch="$ARG"
  [[ -n "$batch" ]] || die "usage: $0 run <host> <batch>   (a file in scripts/batches/, without .json)"
  [[ "$batch" =~ ^[A-Za-z0-9_-]+$ ]] || die "batch name '$batch' must be a plain name"
  [[ -f "scripts/batches/$batch.json" ]] || die "no scripts/batches/$batch.json"
  rsync -az -e "ssh ${SSH_OPTS[*]}" "scripts/batches/$batch.json" "$HOST:$REMOTE_DIR/scripts/batches/"
  rsh "bash -s" <<REMOTE
set -euo pipefail
cd ~/$REMOTE_DIR
for f in batch.pid matrix.pid; do
  if [ -f \$f ] && kill -0 \$(cat \$f) 2>/dev/null; then
    echo "\$f is already running; use 'status'"; exit 1
  fi
done
ts=\$(date +%Y%m%d-%H%M%S)
out=runs/session/$batch-\$ts.json
mkdir -p runs/session
nohup ./venv/bin/python scripts/run_batch.py scripts/batches/$batch.json \
  --env .env --out "\$out" > runs/session/$batch-\$ts.log 2>&1 &
echo \$! > batch.pid
echo "started $batch (pid \$(cat batch.pid)); report -> \$out"
REMOTE
  echo "Follow with:  $0 status $HOST"
}

stage_status() {
  rsh "bash -s" <<REMOTE
set -u
cd ~/$REMOTE_DIR
printf 'api:   '; curl -sf http://127.0.0.1:$API_PORT/health || echo "DOWN"; echo
if [ -f batch.pid ] && kill -0 \$(cat batch.pid) 2>/dev/null; then echo "batch: running"; else echo "batch: not running"; fi
if [ -f matrix.pid ] && kill -0 \$(cat matrix.pid) 2>/dev/null; then echo "matrix: running"; else echo "matrix: not running"; fi
if [ -f runs/model-matrix/matrix-report.json ]; then
  ./venv/bin/python - <<'PY'
import json
d = json.load(open("runs/model-matrix/matrix-report.json"))
print("matrix result:", d.get("status"), "phase:", d.get("action"))
for key, row in d.get("results", {}).items():
    print(" ", key, row.get("status"), row.get("error", ""))
for error in d.get("errors", []):
    print(" ", error)
PY
fi
log=\$(ls -t runs/session/*.log 2>/dev/null | head -1)
[ -n "\$log" ] && { echo "--- \$log"; tail -n 25 "\$log"; }
[ -f runs/model-matrix/matrix.log ] && { echo "--- matrix log"; tail -n 10 runs/model-matrix/matrix.log; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu:   /'
REMOTE
}

stage_matrix() {
  local action="${VERB#matrix-}"
  local selected="${ARG:-qwen3-0.6b,qwen3-1.7b,qwen3-8b}"
  [[ "$selected" =~ ^[A-Za-z0-9.,_/-]+$ ]] || die "model selection must contain catalog keys or repository IDs separated by commas, or all"
  if [[ "$action" == "plan" ]]; then
    rsh "cd ~/$REMOTE_DIR && ./venv/bin/python scripts/validate_model_matrix.py plan --models '$selected'"
    return
  fi
  rsh "bash -s" <<REMOTE
set -euo pipefail
cd ~/$REMOTE_DIR
for f in matrix.pid batch.pid; do
  if [ -f \$f ] && kill -0 \$(cat \$f) 2>/dev/null; then
    echo "\$f is already running; use status and wait for it to finish"; exit 1
  fi
done
# Do the inexpensive checks synchronously, before starting a detached task.
./venv/bin/python scripts/validate_model_matrix.py plan --models '$selected'
mkdir -p runs/model-matrix
nohup ./venv/bin/python scripts/validate_model_matrix.py '$action' --models '$selected' --resume \
  >> runs/model-matrix/matrix.log 2>&1 &
echo \$! > matrix.pid
echo "started matrix $action (pid \$(cat matrix.pid)); log: runs/model-matrix/matrix.log"
REMOTE
}

stage_pull() {
  echo "== pull results from $HOST =="
  mkdir -p runs/remote/"$HOST"
  # Reports, directions, sweeps and manifests. Never model weights.
  rsync -az --stats -e "ssh ${SSH_OPTS[*]}" \
    --include '*/' --include '*.json' --include '*.log' --include '*.html' --include '*.npy' --include 'direction.safetensors' \
    --exclude '*' \
    "$HOST:$REMOTE_DIR/runs/" "runs/remote/$HOST/runs/"
  rsync -az -e "ssh ${SSH_OPTS[*]}" --include '*/' --include 'asylum_surgery.json' --exclude '*' \
    "$HOST:$REMOTE_DIR/models/" "runs/remote/$HOST/manifests/" || true
  echo "pulled into runs/remote/$HOST/"
  latest=$(ls -t runs/remote/"$HOST"/runs/session/*.json 2>/dev/null | head -1 || true)
  [[ -n "$latest" ]] && python3 - "$latest" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"\n== {d['batch']}  ({d.get('elapsed_s', 0) / 60:.1f} min) ==")
for s in d["steps"]:
    flag = "FAILED EXPECTATION" if s.get("expect_failures") else s.get("status", "refused")
    print(f"  {s['name']:14s} {flag:20s} {s.get('elapsed_s', 0):7.1f}s  {s.get('error') or ''}")
    for f in s.get("expect_failures") or []:
        print(f"      {f}")
PY
  if [[ -f "$SESSION_T0_FILE" ]]; then
    echo "session wall-clock so far: $(( ($(date +%s) - $(cat "$SESSION_T0_FILE")) / 60 )) min"
  fi
}

stage_down() {
  echo "== stop the API and any batch on $HOST =="
  rsh "bash -s" <<REMOTE
cd ~/$REMOTE_DIR
for f in batch.pid matrix.pid api.pid; do
  [ -f \$f ] && kill \$(cat \$f) 2>/dev/null && echo "stopped \${f%.pid}"
  rm -f \$f
done
true
REMOTE
  if [[ -f "$SESSION_T0_FILE" ]]; then
    echo "session wall-clock: $(( ($(date +%s) - $(cat "$SESSION_T0_FILE")) / 60 )) min"
    rm -f "$SESSION_T0_FILE"
  fi
  echo "models/ kept on the box. Stop the instance itself in your provider's console to stop billing."
}

stage_rotate() {
  echo "== rotate the key and secrets on $HOST =="
  rsh "cd ~/$REMOTE_DIR && rm -f .env"
  stage_up
  echo "every previous key and session is now invalid"
}

case "$VERB" in
  preflight) stage_preflight ;;
  sync)      stage_sync ;;
  install)   stage_install ;;
  up)        stage_up ;;
  tunnel)    stage_tunnel ;;
  run)       stage_run ;;
  matrix-plan|matrix-download|matrix-run) stage_matrix ;;
  status)    stage_status ;;
  pull)      stage_pull ;;
  down)      stage_down ;;
  rotate)    stage_rotate ;;
  *)         die "unknown verb '$VERB'" ;;
esac
