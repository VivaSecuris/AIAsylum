#!/usr/bin/env bash
# A separate UI and authenticated API tunnel for the remote GPU server.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
remote_host="${1:-gpu-box}"
[[ "$remote_host" =~ ^[A-Za-z0-9][A-Za-z0-9._@-]*$ ]] || { echo "Invalid SSH host" >&2; exit 1; }
# Do not launch a second UI against an unrelated or already-owned local port.
if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:18000 -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Cannot open GPU API tunnel: local port 18000 is already in use. Stop its existing listener before starting this UI." >&2
  exit 1
fi
task_tmp="$(mktemp -d "${TMPDIR:-/tmp}/aiasylum-webui.XXXXXX")"
tunnel_log="$task_tmp/ssh-error.log"
tunnel_pid=""
ui_pid=""
# Separate background process groups let cleanup stop npm's Next.js children
# and SSH proxy children as well as their direct parent processes.
set -m
cleanup() {
  trap '' INT TERM
  if [[ -n "$tunnel_pid" ]]; then
    kill -- "-$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
  fi
  if [[ -n "$ui_pid" ]]; then
    kill -- "-$ui_pid" 2>/dev/null || true
    wait "$ui_pid" 2>/dev/null || true
  fi
  rm -rf "$task_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_tunnel() {
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
    -o ConnectionAttempts=1 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
    -L 127.0.0.1:18000:127.0.0.1:8000 "$remote_host" 2>"$tunnel_log" &
  tunnel_pid=$!
  tunnel_started=$SECONDS
}
check_bind_failure() {
  if grep -Eqi 'address already in use|cannot listen to port: 18000|could not request local forwarding' "$tunnel_log"; then
    echo "Cannot open GPU API tunnel: local port 18000 could not be bound. Check for an existing listener; automatic reconnect has stopped." >&2
    exit 1
  fi
}

echo "Opening http://127.0.0.1:13000 (GPU API via 127.0.0.1:18000)."
echo "Login key: ~/.config/aiasylum/${remote_host//@/_}.key"
start_tunnel
sleep 1
if ! kill -0 "$tunnel_pid" 2>/dev/null; then check_bind_failure; fi
cd frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:18000 NEXT_DIST_DIR=.next-gpu npm run dev -- --hostname 127.0.0.1 --port 13000 &
ui_pid=$!

# This loop belongs to the UI process lifetime; it never restarts the remote
# API. Short network/egress changes retry with a delay capped at 30 seconds.
retry_delay=1
retry_at=0
while kill -0 "$ui_pid" 2>/dev/null; do
  if [[ -n "$tunnel_pid" ]] && ! kill -0 "$tunnel_pid" 2>/dev/null; then
    kill -- "-$tunnel_pid" 2>/dev/null || true
    wait "$tunnel_pid" 2>/dev/null || true
    tunnel_pid=""
    check_bind_failure
    if (( SECONDS - tunnel_started >= 60 )); then retry_delay=1; fi
    echo "GPU API tunnel disconnected; reconnecting in ${retry_delay}s (UI remains available)." >&2
    retry_at=$((SECONDS + retry_delay))
    retry_delay=$((retry_delay * 2))
    if (( retry_delay > 30 )); then retry_delay=30; fi
  fi
  if [[ -z "$tunnel_pid" ]] && (( SECONDS >= retry_at )); then start_tunnel; fi
  sleep 1
done

ui_status=0
wait "$ui_pid" || ui_status=$?
exit "$ui_status"
