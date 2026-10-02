#!/usr/bin/env bash
# Explicitly copy local edited checkpoints to the GPU host. Source sync excludes
# models/ on purpose; this command verifies and publishes complete checkpoints.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HOST=gpu-box
REMOTE_DIR=aiasylum
DRY_RUN=0
BASIS_ROOT=""
usage() {
  cat <<'EOF'
Usage: scripts/remote_models.sh [--host SSH_ALIAS] [--remote-dir DIR] [--dry-run] [--basis-root DIR] all|NAME...

Uploads selected directories from models/ without changing the local originals.
The default target is gpu-box:~/aiasylum/models/. A complete SHA-256 manifest
is saved under runs/model-transfers/. Partial transfers resume outside models/;
verified checkpoints are published by atomic rename. Existing different models
are never overwritten. --dry-run hashes and checks resources without uploading weights.
EOF
}
die() { echo "error: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) [[ $# -ge 2 ]] || die '--host requires a value'; HOST="$2"; shift 2 ;;
    --remote-dir) [[ $# -ge 2 ]] || die '--remote-dir requires a value'; REMOTE_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --basis-root) [[ $# -ge 2 ]] || die "--basis-root requires a value"; BASIS_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --*) die "unknown option: $1" ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || { usage; exit 1; }
[[ "$HOST" =~ ^[A-Za-z0-9][A-Za-z0-9._@-]*$ ]] || die 'host must be a plain SSH alias/address'
[[ "$REMOTE_DIR" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "/$REMOTE_DIR/" != *"/../"* ]] || die 'remote directory must be a relative path inside the SSH home'
[[ -z "$BASIS_ROOT" || ( "$BASIS_ROOT" =~ ^[A-Za-z0-9.][A-Za-z0-9._/-]*$ && "/$BASIS_ROOT/" != *"/../"* ) ]] || die 'basis root must be a relative path inside the remote project'
RSYNC_BIN="${RSYNC_BIN:-rsync}"
if [[ "$RSYNC_BIN" == rsync && -x /opt/homebrew/bin/rsync ]]; then RSYNC_BIN=/opt/homebrew/bin/rsync; fi
command -v "$RSYNC_BIN" >/dev/null || die 'rsync is required locally'
if [[ -x venv/bin/python ]]; then PYTHON=venv/bin/python; else PYTHON=python3; fi
command -v "$PYTHON" >/dev/null || die 'Python 3 is required locally'
AUDIT="runs/model-transfers/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$AUDIT"
HELPER="$AUDIT/checkpoint_transfer.py"
cat > "$HELPER" <<'PY'
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


def fail(message):
    raise SystemExit(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        fail(f'Invalid JSON {path}: {exc}')


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        fail(f'Invalid checkpoint name: {name!r}')
    return name


def inspect_checkpoint(root):
    if root.is_symlink() or not root.is_dir():
        fail(f'Checkpoint must be a real directory: {root}')
    files = sorted(path for path in root.rglob('*') if not path.is_dir())
    if not files:
        fail(f'Empty checkpoint: {root}')
    for path in root.rglob('*'):
        if path.is_symlink():
            fail(f'Symlinks are not supported in checkpoints: {path}')
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            fail(f'Checkpoint contains empty or nonregular file: {path}')
    config = read_json(root / 'config.json')
    if not config.get('model_type'):
        fail(f'Missing model_type in {root}/config.json')
    read_json(root / 'tokenizer_config.json')
    if not any((root / name).is_file() for name in ('tokenizer.json', 'tokenizer.model', 'spiece.model')):
        fail(f'No complete tokenizer found: {root}')
    if (root / 'tokenizer.json').exists():
        read_json(root / 'tokenizer.json')
    read_json(root / 'asylum_surgery.json')
    indexes = [root / name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json') if (root / name).exists()]
    if indexes:
        for index in indexes:
            weights = read_json(index).get('weight_map', {})
            if not weights:
                fail(f'Empty weight map: {index}')
            for name in set(weights.values()):
                path = root / name
                if path.is_absolute() and not path.resolve().is_relative_to(root.resolve()):
                    fail(f'Weight shard outside checkpoint: {name}')
                if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size == 0:
                    fail(f'Missing or invalid indexed shard {root}/{name}')
    elif not any((root / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        fail(f'No complete weights found: {root}')
    return files


def load_manifest(path):
    manifest = read_json(path)
    for model in manifest['checkpoints']:
        safe_name(model['name'])
        for file in model['files']:
            relative = Path(file['path'])
            if relative.is_absolute() or '..' in relative.parts:
                fail('Manifest contains a path outside its checkpoint')
    return manifest


def verify(root, model):
    files = inspect_checkpoint(root)
    expected = {file['path']: file for file in model['files']}
    actual = {str(path.relative_to(root)) for path in files}
    if actual != set(expected):
        fail(f'File list differs for {root}; refusing to overwrite/publish')
    for name, record in expected.items():
        path = root / name
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            fail(f'Content differs for {path}; refusing to overwrite/publish')


def required_new_bytes(manifest, target, audit):
    needed = 0
    reused = 0
    for model in manifest['checkpoints']:
        if (target / model['name']).exists():
            continue
        needed += model['bytes']
        stage = audit / 'staging' / model['sha256'] / model['name']
        if stage.is_symlink() or not stage.is_dir():
            continue
        for record in model['files']:
            path = stage / record['path']
            # Only exact complete files avoid rsync's temporary replacement
            # space. Partial or corrupt files may need a full replacement copy.
            if (path.is_file() and not path.is_symlink()
                    and path.resolve().is_relative_to(stage.resolve())
                    and path.stat().st_size == record['bytes']
                    and digest(path) == record['sha256']):
                needed -= record['bytes']
                reused += record['bytes']
    return needed, reused


def main():
    action = sys.argv[1]
    if action == 'plan':
        root = Path(sys.argv[2])
        names = sys.argv[4:]
        if names == ['all']:
            names = sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith('.'))
        elif 'all' in names:
            fail('Use all alone, or pass individual checkpoint names')
        if not names:
            fail('No checkpoints selected')
        metadata = {}
        result = {'version': 1, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checkpoints': []}
        for name in dict.fromkeys(names):
            safe_name(name)
            checkpoint = root / name
            files = inspect_checkpoint(checkpoint)
            metadata[name] = {'config': read_json(checkpoint / 'config.json'), 'surgery': read_json(checkpoint / 'asylum_surgery.json')}
            print(f'Hashing {name} ({sum(path.stat().st_size for path in files) / 2**30:.2f} GiB)', flush=True)
            records = []
            for path in files:
                before = path.stat()
                sha = digest(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    fail(f'File changed while hashing: {path}')
                records.append({'path': str(path.relative_to(checkpoint)), 'bytes': after.st_size, 'sha256': sha})
            fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            result['checkpoints'].append({'name': name, 'bytes': sum(record['bytes'] for record in records), 'sha256': fingerprint, 'files': records})
        result['total_bytes'] = sum(model['bytes'] for model in result['checkpoints'])
        Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + '\n')
        Path(sys.argv[3]).with_name('metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f"Validated {len(names)} checkpoints; {result['total_bytes'] / 2**30:.2f} GiB", flush=True)
        return
    if action == 'names':
        for model in load_manifest(sys.argv[2])['checkpoints']:
            print(model['name'])
        return
    base = Path.home() / sys.argv[2]
    if not base.is_dir() or not base.resolve().is_relative_to(Path.home().resolve()):
        fail('Remote project directory must exist inside the SSH home')
    audit = base / '.model-transfers'
    lock = audit / 'transfer.lock'
    if action == 'lock':
        audit.mkdir(mode=0o700, exist_ok=True)
        try:
            lock.mkdir(mode=0o700)
        except FileExistsError:
            fail(f'Another transfer owns {lock}. Check that process before removing a stale lock.')
        (lock / 'owner').write_text(sys.argv[3])
        return
    if action == 'unlock':
        if (lock / 'owner').is_file() and (lock / 'owner').read_text() == sys.argv[3]:
            (lock / 'owner').unlink()
            lock.rmdir()
        return
    manifest = load_manifest(sys.argv[3])
    target = base / 'models'
    if target.is_symlink():
        fail('Remote models directory must not be a symlink')
    if action in ('check', 'prepare'):
        free = shutil.disk_usage(base).free
        need, reused = required_new_bytes(manifest, target, audit)
        reserve = 10 * 2**30
        if free < need + reserve:
            fail(f'Insufficient remote disk: {free / 2**30:.1f} GiB free; need {need / 2**30:.1f} GiB plus 10 GiB reserve')
        print(f'Remote disk: {free / 2**30:.1f} GiB free; at most {need / 2**30:.1f} GiB new space; {reused / 2**30:.1f} GiB verified staging reused', flush=True)
        for model in manifest['checkpoints']:
            destination = target / model['name']
            if destination.exists():
                print(f"Verifying existing {model['name']} before skipping", flush=True)
                verify(destination, model)
            elif action == 'prepare':
                stage = audit / 'staging' / model['sha256'] / model['name']
                if stage.is_symlink():
                    fail(f'Staging directory must not be a symlink: {stage}')
                stage.mkdir(parents=True, mode=0o700, exist_ok=True)
        return
    model = next(model for model in manifest['checkpoints'] if model['name'] == sys.argv[4])
    destination = target / model['name']
    stage = audit / 'staging' / model['sha256'] / model['name']
    if action == 'basis':
        weights = [item for item in model['files'] if item['path'].endswith(('.safetensors', '.bin'))]
        for other in manifest['checkpoints']:
            other_weights = [item for item in other['files'] if item['path'].endswith(('.safetensors', '.bin'))]
            candidate = target / other['name']
            if other['name'] != model['name'] and weights == other_weights and candidate.is_dir():
                print(candidate)
                return
        print('NONE')
        return
    if action == 'retire-partials':
        partials = stage / '.rsync-partial'
        if partials.exists():
            retained = audit / 'interrupted'
            retained.mkdir(exist_ok=True)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            os.rename(partials, retained / f"{model['name']}-{stamp}")
        return
    if action == 'stage':
        if destination.exists():
            print('EXISTS')
        else:
            print(stage)
        return
    if action == 'publish':
        if destination.exists():
            verify(destination, model)
            outcome = 'already_present'
        else:
            partials = stage / '.rsync-partial'
            if partials.exists():
                # rsync removes completed partial files itself. A leftover means
                # this upload did not complete, so never publish it.
                if any(partials.iterdir()):
                    fail(f'Incomplete rsync files remain in {partials}')
                partials.rmdir()
            verify(stage, model)
            target.mkdir(mode=0o700, exist_ok=True)
            if destination.exists():
                fail(f'Destination appeared during verification: {destination}')
            os.rename(stage, destination)
            outcome = 'published'
        report = {'name': model['name'], 'status': outcome, 'sha256': model['sha256'], 'bytes': model['bytes'], 'files_verified': len(model['files']), 'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'destination': str(destination)}
        print(json.dumps(report), flush=True)
        return
    fail(f'Unknown action: {action}')


if __name__ == '__main__':
    main()
PY
MANIFEST="$AUDIT/manifest.json"
"$PYTHON" "$HELPER" plan models "$MANIFEST" "$@"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30
          -o ControlMaster=no -o ControlPath=none)
rsh() { ssh -n "${SSH_OPTS[@]}" "$HOST" "$@"; }
rsh "test -d ~/$REMOTE_DIR && command -v python3 >/dev/null && command -v rsync >/dev/null" || die 'remote project, Python 3 and rsync must exist'
REMOTE_AUDIT="$REMOTE_DIR/.model-transfers/$(basename "$AUDIT")"
rsh "mkdir -p ~/$REMOTE_AUDIT && chmod 700 ~/$REMOTE_DIR/.model-transfers ~/$REMOTE_AUDIT"
"$RSYNC_BIN" -rltp -e "ssh ${SSH_OPTS[*]}" "$HELPER" "$MANIFEST" "$AUDIT/metadata.json" "$HOST:$REMOTE_AUDIT/"
REMOTE_HELPER="$REMOTE_AUDIT/checkpoint_transfer.py"
REMOTE_MANIFEST="$REMOTE_AUDIT/manifest.json"
if [[ "$DRY_RUN" == 1 ]]; then
  rsh "python3 ~/$REMOTE_HELPER check '$REMOTE_DIR' ~/$REMOTE_MANIFEST"
  echo "Ready. Dry-run audit: $MANIFEST"
  exit 0
fi
OWNER="$(basename "$AUDIT")"
rsh "python3 ~/$REMOTE_HELPER lock '$REMOTE_DIR' '$OWNER'"
cleanup() { rsh "python3 ~/$REMOTE_HELPER unlock '$REMOTE_DIR' '$OWNER'" || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
rsh "python3 ~/$REMOTE_HELPER prepare '$REMOTE_DIR' ~/$REMOTE_MANIFEST"
"$PYTHON" "$HELPER" names "$MANIFEST" > "$AUDIT/names.txt"
while IFS= read -r NAME; do
  STAGE="$(rsh "python3 ~/$REMOTE_HELPER stage '$REMOTE_DIR' ~/$REMOTE_MANIFEST '$NAME'")"
  if [[ "$STAGE" != EXISTS ]]; then
    [[ "$STAGE" =~ ^/[A-Za-z0-9._/-]+$ ]] || die 'remote staging path contains unsupported characters'
    echo "Uploading $NAME to staging on $HOST (resume supported)"
    RSYNC_ARGS=(-rltp --checksum --partial --partial-dir=.rsync-partial --stats)
    if [[ -n "$BASIS_ROOT" ]]; then
      BASIS="$(rsh "python3 ~/$REMOTE_HELPER basis '$REMOTE_DIR' ~/$REMOTE_MANIFEST '$NAME'")"
      if [[ "$BASIS" == NONE ]]; then
        BASIS="$(rsh "cd ~/$REMOTE_DIR && test -d '$BASIS_ROOT/$NAME' && cd '$BASIS_ROOT/$NAME' && pwd")"
      fi
      [[ "$BASIS" =~ ^/[A-Za-z0-9._/-]+$ ]] || die 'remote basis path contains unsupported characters'
      # Small blocks correct sparse cross-CPU rounding differences without
      # uploading most of a large weight file. A full basis is better than
      # an interrupted prefix; preserve those partial files outside staging.
      rsh "python3 ~/$REMOTE_HELPER retire-partials '$REMOTE_DIR' ~/$REMOTE_MANIFEST '$NAME'"
      RSYNC_ARGS+=(--copy-dest="$BASIS" --block-size=1024 --compress)
    fi
    "$RSYNC_BIN" "${RSYNC_ARGS[@]}" \
      -e "ssh ${SSH_OPTS[*]}" "models/$NAME/" "$HOST:$STAGE/" | tee "$AUDIT/$NAME.rsync.log"
  fi
  echo "Verifying and publishing $NAME"
  rsh "python3 ~/$REMOTE_HELPER publish '$REMOTE_DIR' ~/$REMOTE_MANIFEST '$NAME'" | tee "$AUDIT/$NAME.result.json"
done < "$AUDIT/names.txt"
"$PYTHON" - "$AUDIT" "$HOST" <<'PYREPORT'
import datetime
import json
from pathlib import Path
import sys

audit = Path(sys.argv[1])
manifest = json.loads((audit / "manifest.json").read_text())
results = []
for expected in manifest["checkpoints"]:
    path = audit / f"{expected['name']}.result.json"
    if not path.is_file():
        raise SystemExit(f"Incomplete transfer: no verified result for {expected['name']}")
    result = json.loads(path.read_text())
    if (result["status"] not in ("published", "already_present")
            or result["sha256"] != expected["sha256"]
            or result["bytes"] != expected["bytes"]
            or result["files_verified"] != len(expected["files"])):
        raise SystemExit(f"Incomplete transfer: mismatched result for {expected['name']}")
    results.append(result)
report = {"status": "passed", "host": sys.argv[2], "checkpoints_verified": len(results),
          "files_verified": sum(item["files_verified"] for item in results),
          "bytes_verified": sum(item["bytes"] for item in results),
          "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "checkpoints": results}
(audit / "transfer-report.json").write_text(json.dumps(report, indent=2) + "\n")
PYREPORT
echo "Complete. Originals preserved. Transfer audit: $AUDIT"
