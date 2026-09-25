#!/usr/bin/env bash
# Keep latest-model dependencies separate from the validated API/interp venv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_PYTHON="${BASE_PYTHON:-venv/bin/python}"
BENCHMARK_VENV="${BENCHMARK_VENV:-venv-benchmark}"
"$BASE_PYTHON" - "$BENCHMARK_VENV" <<'PY'
from pathlib import Path
import subprocess
import sys
import sysconfig

root = Path.cwd()
target = Path(sys.argv[1]).resolve()
original_site = sysconfig.get_paths()['purelib']
if target == Path(sys.prefix).resolve():
    raise SystemExit('The benchmark environment must be separate from the base environment')
if not (target / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
site = subprocess.check_output([str(target / 'bin/python'), '-c',
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
# The new venv's packages precede this path, so Transformers is upgraded only
# there while Torch/CUDA and the app's other dependencies are reused.
Path(site, 'aiasylum_shared_runtime.pth').write_text(original_site + '\n' + str(root) + '\n')
print('Benchmark Python:', target / 'bin/python')
PY
"$BENCHMARK_VENV/bin/python" -m pip install --upgrade 'transformers==5.17.0' 'mistral-common==1.12.0' 'datasets==5.0.1'
"$BENCHMARK_VENV/bin/python" - <<'PY'
import torch
import transformers
print('Transformers:', transformers.__version__, transformers.__file__)
print('Reused Torch:', torch.__version__, torch.__file__)
PY
