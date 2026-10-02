#!/usr/bin/env bash
# Write CycloneDX SBOMs for the default Python install and the frontend lockfile.
# Outputs:
#   sbom/python-core.cdx.json   packages installed by requirements.txt
#   sbom/frontend.cdx.json      packages in frontend/package-lock.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p sbom

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
    PYTHON_BOOTSTRAP=python3
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Installing the default Python requirements into a temporary environment..."
"$PYTHON_BOOTSTRAP" -m venv "$WORK/app"
"$WORK/app/bin/python" -m pip install --upgrade pip
"$WORK/app/bin/python" -m pip install -r requirements.txt
"$WORK/app/bin/python" -m pip freeze --exclude pip --exclude setuptools --exclude wheel > "$WORK/freeze.txt"

echo "Installing CycloneDX into a separate environment..."
"$PYTHON_BOOTSTRAP" -m venv "$WORK/cdx"
"$WORK/cdx/bin/python" -m pip install --upgrade pip
"$WORK/cdx/bin/python" -m pip install 'cyclonedx-bom>=5,<7'

"$WORK/cdx/bin/cyclonedx-py" requirements "$WORK/freeze.txt" \
    --output-format JSON \
    --output-file sbom/python-core.cdx.json \
    --spec-version 1.6

echo "Writing the frontend SBOM from frontend/package-lock.json..."
(cd frontend && npx --yes @cyclonedx/cyclonedx-npm@1 \
    --output-file ../sbom/frontend.cdx.json \
    --spec-version 1.6 \
    --package-lock-only \
    --omit dev)

echo "Wrote sbom/python-core.cdx.json and sbom/frontend.cdx.json"
