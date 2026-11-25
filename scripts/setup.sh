#!/usr/bin/env bash
# Creates .venv in the project folder and installs everything (macOS / Linux).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ "$(uname)" == "Linux" ]]; then
  # CPU-only PyTorch (much smaller download than the default CUDA build)
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || .venv/bin/python -m pip install torch
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
echo
echo "Setup complete. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv, then press F5."
