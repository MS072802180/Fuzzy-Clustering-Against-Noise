#!/usr/bin/env bash
# Runs tests then the whole study.  Usage: bash scripts/run_all.sh [configs/full.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests
python main.py --config "${1:-configs/full.yaml}"
