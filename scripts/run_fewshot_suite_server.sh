#!/usr/bin/env bash
# Run from a Git checkout. Data stay on the server; only code is synchronized.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_fewshot_suite_server.sh {check|run|pack} [options]" >&2
  exit 2
fi
exec "$PYTHON_BIN" -u -m scripts.run_resnet_fewshot_suite "$@"
