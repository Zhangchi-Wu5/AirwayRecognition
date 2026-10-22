#!/usr/bin/env bash
# Run from an extracted server kit; AIRWAY_IMAGE_ROOT may override ../dataset.
set -euo pipefail
KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$KIT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_ROOT="${AIRWAY_IMAGE_ROOT:-../dataset}"
OUTPUT_ROOT="outputs/resnet_fewshot_finetune_20261022"
MODE="${1:-}"
COMMON=(--reference-root fewshot_reference_20261022 --splits-dir fewshot_splits_20261022 --image-root "$IMAGE_ROOT")
case "$MODE" in
  check)
    EXTRA=(--prepare-only)
    if [[ -f "$OUTPUT_ROOT/protocol.json" ]]; then EXTRA+=(--resume); fi
    ;;
  run) EXTRA=(--resume) ;;
  full) EXTRA=(--resume --run-ids full) ;;
  smoke)
    OUTPUT_ROOT="outputs/resnet_fewshot_finetune_smoke_20261022"
    EXTRA=(--smoke-test)
    if [[ -f "$OUTPUT_ROOT/protocol.json" ]]; then EXTRA+=(--resume); fi
    ;;
  *)
    echo "Usage: bash scripts/run_fewshot_finetune_server.sh {check|smoke|full|run}" >&2
    exit 2
    ;;
esac
exec "$PYTHON_BIN" -u -m scripts.run_resnet_fewshot_finetune "${COMMON[@]}" --output-root "$OUTPUT_ROOT" "${EXTRA[@]}"
