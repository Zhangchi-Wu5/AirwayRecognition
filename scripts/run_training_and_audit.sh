#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/airway_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/airway_cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -m scripts.train_model "$@"

python -m scripts.audit_pseudo_features --split test --risk-threshold 0.30 --save-top-k 12
python -m scripts.audit_pseudo_features --split val --risk-threshold 0.30 --save-top-k 12

echo
echo "Done."
echo "Checkpoint: checkpoints/best_model.pt"
echo "Metrics: outputs/test_metrics.txt"
echo "Curves: outputs/training_curves.png"
echo "Confusion matrix: outputs/confusion_matrix.png"
echo "Pseudo-feature audit: outputs/pseudo_feature_audit/"
