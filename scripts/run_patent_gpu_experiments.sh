#!/usr/bin/env bash
# Resumable GPU experiment pipeline for patent evidence.
#
# Recommended first pass:
#   bash scripts/run_patent_gpu_experiments.sh
#
# Full five-variant ablation after the core result looks promising:
#   MODE=full bash scripts/run_patent_gpu_experiments.sh
#
# All settings can be overridden as environment variables; see docs/PATENT_GPU_EXPERIMENTS.md.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-core}"
SEEDS="${SEEDS:-42 2026 3407}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/dataset}"
SPLITS_DIR="${SPLITS_DIR:-$PROJECT_ROOT/data_splits_patent}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/patent_runs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
ROBUST_SEEDS="${ROBUST_SEEDS:-10}"
ROBUST_BASE_SEED="${ROBUST_BASE_SEED:-20260714}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
RUN_AUDIT="${RUN_AUDIT:-1}"
AUDIT_TOP_K="${AUDIT_TOP_K:-12}"

case "$MODE" in
  core)
    VARIANTS=(baseline full)
    ;;
  full)
    VARIANTS=(baseline crop_only attention regularized full)
    ;;
  *)
    echo "MODE must be 'core' or 'full'; got: $MODE" >&2
    exit 2
    ;;
esac

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory does not exist: $DATASET_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$SPLITS_DIR"
MASTER_LOG="$RUN_ROOT/logs/pipeline_${MODE}.log"

run_logged() {
  echo "[$(date '+%F %T')] RUN: $*" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
}

if [[ "$DEVICE" == cuda* ]]; then
  "$PYTHON_BIN" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print("GPU:", torch.cuda.get_device_name(0))' \
    2>&1 | tee -a "$MASTER_LOG"
fi

"$PYTHON_BIN" -c \
  'import numpy, pandas, scipy, sklearn, torch, torchvision; print("Python dependencies: OK")' \
  2>&1 | tee -a "$MASTER_LOG"

read -r -a SEED_ARRAY <<< "$SEEDS"

for seed in "${SEED_ARRAY[@]}"; do
  seed_dir="$RUN_ROOT/seed_$seed"
  mkdir -p "$seed_dir"

  for variant in "${VARIANTS[@]}"; do
    variant_dir="$seed_dir/$variant"
    checkpoint="$variant_dir/best_model.pt"
    mkdir -p "$variant_dir"

    train_flags=()
    eval_flags=()
    audit_pf_mask="combined"
    case "$variant" in
      baseline)
        ;;
      crop_only)
        train_flags+=(--crop-border)
        eval_flags+=(--crop-border)
        ;;
      attention)
        train_flags+=(--attention)
        eval_flags+=(--attention)
        ;;
      regularized)
        train_flags+=(--attention --reg --pf-mask combined)
        eval_flags+=(--attention)
        ;;
      full)
        train_flags+=(--attention --reg --crop-border --pf-mask specular_highlight)
        eval_flags+=(--attention --crop-border)
        audit_pf_mask="specular_highlight"
        ;;
      *)
        echo "Unknown variant: $variant" >&2
        exit 2
        ;;
    esac

    config_is_current=0
    if [[ -s "$variant_dir/run_config.json" ]] && grep -q '"split_fingerprints"' "$variant_dir/run_config.json"; then
      config_is_current=1
    fi
    if [[ -s "$checkpoint" && -s "$variant_dir/test_predictions.csv" && "$config_is_current" == "1" ]]; then
      echo "[$(date '+%F %T')] SKIP completed training: seed=$seed variant=$variant" | tee -a "$MASTER_LOG"
    else
      run_logged "$PYTHON_BIN" -m scripts.train_model \
        --dataset-dir "$DATASET_DIR" \
        --splits-dir "$SPLITS_DIR" \
        --output-dir "$variant_dir" \
        --checkpoint-path "$checkpoint" \
        --seed "$seed" \
        --split-seed "$SPLIT_SEED" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --stage1-epochs "$STAGE1_EPOCHS" \
        --stage2-epochs "$STAGE2_EPOCHS" \
        --early-stopping-patience "$PATIENCE" \
        "${train_flags[@]}"
    fi

    audit_csv="$variant_dir/pseudo_feature_audit/test/test_pseudo_feature_audit.csv"
    audit_is_current=0
    if [[ -s "$audit_csv" ]] && head -n 1 "$audit_csv" | grep -q "configured_attention_enrichment"; then
      audit_is_current=1
    fi
    should_audit=0
    if [[ "$RUN_AUDIT" == "1" && ( "$variant" == "baseline" || "$variant" == "full" ) ]]; then
      should_audit=1
    fi
    if [[ "$should_audit" == "1" ]]; then
      if [[ "$audit_is_current" == "1" ]]; then
        echo "[$(date '+%F %T')] SKIP completed audit: seed=$seed variant=$variant" | tee -a "$MASTER_LOG"
      else
        run_logged "$PYTHON_BIN" -m scripts.audit_pseudo_features \
          --checkpoint-path "$checkpoint" \
          --splits-dir "$SPLITS_DIR" \
          --output-dir "$variant_dir/pseudo_feature_audit" \
          --split test \
          --device "$DEVICE" \
          --pf-mask "$audit_pf_mask" \
          --save-top-k "$AUDIT_TOP_K" \
          "${eval_flags[@]}"
        audit_is_current=1
      fi
    fi

    probability_csv="$variant_dir/data/probabilities_${variant}.csv"
    if [[ -s "$probability_csv" && -s "$variant_dir/data/auc_${variant}.csv" ]]; then
      echo "[$(date '+%F %T')] SKIP completed AUC: seed=$seed variant=$variant" | tee -a "$MASTER_LOG"
    else
      auc_args=(
        --checkpoint-path "$checkpoint"
        --model-label "$variant"
        --splits-dir "$SPLITS_DIR"
        --split test
        --output-dir "$variant_dir"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --bootstrap "$BOOTSTRAP"
        --seed "$seed"
      )
      if [[ "$audit_is_current" == "1" ]]; then
        auc_args+=(--reference-audit-csv "$audit_csv")
      fi
      run_logged "$PYTHON_BIN" -m scripts.export_patent_auc "${auc_args[@]}" "${eval_flags[@]}"
    fi

    robustness_csv="$variant_dir/robustness.csv"
    if [[ -s "$robustness_csv" ]]; then
      echo "[$(date '+%F %T')] SKIP completed robustness: seed=$seed variant=$variant" | tee -a "$MASTER_LOG"
    else
      run_logged "$PYTHON_BIN" -m scripts.robustness_eval \
        --checkpoint-path "$checkpoint" \
        --splits-dir "$SPLITS_DIR" \
        --split test \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --seed "$ROBUST_BASE_SEED" \
        --seeds "$ROBUST_SEEDS" \
        --label "$variant" \
        --output-csv "$robustness_csv" \
        "${eval_flags[@]}"
    fi
  done

  baseline_prob="$seed_dir/baseline/data/probabilities_baseline.csv"
  full_prob="$seed_dir/full/data/probabilities_full.csv"
  if [[ -s "$baseline_prob" && -s "$full_prob" ]]; then
    metric_diff="$seed_dir/paired_metric_difference_full_vs_baseline.csv"
    if [[ ! -s "$metric_diff" ]]; then
      run_logged "$PYTHON_BIN" -m scripts.compare_patent_predictions \
        --baseline-csv "$baseline_prob" \
        --candidate-csv "$full_prob" \
        --bootstrap "$BOOTSTRAP" \
        --seed "$seed" \
        --output-csv "$metric_diff"
    fi

    auc_diff="$seed_dir/auc_paired_difference_full_vs_baseline.csv"
    if [[ ! -s "$auc_diff" ]]; then
      run_logged "$PYTHON_BIN" -m scripts.compare_patent_auc \
        --baseline-csv "$baseline_prob" \
        --candidate-csv "$full_prob" \
        --bootstrap "$BOOTSTRAP" \
        --seed "$seed" \
        --output-csv "$auc_diff"
    fi
  fi
done

run_logged "$PYTHON_BIN" -m scripts.aggregate_patent_experiments \
  --run-root "$RUN_ROOT" \
  --summary-dir "$RUN_ROOT/summary"

echo "[$(date '+%F %T')] DONE. Summary: $RUN_ROOT/summary/SUMMARY.md" | tee -a "$MASTER_LOG"
