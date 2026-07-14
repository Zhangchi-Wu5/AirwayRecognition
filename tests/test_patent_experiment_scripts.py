from __future__ import annotations

import pandas as pd

from scripts.aggregate_patent_experiments import (
    compare_robustness,
    plot_clean,
    plot_robustness,
    summarize_clean,
    summarize_robustness,
    write_summary_markdown,
)
from scripts.compare_patent_predictions import align_predictions, paired_differences
from scripts.export_patent_auc import validate_against_persisted_audit


def _prediction_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.DataFrame(
        {
            "index": [0, 1, 2, 3, 4, 5],
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "true_label": ["lt", "yz", "zz", "lt", "yz", "zz"],
            "pred_label": ["lt", "lt", "zz", "yz", "yz", "lt"],
        }
    )
    candidate = baseline.copy()
    candidate["pred_label"] = ["lt", "yz", "zz", "lt", "yz", "lt"]
    return baseline, candidate


def test_paired_patient_bootstrap_detects_candidate_improvement() -> None:
    baseline, candidate = _prediction_frames()
    aligned = align_predictions(baseline, candidate)
    point, intervals = paired_differences(aligned, n_boot=100, seed=42)

    assert point["accuracy"] > 0
    assert point["macro_f1"] > 0
    assert intervals["accuracy"][0] <= point["accuracy"] <= intervals["accuracy"][1]


def test_aggregate_plots_and_markdown_do_not_require_tabulate(tmp_path) -> None:
    clean = pd.DataFrame(
        [
            {"seed": 42, "variant": "baseline", "accuracy": 0.90, "macro_f1": 0.89, "macro_auc_ovr": 0.95,
             "micro_auc_ovr": 0.95, "auc_lt": 0.96, "auc_yz": 0.94, "auc_zz": 0.95},
            {"seed": 42, "variant": "full", "accuracy": 0.94, "macro_f1": 0.93, "macro_auc_ovr": 0.97,
             "micro_auc_ovr": 0.97, "auc_lt": 0.98, "auc_yz": 0.96, "auc_zz": 0.97},
        ]
    )
    robustness = pd.DataFrame(
        [
            {"seed": 42, "variant": variant, "perturbation": perturbation, "level": level, "accuracy": accuracy}
            for variant, offset in (("baseline", 0.0), ("full", 0.04))
            for perturbation in ("rotation", "glare", "occlusion")
            for level, accuracy in ((0.0, 0.90 + offset), (0.1, 0.80 + offset))
        ]
    )

    clean_summary = summarize_clean(clean)
    robust_summary = summarize_robustness(robustness)
    _, robust_difference_summary = compare_robustness(robustness)
    plot_clean(clean_summary, tmp_path)
    plot_robustness(robust_summary, tmp_path)
    write_summary_markdown(
        tmp_path / "SUMMARY.md",
        clean,
        clean_summary,
        robust_summary,
        robust_difference_summary,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert (tmp_path / "clean_performance_multiseed.pdf").is_file()
    assert (tmp_path / "robustness_baseline_vs_full.png").is_file()
    assert "| seed | variant |" in (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert robust_difference_summary["mean"].gt(0).all()


def test_auc_audit_validation_is_opt_in() -> None:
    predictions = pd.DataFrame()
    assert (
        validate_against_persisted_audit(predictions, "baseline")
        == "not_checked_reference_not_provided"
    )
