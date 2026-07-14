"""Compute paired patient-bootstrap AUC differences between two saved probability tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_patent_auc import (
    CLASS_ORDER,
    auc_point_estimates,
    validate_probabilities,
)


ROOT = Path(__file__).resolve().parent.parent
AUC_KEYS = [*CLASS_ORDER, "macro_ovr", "micro_ovr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of two multiclass AUC results.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="full")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "patent_evidence" / "data" / "auc_paired_difference.csv",
    )
    return parser.parse_args()


def align_predictions(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    validate_probabilities(baseline)
    validate_probabilities(candidate)
    keys = ["index", "patient_id", "true_label"]
    prob_cols = ["prob_lt", "prob_yz", "prob_zz"]
    aligned = baseline[keys + prob_cols].merge(
        candidate[keys + prob_cols],
        on=keys,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("Baseline and candidate probability tables do not contain the same test samples.")
    return aligned.drop(columns="_merge")


def side_frame(aligned: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": aligned["index"],
            "patient_id": aligned["patient_id"].astype(str),
            "true_label": aligned["true_label"],
            **{
                column: aligned[f"{column}_{suffix}"]
                for column in ["prob_lt", "prob_yz", "prob_zz"]
            },
        }
    )


def paired_auc_differences(
    aligned: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    baseline = side_frame(aligned, "baseline")
    candidate = side_frame(aligned, "candidate")
    point_baseline = auc_point_estimates(baseline)
    point_candidate = auc_point_estimates(candidate)
    point = {key: point_candidate[key] - point_baseline[key] for key in AUC_KEYS}

    patient_ids = aligned["patient_id"].astype(str).unique()
    groups = {patient: aligned[aligned["patient_id"].astype(str) == patient] for patient in patient_ids}
    rng = np.random.default_rng(seed)
    diffs = {key: [] for key in AUC_KEYS}
    attempts = 0
    while len(diffs["macro_ovr"]) < n_boot and attempts < n_boot * 2:
        attempts += 1
        chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot = pd.concat([groups[patient] for patient in chosen], ignore_index=True)
        try:
            base_auc = auc_point_estimates(side_frame(boot, "baseline"))
            candidate_auc = auc_point_estimates(side_frame(boot, "candidate"))
        except ValueError:
            continue
        for key in AUC_KEYS:
            diffs[key].append(candidate_auc[key] - base_auc[key])
    if len(diffs["macro_ovr"]) < n_boot:
        raise RuntimeError("Too many invalid bootstrap samples; verify every class is represented.")
    intervals = {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in diffs.items()
    }
    return point, intervals


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline_csv, dtype={"patient_id": str})
    candidate = pd.read_csv(args.candidate_csv, dtype={"patient_id": str})
    aligned = align_predictions(baseline, candidate)
    point, intervals = paired_auc_differences(aligned, args.bootstrap, args.seed)
    rows = []
    for key in AUC_KEYS:
        rows.append(
            {
                "comparison": f"{args.candidate_label} - {args.baseline_label}",
                "auc_type": key,
                "auc_difference": point[key],
                "ci_low": intervals[key][0],
                "ci_high": intervals[key][1],
                "ci_method": f"配对患者级Bootstrap（{args.bootstrap}次）",
                "n_images": len(aligned),
                "n_patients": aligned["patient_id"].nunique(),
                "seed": args.seed,
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved paired AUC differences: {args.output_csv}")


if __name__ == "__main__":
    main()
