"""Paired patient-level comparison of two classification probability tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parent.parent
CLASS_ORDER = ["lt", "yz", "zz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired accuracy/F1 comparison for patent experiments.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="full")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "patent_evidence" / "data" / "paired_metric_difference.csv",
    )
    return parser.parse_args()


def align_predictions(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    required = {"index", "patient_id", "true_label", "pred_label"}
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
    keys = ["index", "patient_id", "true_label"]
    aligned = baseline[keys + ["pred_label"]].merge(
        candidate[keys + ["pred_label"]],
        on=keys,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("Baseline and candidate do not contain the same test samples.")
    return aligned.drop(columns="_merge")


def metric_values(frame: pd.DataFrame) -> tuple[float, float]:
    accuracy = float((frame["pred_label"] == frame["true_label"]).mean())
    macro_f1 = float(
        f1_score(
            frame["true_label"],
            frame["pred_label"],
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )
    return accuracy, macro_f1


def side_frame(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    return frame[["patient_id", "true_label", f"pred_label_{side}"]].rename(
        columns={f"pred_label_{side}": "pred_label"}
    )


def paired_differences(
    aligned: pd.DataFrame, n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    baseline_values = metric_values(side_frame(aligned, "baseline"))
    candidate_values = metric_values(side_frame(aligned, "candidate"))
    point = {
        "accuracy": candidate_values[0] - baseline_values[0],
        "macro_f1": candidate_values[1] - baseline_values[1],
    }

    patient_ids = aligned["patient_id"].astype(str).unique()
    groups = {
        patient: aligned[aligned["patient_id"].astype(str) == patient]
        for patient in patient_ids
    }
    rng = np.random.default_rng(seed)
    boot = {"accuracy": np.empty(n_boot), "macro_f1": np.empty(n_boot)}
    for index in range(n_boot):
        chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sample = pd.concat([groups[patient] for patient in chosen], ignore_index=True)
        base = metric_values(side_frame(sample, "baseline"))
        candidate = metric_values(side_frame(sample, "candidate"))
        boot["accuracy"][index] = candidate[0] - base[0]
        boot["macro_f1"][index] = candidate[1] - base[1]
    intervals = {
        metric: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for metric, values in boot.items()
    }
    return point, intervals


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline_csv, dtype={"patient_id": str})
    candidate = pd.read_csv(args.candidate_csv, dtype={"patient_id": str})
    aligned = align_predictions(baseline, candidate)
    point, intervals = paired_differences(aligned, args.bootstrap, args.seed)

    baseline_correct = aligned["pred_label_baseline"] == aligned["true_label"]
    candidate_correct = aligned["pred_label_candidate"] == aligned["true_label"]
    baseline_only = int((baseline_correct & ~candidate_correct).sum())
    candidate_only = int((~baseline_correct & candidate_correct).sum())
    discordant = baseline_only + candidate_only
    mcnemar_p = float(binomtest(candidate_only, discordant, p=0.5).pvalue) if discordant else 1.0

    rows = []
    for metric in ("accuracy", "macro_f1"):
        rows.append(
            {
                "comparison": f"{args.candidate_label} - {args.baseline_label}",
                "metric": metric,
                "difference": point[metric],
                "ci_low": intervals[metric][0],
                "ci_high": intervals[metric][1],
                "ci_method": f"配对患者级Bootstrap（{args.bootstrap}次）",
                "mcnemar_exact_p": mcnemar_p if metric == "accuracy" else np.nan,
                "baseline_only_correct": baseline_only,
                "candidate_only_correct": candidate_only,
                "n_images": len(aligned),
                "n_patients": aligned["patient_id"].astype(str).nunique(),
                "seed": args.seed,
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved paired metric differences: {args.output_csv}")


if __name__ == "__main__":
    main()
