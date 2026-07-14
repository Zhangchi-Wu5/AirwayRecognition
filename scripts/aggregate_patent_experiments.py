"""Aggregate multi-seed patent experiments and generate comparison figures.

Expected layout is produced by ``scripts/run_patent_gpu_experiments.sh``::

    patent_runs/seed_42/baseline/...
    patent_runs/seed_42/full/...
    patent_runs/seed_2026/baseline/...

The aggregation is CPU-only and can be rerun after interrupted/resumed GPU jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_patent_aggregate_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_patent_aggregate_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent.parent
VARIANT_NAMES = {
    "baseline": "Baseline",
    "crop_only": "Crop only",
    "attention": "Attention",
    "regularized": "Attention + regularization",
    "full": "Full method",
}
COLORS = {
    "baseline": "#7F8C8D",
    "crop_only": "#F2CF5B",
    "attention": "#4C78A8",
    "regularized": "#72B7B2",
    "full": "#E45756",
}
VARIANT_ORDER = ["baseline", "crop_only", "attention", "regularized", "full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed patent experiments.")
    parser.add_argument("--run-root", type=Path, default=ROOT / "patent_runs")
    parser.add_argument("--summary-dir", type=Path, default=None)
    return parser.parse_args()


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def seed_from_dir(seed_dir: Path) -> int:
    return int(seed_dir.name.removeprefix("seed_"))


def collect_clean_metrics(run_root: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = seed_from_dir(seed_dir)
        for variant_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            variant = variant_dir.name
            probability_path = variant_dir / "data" / f"probabilities_{variant}.csv"
            auc_path = variant_dir / "data" / f"auc_{variant}.csv"
            if not probability_path.exists() or not auc_path.exists():
                continue
            predictions = pd.read_csv(probability_path)
            auc_table = pd.read_csv(auc_path)
            accuracy = accuracy_score(predictions["true_label"], predictions["pred_label"])
            macro_f1 = f1_score(
                predictions["true_label"],
                predictions["pred_label"],
                labels=["lt", "yz", "zz"],
                average="macro",
                zero_division=0,
            )
            auc_values = auc_table.set_index("auc_type")["auc"]
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "macro_auc_ovr": float(auc_values["macro_ovr"]),
                    "micro_auc_ovr": float(auc_values["micro_ovr"]),
                    "auc_lt": float(auc_values["lt"]),
                    "auc_yz": float(auc_values["yz"]),
                    "auc_zz": float(auc_values["zz"]),
                    "n_images": len(predictions),
                    "n_patients": predictions["patient_id"].astype(str).nunique(),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No complete probability/AUC outputs found under {run_root}")
    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


def summarize_clean(clean: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "macro_auc_ovr", "micro_auc_ovr", "auc_lt", "auc_yz", "auc_zz"]
    summary = clean.groupby("variant")[metrics].agg(["mean", "std", "min", "max", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["_order"] = summary["variant"].map(
        {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    ).fillna(len(VARIANT_ORDER))
    return summary.sort_values(["_order", "variant"]).drop(columns="_order").reset_index(drop=True)


def validate_run_configs(run_root: Path, clean: pd.DataFrame) -> pd.DataFrame:
    """Fail closed if completed runs did not use exactly the same patient split."""
    rows = []
    for record in clean[["seed", "variant"]].itertuples(index=False):
        path = run_root / f"seed_{record.seed}" / record.variant / "run_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing run configuration: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        fingerprints = config.get("split_fingerprints")
        if not isinstance(fingerprints, dict) or set(fingerprints) != {"train", "val", "test"}:
            raise ValueError(f"Run configuration lacks split fingerprints: {path}")
        rows.append(
            {
                "seed": record.seed,
                "variant": record.variant,
                "split_seed": config.get("resolved_split_seed"),
                "train_fingerprint": fingerprints["train"],
                "val_fingerprint": fingerprints["val"],
                "test_fingerprint": fingerprints["test"],
                "git_commit": config.get("git_commit", "unavailable"),
            }
        )
    audit = pd.DataFrame(rows)
    if audit["split_seed"].nunique(dropna=False) != 1:
        raise ValueError("Completed runs used different patient split seeds.")
    for column in ("train_fingerprint", "val_fingerprint", "test_fingerprint"):
        if audit[column].nunique(dropna=False) != 1:
            raise ValueError(f"Completed runs used different split membership: {column}")
    return audit


def collect_robustness(run_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_root.glob("seed_*/*/robustness.csv")):
        seed = seed_from_dir(path.parents[1])
        variant = path.parent.name
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["variant"] = variant
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_robustness(robustness: pd.DataFrame) -> pd.DataFrame:
    if robustness.empty:
        return pd.DataFrame()
    return (
        robustness.groupby(["variant", "perturbation", "level"])["accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


def compare_robustness(robustness: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired full-minus-baseline accuracy at each stress level."""
    if robustness.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["seed", "perturbation", "level"]
    baseline = robustness[robustness["variant"] == "baseline"][keys + ["accuracy"]].rename(
        columns={"accuracy": "accuracy_baseline"}
    )
    full = robustness[robustness["variant"] == "full"][keys + ["accuracy"]].rename(
        columns={"accuracy": "accuracy_full"}
    )
    paired = baseline.merge(full, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        return paired, pd.DataFrame()
    paired["accuracy_difference"] = paired["accuracy_full"] - paired["accuracy_baseline"]
    summary = (
        paired.groupby(["perturbation", "level"])["accuracy_difference"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    return paired, summary


def collect_comparison_tables(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    auc_frames = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = seed_from_dir(seed_dir)
        metric_path = seed_dir / "paired_metric_difference_full_vs_baseline.csv"
        auc_path = seed_dir / "auc_paired_difference_full_vs_baseline.csv"
        if metric_path.exists():
            frame = pd.read_csv(metric_path)
            frame.insert(0, "training_seed", seed)
            metric_frames.append(frame)
        if auc_path.exists():
            frame = pd.read_csv(auc_path)
            frame.insert(0, "training_seed", seed)
            auc_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    auc = pd.concat(auc_frames, ignore_index=True) if auc_frames else pd.DataFrame()
    return metrics, auc


def collect_attention_qc(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.glob("seed_*/*/pseudo_feature_audit/test/test_pseudo_feature_audit.csv")):
        seed = seed_from_dir(path.parents[3])
        variant = path.parents[2].name
        frame = pd.read_csv(path)
        required = {
            "dark_border_attention_score",
            "specular_highlight_attention_score",
            "dark_border_attention_enrichment",
            "specular_highlight_attention_enrichment",
            "dark_border_area_ratio",
            "specular_highlight_area_ratio",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"Audit file predates aligned/enrichment fix: {path}")
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "dark_attention_mean": frame["dark_border_attention_score"].mean(),
                "dark_enrichment_mean": frame["dark_border_attention_enrichment"].mean(),
                "dark_area_mean": frame["dark_border_area_ratio"].mean(),
                "specular_attention_mean": frame["specular_highlight_attention_score"].mean(),
                "specular_enrichment_mean": frame["specular_highlight_attention_enrichment"].mean(),
                "specular_area_mean": frame["specular_highlight_area_ratio"].mean(),
                "n_images": len(frame),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"{stem}.{suffix}")
    plt.close(fig)


def plot_clean(clean_summary: pd.DataFrame, figure_dir: Path) -> None:
    variants = clean_summary["variant"].tolist()
    labels = [VARIANT_NAMES.get(variant, variant) for variant in variants]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    specs = [
        ("accuracy", "Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("macro_auc_ovr", "Macro OvR AUC (%)"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs):
        means = clean_summary[f"{metric}_mean"].to_numpy() * 100
        stds = clean_summary[f"{metric}_std"].fillna(0).to_numpy() * 100
        colors = [COLORS.get(variant, "#777777") for variant in variants]
        for x_value, mean, std, color in zip(x, means, stds, colors):
            ax.errorbar(
                x_value,
                mean,
                yerr=std,
                fmt="none",
                ecolor=color,
                capsize=3,
                linewidth=1.1,
            )
        ax.scatter(x, means, s=55, color=colors, zorder=3)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        lower = max(0, min(means - stds) - 1.5)
        ax.set_ylim(lower, 100.5)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    save_figure(fig, figure_dir, "clean_performance_multiseed")


def plot_robustness(robust_summary: pd.DataFrame, figure_dir: Path) -> None:
    if robust_summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    for ax, perturbation in zip(axes, ["rotation", "glare", "occlusion"]):
        for variant in ["baseline", "full"]:
            sub = robust_summary[
                (robust_summary["perturbation"] == perturbation)
                & (robust_summary["variant"] == variant)
            ].sort_values("level")
            if sub.empty:
                continue
            x = sub["level"].to_numpy()
            mean = sub["mean"].to_numpy() * 100
            std = sub["std"].fillna(0).to_numpy() * 100
            color = COLORS[variant]
            ax.plot(x, mean, marker="o", linewidth=1.7, color=color, label=VARIANT_NAMES[variant])
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        ax.set_xlabel("Degrees" if perturbation == "rotation" else "Area coverage")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 101)
        ax.set_title(perturbation.capitalize())
        ax.grid(color="#DDDDDD", linewidth=0.6)
    axes[0].legend(frameon=False)
    save_figure(fig, figure_dir, "robustness_baseline_vs_full")


def write_summary_markdown(
    output_path: Path,
    clean: pd.DataFrame,
    clean_summary: pd.DataFrame,
    robust_summary: pd.DataFrame,
    robust_difference_summary: pd.DataFrame,
    attention: pd.DataFrame,
    paired_metrics: pd.DataFrame,
    paired_auc: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render a small DataFrame without requiring the optional tabulate package."""
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:.4f}"
                )
        headers = [str(column) for column in display.columns]
        rows = [headers, ["---"] * len(headers)]
        rows.extend(
            [str(value).replace("|", "\\|") for value in row]
            for row in display.itertuples(index=False, name=None)
        )
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

    lines = [
        "# Patent Experiment Summary",
        "",
        "> Auto-generated from multi-seed GPU runs. Interpret clean-set differences together with robustness results.",
        "",
        "## Completed runs",
        "",
        markdown_table(clean[["seed", "variant", "accuracy", "macro_f1", "macro_auc_ovr"]]),
        "",
        "## Across-seed summary",
        "",
        markdown_table(clean_summary),
        "",
    ]
    if not robust_summary.empty:
        lines += ["## Robustness summary", "", markdown_table(robust_summary), ""]
    if not robust_difference_summary.empty:
        lines += [
            "## Robustness difference (full - baseline)",
            "",
            markdown_table(robust_difference_summary),
            "",
        ]
    if not paired_metrics.empty:
        lines += [
            "## Paired patient-level clean metric differences",
            "",
            markdown_table(paired_metrics),
            "",
        ]
    if not paired_auc.empty:
        lines += [
            "## Paired patient-level AUC differences",
            "",
            markdown_table(paired_auc),
            "",
        ]
    if not attention.empty:
        lines += [
            "## Aligned Grad-CAM pseudo-feature QC (exploratory)",
            "",
            "> Automated masks are secondary evidence; anatomical validity still requires blinded expert review.",
            "",
            markdown_table(attention),
            "",
        ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    summary_dir = (args.summary_dir or run_root / "summary").resolve()
    figure_dir = summary_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    clean = collect_clean_metrics(run_root)
    config_audit = validate_run_configs(run_root, clean)
    clean_summary = summarize_clean(clean)
    robustness = collect_robustness(run_root)
    robust_summary = summarize_robustness(robustness)
    robust_differences, robust_difference_summary = compare_robustness(robustness)
    attention = collect_attention_qc(run_root)
    paired_metrics, paired_auc = collect_comparison_tables(run_root)

    clean.to_csv(summary_dir / "clean_metrics_by_seed.csv", index=False)
    config_audit.to_csv(summary_dir / "run_config_audit.csv", index=False)
    clean_summary.to_csv(summary_dir / "clean_metrics_summary.csv", index=False)
    if not robustness.empty:
        robustness.to_csv(summary_dir / "robustness_by_seed.csv", index=False)
        robust_summary.to_csv(summary_dir / "robustness_summary.csv", index=False)
    if not robust_differences.empty:
        robust_differences.to_csv(summary_dir / "robustness_differences_by_seed.csv", index=False)
        robust_difference_summary.to_csv(
            summary_dir / "robustness_difference_summary.csv", index=False
        )
    if not attention.empty:
        attention.to_csv(summary_dir / "attention_qc_by_seed.csv", index=False)
    if not paired_metrics.empty:
        paired_metrics.to_csv(summary_dir / "paired_metric_differences.csv", index=False)
    if not paired_auc.empty:
        paired_auc.to_csv(summary_dir / "paired_auc_differences.csv", index=False)

    plot_clean(clean_summary, figure_dir)
    plot_robustness(robust_summary, figure_dir)
    write_summary_markdown(
        summary_dir / "SUMMARY.md",
        clean,
        clean_summary,
        robust_summary,
        robust_difference_summary,
        attention,
        paired_metrics,
        paired_auc,
    )
    manifest = {
        "run_root": str(run_root),
        "seeds": sorted(int(value) for value in clean["seed"].unique()),
        "variants": sorted(clean["variant"].unique().tolist()),
        "completed_clean_runs": len(clean),
        "has_robustness": not robustness.empty,
        "has_paired_robustness_differences": not robust_differences.empty,
        "has_aligned_attention_qc": not attention.empty,
        "has_paired_clean_metric_differences": not paired_metrics.empty,
        "has_paired_auc_differences": not paired_auc.empty,
        "split_configuration_validation": "passed",
    }
    (summary_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Aggregated results: {summary_dir}")


if __name__ == "__main__":
    main()
