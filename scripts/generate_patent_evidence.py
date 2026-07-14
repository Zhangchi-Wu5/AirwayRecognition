"""Generate a reproducible patent-evidence package from completed experiments.

The script only uses persisted experiment outputs. It does not retrain a model and
does not invent unavailable ROC/AUC values. All summary tables and figures are
written to ``patent_evidence/``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_patent_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_patent_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from src.data import CropBlackBorder


OUT = ROOT / "patent_evidence"
DATA_OUT = OUT / "data"
FIG_OUT = OUT / "figures"

MODEL_CONFIG = [
    {
        "model_key": "baseline",
        "model_name": "ResNet-50基线",
        "short_name": "基线",
        "output_dir": ROOT / "outputs_baseline",
    },
    {
        "model_key": "attention",
        "model_name": "解剖注意力模型",
        "short_name": "注意力",
        "output_dir": ROOT / "outputs_attn",
    },
    {
        "model_key": "regularized",
        "model_name": "注意力+双正则",
        "short_name": "双正则",
        "output_dir": ROOT / "outputs_reg",
    },
    {
        "model_key": "full",
        "model_name": "完整方案（有效视野+注意力+双正则）",
        "short_name": "完整方案",
        "output_dir": ROOT / "outputs_reg_crop",
    },
    {
        "model_key": "hires",
        "model_name": "高分辨率注意力变体",
        "short_name": "高分辨率",
        "output_dir": ROOT / "outputs_reg_crop_hires",
    },
]

CLASS_ORDER = ["lt", "yz", "zz"]
CLASS_CN = {"lt": "隆突", "yz": "右总支气管", "zz": "左总支气管"}
COLORS = {
    "baseline": "#7F8C8D",
    "attention": "#4C78A8",
    "regularized": "#72B7B2",
    "full": "#E45756",
    "hires": "#F2CF5B",
}


def configure_plot_style() -> None:
    """Use a print-friendly style with an explicit Chinese-capable font."""
    candidates = [
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path:
        fm.fontManager.addfont(str(font_path))
        family = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"{stem}.{suffix}")
    plt.close(fig)


def parse_metrics_file(path: Path) -> dict:
    """Parse sklearn's persisted classification report."""
    text = path.read_text(encoding="utf-8")
    accuracy_match = re.search(r"^accuracy=([0-9.]+)", text, flags=re.MULTILINE)
    macro_match = re.search(
        r"^\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
        text,
        flags=re.MULTILINE,
    )
    if not accuracy_match or not macro_match:
        raise ValueError(f"Could not parse metrics file: {path}")
    return {
        "accuracy": float(accuracy_match.group(1)),
        "macro_precision": float(macro_match.group(1)),
        "macro_recall": float(macro_match.group(2)),
        "macro_f1": float(macro_match.group(3)),
        "support": int(macro_match.group(4)),
    }


def audit_path(output_dir: Path) -> Path:
    return output_dir / "pseudo_feature_audit" / "test" / "test_pseudo_feature_audit.csv"


def cluster_bootstrap_metrics(
    df: pd.DataFrame, n_boot: int = 20_000, seed: int = 20260714
) -> dict:
    """Patient-cluster bootstrap CIs for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    groups = {patient: sub for patient, sub in df.groupby("patient_id", sort=True)}
    patient_ids = np.asarray(list(groups))
    accuracies = np.empty(n_boot)
    macro_f1s = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot = pd.concat([groups[patient] for patient in sampled], ignore_index=True)
        correct = boot["true_label"].to_numpy() == boot["pred_label"].to_numpy()
        accuracies[i] = correct.mean()
        macro_f1s[i] = f1_score(
            boot["true_label"],
            boot["pred_label"],
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    return {
        "accuracy_ci_low": float(np.quantile(accuracies, 0.025)),
        "accuracy_ci_high": float(np.quantile(accuracies, 0.975)),
        "macro_f1_ci_low": float(np.quantile(macro_f1s, 0.025)),
        "macro_f1_ci_high": float(np.quantile(macro_f1s, 0.975)),
    }


def paired_cluster_bootstrap_difference(
    baseline: pd.DataFrame,
    full: pd.DataFrame,
    n_boot: int = 20_000,
    seed: int = 20260714,
) -> tuple[float, float, float]:
    """Paired patient-cluster bootstrap for accuracy(full)-accuracy(baseline)."""
    key_cols = ["index", "patient_id"]
    b = baseline[key_cols + ["true_label", "pred_label"]].rename(
        columns={"pred_label": "pred_baseline"}
    )
    f = full[key_cols + ["pred_label"]].rename(columns={"pred_label": "pred_full"})
    paired = b.merge(f, on=key_cols, validate="one_to_one")
    paired["correct_baseline"] = paired["true_label"] == paired["pred_baseline"]
    paired["correct_full"] = paired["true_label"] == paired["pred_full"]
    groups = {patient: sub for patient, sub in paired.groupby("patient_id", sort=True)}
    patient_ids = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot = pd.concat([groups[patient] for patient in sampled], ignore_index=True)
        diffs[i] = boot["correct_full"].mean() - boot["correct_baseline"].mean()
    point = float(paired["correct_full"].mean() - paired["correct_baseline"].mean())
    return point, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def build_dataset_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    split_frames = []
    patient_sets = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(ROOT / "data_splits" / f"{split}.csv", dtype={"patient_id": str})
        patient_sets[split] = set(df["patient_id"])
        for label in CLASS_ORDER:
            sub = df[df["label"] == label]
            split_frames.append(
                {
                    "split": split,
                    "split_cn": {"train": "训练集", "val": "验证集", "test": "测试集"}[split],
                    "class": label,
                    "class_cn": CLASS_CN[label],
                    "images": len(sub),
                    "patients_with_class": sub["patient_id"].nunique(),
                }
            )
    overlap_rows = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap_rows.append(
            {
                "split_pair": f"{left}-{right}",
                "overlapping_patients": len(patient_sets[left] & patient_sets[right]),
            }
        )
    return pd.DataFrame(split_frames), pd.DataFrame(overlap_rows)


def build_model_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    class_rows = []
    audits = {}
    for cfg in MODEL_CONFIG:
        metrics = parse_metrics_file(cfg["output_dir"] / "test_metrics.txt")
        audit = pd.read_csv(audit_path(cfg["output_dir"]), dtype={"patient_id": str}).sort_values("index")
        audits[cfg["model_key"]] = audit
        bootstrap = cluster_bootstrap_metrics(audit)
        summary_rows.append(
            {
                "model_key": cfg["model_key"],
                "model_name": cfg["model_name"],
                "short_name": cfg["short_name"],
                **metrics,
                **bootstrap,
                "mean_confidence": audit["confidence"].mean(),
                "median_confidence": audit["confidence"].median(),
                "errors": int((audit["true_label"] != audit["pred_label"]).sum()),
            }
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            audit["true_label"],
            audit["pred_label"],
            labels=CLASS_ORDER,
            zero_division=0,
        )
        for idx, label in enumerate(CLASS_ORDER):
            class_rows.append(
                {
                    "model_key": cfg["model_key"],
                    "model_name": cfg["model_name"],
                    "class": label,
                    "class_cn": CLASS_CN[label],
                    "precision": precision[idx],
                    "recall_sensitivity": recall[idx],
                    "f1": f1[idx],
                    "support": int(support[idx]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows), audits


def build_statistical_comparison(
    summary: pd.DataFrame,
    audits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    baseline = audits["baseline"]
    full = audits["full"]
    merged = baseline[["index", "true_label", "pred_label"]].rename(
        columns={"pred_label": "pred_baseline"}
    ).merge(
        full[["index", "pred_label"]].rename(columns={"pred_label": "pred_full"}),
        on="index",
        validate="one_to_one",
    )
    correct_baseline = merged["true_label"] == merged["pred_baseline"]
    correct_full = merged["true_label"] == merged["pred_full"]
    baseline_only = int((correct_baseline & ~correct_full).sum())
    full_only = int((~correct_baseline & correct_full).sum())
    mcnemar_p = float(binomtest(full_only, baseline_only + full_only, p=0.5).pvalue)
    diff, diff_low, diff_high = paired_cluster_bootstrap_difference(baseline, full)

    s = summary.set_index("model_key")
    error_reduction = 1 - s.loc["full", "errors"] / s.loc["baseline", "errors"]
    return pd.DataFrame(
        [
            {
                "comparison": "完整方案 vs ResNet-50基线",
                "metric": "测试准确率绝对差",
                "estimate": diff,
                "ci_low": diff_low,
                "ci_high": diff_high,
                "test": "患者级配对Bootstrap 95%CI；McNemar精确检验",
                "p_value": mcnemar_p,
                "notes": f"基线独对{baseline_only}例；完整方案独对{full_only}例",
            },
            {
                "comparison": "完整方案 vs ResNet-50基线",
                "metric": "错误数相对减少",
                "estimate": error_reduction,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "test": "描述性统计",
                "p_value": np.nan,
                "notes": f"{int(s.loc['baseline', 'errors'])}例降至{int(s.loc['full', 'errors'])}例",
            },
        ]
    )


def plot_dataset_distribution(dataset: pd.DataFrame) -> None:
    pivot = dataset.pivot(index="split_cn", columns="class_cn", values="images").loc[
        ["训练集", "验证集", "测试集"], [CLASS_CN[c] for c in CLASS_ORDER]
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for color, label in zip(colors, pivot.columns):
        values = pivot[label].to_numpy()
        bars = ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.62)
        for bar, value, base in zip(bars, values, bottom):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                str(int(value)),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        bottom += values
    for i, total in enumerate(bottom):
        ax.text(i, total + 7, f"合计 {int(total)}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("图像数量（张）")
    ax.set_ylim(0, max(bottom) * 1.14)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
    save_figure(fig, "fig1_dataset_distribution")


def plot_model_performance(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(summary))
    colors = [COLORS[key] for key in summary["model_key"]]
    acc = summary["accuracy"].to_numpy() * 100
    f1 = summary["macro_f1"].to_numpy() * 100
    acc_low = summary["accuracy_ci_low"].to_numpy() * 100
    acc_high = summary["accuracy_ci_high"].to_numpy() * 100
    f1_low = summary["macro_f1_ci_low"].to_numpy() * 100
    f1_high = summary["macro_f1_ci_high"].to_numpy() * 100
    for i, color in enumerate(colors):
        ax.errorbar(
            acc[i],
            y[i] - 0.12,
            xerr=[[acc[i] - acc_low[i]], [acc_high[i] - acc[i]]],
            fmt="o",
            markersize=7,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
        )
        ax.errorbar(
            f1[i],
            y[i] + 0.12,
            xerr=[[f1[i] - f1_low[i]], [f1_high[i] - f1[i]]],
            fmt="s",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
        )
    ax.scatter([], [], s=48, color="#666666", marker="o", label="准确率（患者级95%CI）")
    ax.scatter([], [], s=48, facecolor="white", edgecolor="#666666", marker="s", label="宏平均F1（患者级95%CI）")
    for i, (a, m) in enumerate(zip(acc, f1)):
        ax.text(a + 0.08, i - 0.12, f"{a:.2f}", va="center", fontsize=8)
        ax.text(m + 0.08, i + 0.12, f"{m:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, summary["short_name"])
    ax.invert_yaxis()
    ax.set_xlim(90.5, 100.7)
    ax.set_xlabel("内部患者级留出测试集性能（%，局部放大）")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    save_figure(fig, "fig2_model_performance")


def plot_class_performance(class_table: pd.DataFrame) -> None:
    selected = class_table[class_table["model_key"].isin(["baseline", "full"])].copy()
    pivot = selected.pivot(index="class_cn", columns="model_key", values="f1").loc[
        [CLASS_CN[c] for c in CLASS_ORDER]
    ]
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    base = pivot["baseline"].to_numpy() * 100
    full = pivot["full"].to_numpy() * 100
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [base[i], full[i]], color="#BBBBBB", linewidth=1.4, zorder=1)
    ax.scatter(x - 0.035, base, s=68, color=COLORS["baseline"], label="ResNet-50基线", zorder=3)
    ax.scatter(x + 0.035, full, s=68, color=COLORS["full"], label="完整方案", zorder=3)
    for i, value in enumerate(base):
        ax.text(x[i] - 0.06, value + 0.30, f"{value:.2f}", ha="right", va="bottom", fontsize=8)
    for i, value in enumerate(full):
        ax.text(x[i] + 0.06, value + 0.30, f"{value:.2f}", ha="left", va="bottom", fontsize=8)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("F1分数（%，局部放大）")
    ax.set_ylim(90, 101.2)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    save_figure(fig, "fig3_class_f1_comparison")


def plot_confusion_matrices(audits: dict[str, pd.DataFrame]) -> None:
    cmap = LinearSegmentedColormap.from_list("patent_red", ["#FFFFFF", "#F9C5C5", "#E45756"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ("baseline", "full"),
        ("ResNet-50基线", "完整方案"),
    ):
        df = audits[key]
        cm = confusion_matrix(df["true_label"], df["pred_label"], labels=CLASS_ORDER)
        row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if row_pct[i, j] > 60 else "#333333"
                ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)", ha="center", va="center", color=color)
        ax.set_xticks(range(3), [CLASS_CN[c] for c in CLASS_ORDER])
        ax.set_yticks(range(3), [CLASS_CN[c] for c in CLASS_ORDER])
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(label, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    save_figure(fig, "fig4_confusion_matrix_comparison")


def build_fov_qc_table() -> pd.DataFrame:
    """Measure the rectangular frame retained by effective-field-of-view cropping."""
    split = pd.read_csv(ROOT / "data_splits" / "test.csv", dtype={"patient_id": str})
    cropper = CropBlackBorder()
    rows = []
    for index, row in split.iterrows():
        original = Image.open(row["path"]).convert("RGB")
        cropped = cropper(original)
        original_area = original.width * original.height
        cropped_area = cropped.width * cropped.height
        rows.append(
            {
                "index": index,
                "patient_id": row["patient_id"],
                "label": row["label"],
                "path": row["path"],
                "original_width": original.width,
                "original_height": original.height,
                "cropped_width": cropped.width,
                "cropped_height": cropped.height,
                "retained_rectangular_area_ratio": cropped_area / original_area,
                "removed_rectangular_area_ratio": 1 - cropped_area / original_area,
            }
        )
    return pd.DataFrame(rows)


def plot_effective_fov_comparison() -> None:
    """Show only the deterministic input processing; no unaligned heatmap overlays."""
    sources = pd.read_csv(DATA_OUT / "fov_visual_sources.csv")
    cropper = CropBlackBorder()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.8), constrained_layout=True)
    for row_idx, source in sources.iterrows():
        original = Image.open(ROOT / source["relative_path"]).convert("RGB")
        cropped = cropper(original)
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 1].imshow(cropped)
        axes[row_idx, 0].text(
            0.02,
            0.05,
            source["class_cn"],
            transform=axes[row_idx, 0].transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2},
        )
        for col in range(2):
            axes[row_idx, col].axis("off")
    axes[0, 0].set_title("A  原始气管镜图像", pad=7)
    axes[0, 1].set_title("B  有效视野处理后", pad=7)
    save_figure(fig, "fig5_effective_fov_comparison")


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    for cfg in MODEL_CONFIG:
        history = pd.read_csv(cfg["output_dir"] / "training_history.csv")
        history = history[history["stage"] == 2].reset_index(drop=True)
        x = np.arange(1, len(history) + 1)
        color = COLORS[cfg["model_key"]]
        axes[0].plot(x, history["val_acc"] * 100, marker="o", markersize=2.7, linewidth=1.2, color=color, label=cfg["short_name"])
        axes[1].plot(x, history["val_loss"], marker="o", markersize=2.7, linewidth=1.2, color=color, label=cfg["short_name"])
    axes[0].set_xlabel("全网络微调轮次")
    axes[0].set_ylabel("验证集准确率（%）")
    axes[0].set_ylim(87, 101)
    axes[1].set_xlabel("全网络微调轮次")
    axes[1].set_ylabel("验证集损失")
    for ax in axes:
        ax.grid(color="#DDDDDD", linewidth=0.6, alpha=0.8)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, "fig7_validation_curves")


def write_machine_summary(
    dataset: pd.DataFrame,
    model_summary: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    totals = dataset.groupby("split", sort=False)["images"].sum()
    patients = {
        split: pd.read_csv(ROOT / "data_splits" / f"{split}.csv", dtype={"patient_id": str})["patient_id"].nunique()
        for split in ("train", "val", "test")
    }
    s = model_summary.set_index("model_key")
    payload = {
        "dataset": {
            "total_images": int(totals.sum()),
            "total_patients": int(pd.read_csv(ROOT / "data_splits" / "manifest.csv", dtype={"patient_id": str})["patient_id"].nunique()),
            "splits": {
                split: {"images": int(totals[split]), "patients": int(patients[split])}
                for split in ("train", "val", "test")
            },
        },
        "primary_comparison": {
            "baseline_accuracy": float(s.loc["baseline", "accuracy"]),
            "full_accuracy": float(s.loc["full", "accuracy"]),
            "accuracy_absolute_difference": float(s.loc["full", "accuracy"] - s.loc["baseline", "accuracy"]),
            "baseline_macro_f1": float(s.loc["baseline", "macro_f1"]),
            "full_macro_f1": float(s.loc["full", "macro_f1"]),
            "baseline_errors": int(s.loc["baseline", "errors"]),
            "full_errors": int(s.loc["full", "errors"]),
        },
        "gradcam_audit_status": "excluded: persisted heatmaps and masks used inconsistent geometry",
        "auc_status": "not_computable_from_persisted_outputs_without_per-class probabilities or checkpoints",
        "statistical_comparison": stats.replace({np.nan: None}).to_dict(orient="records"),
    }
    (DATA_OUT / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    exploratory_dir = OUT / "exploratory_unvalidated"
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    # Quarantine figures/tables generated from the legacy unaligned Grad-CAM audit.
    legacy_paths = [
        DATA_OUT / "attention_audit_summary.csv",
        DATA_OUT / "qualitative_sources.csv",
        FIG_OUT / "fig5_dark_border_attention.pdf",
        FIG_OUT / "fig5_dark_border_attention.png",
        FIG_OUT / "fig6_qualitative_gradcam_comparison.pdf",
        FIG_OUT / "fig6_qualitative_gradcam_comparison.png",
        FIG_OUT / "fig7_validation_curves.pdf",
        FIG_OUT / "fig7_validation_curves.png",
    ]
    for legacy in legacy_paths:
        if legacy.exists():
            legacy.replace(exploratory_dir / legacy.name)

    test_split = pd.read_csv(ROOT / "data_splits" / "test.csv", dtype={"patient_id": str})
    patient_counts = test_split.groupby("patient_id")["label"].nunique()
    visual_patient = patient_counts[patient_counts == 3].index[0]
    visual_rows = test_split[test_split["patient_id"] == visual_patient].copy()
    visual_rows["class_cn"] = visual_rows["label"].map(CLASS_CN)
    visual_rows["relative_path"] = visual_rows["path"].map(
        lambda value: str(Path(value).resolve().relative_to(ROOT))
    )
    visual_rows = visual_rows.set_index("label").loc[CLASS_ORDER].reset_index()
    visual_rows[["patient_id", "label", "class_cn", "relative_path"]].to_csv(
        DATA_OUT / "fov_visual_sources.csv", index=False
    )

    dataset, overlaps = build_dataset_tables()
    summary, class_table, audits = build_model_tables()
    stats = build_statistical_comparison(summary, audits)
    fov_qc = build_fov_qc_table()

    dataset.to_csv(DATA_OUT / "dataset_distribution.csv", index=False)
    overlaps.to_csv(DATA_OUT / "patient_overlap_check.csv", index=False)
    summary.to_csv(DATA_OUT / "model_performance.csv", index=False)
    class_table.to_csv(DATA_OUT / "class_performance.csv", index=False)
    fov_qc.to_csv(DATA_OUT / "effective_fov_qc.csv", index=False)
    stats.to_csv(DATA_OUT / "statistical_comparison.csv", index=False)
    for key, audit in audits.items():
        audit[
            [
                "index",
                "path",
                "patient_id",
                "true_label",
                "pred_label",
                "pred_label_cn",
                "confidence",
            ]
        ].to_csv(DATA_OUT / f"paired_predictions_{key}.csv", index=False)

    plot_dataset_distribution(dataset)
    plot_model_performance(summary)
    plot_class_performance(class_table)
    plot_confusion_matrices(audits)
    plot_effective_fov_comparison()
    write_machine_summary(dataset, summary, stats)
    print(f"Generated patent evidence package: {OUT}")


if __name__ == "__main__":
    main()
