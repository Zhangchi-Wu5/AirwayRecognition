"""Export real multiclass ROC/AUC data from a saved model checkpoint.

This utility is intentionally separate from the aggregate patent analysis because
ROC/AUC requires the probability assigned to *every* class for every test image.
The previously persisted ``test_metrics.txt`` and audit CSVs only contain the
winning-class confidence, which is insufficient for a valid multiclass ROC curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_auc_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_auc_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from src.attention import build_attention_resnet50
from src.data import get_eval_transforms
from src.models import build_resnet50


ROOT = Path(__file__).resolve().parent.parent
CLASS_ORDER = ["lt", "yz", "zz"]
CLASS_CN = {"lt": "隆突", "yz": "右总支气管", "zz": "左总支气管"}
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_ORDER)}
COLORS = ["#4C78A8", "#F58518", "#54A24B"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test probabilities and real ROC/AUC values.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True, help="Short file-safe label, e.g. baseline or full.")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data_splits")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "patent_evidence")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--attn-hires", action="store_true")
    parser.add_argument("--crop-border", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--reference-audit-csv",
        type=Path,
        default=None,
        help="Optional audit CSV from the same run; predictions must match exactly.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    font_path = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        family = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def select_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    if args.attention:
        model = build_attention_resnet50(
            num_classes=3,
            pretrained=False,
            dropout=args.dropout,
            hires=args.attn_hires,
        )
    else:
        model = build_resnet50(num_classes=3, pretrained=False, dropout=args.dropout)
    try:
        state = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # torch<2.0 compatibility
        state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probabilities(predictions: pd.DataFrame) -> None:
    proba = predictions[["prob_lt", "prob_yz", "prob_zz"]].to_numpy(dtype=float)
    if not np.isfinite(proba).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values.")
    if ((proba < 0) | (proba > 1)).any():
        raise ValueError("Predicted probabilities fall outside [0, 1].")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Per-image class probabilities do not sum to 1.")


def validate_against_persisted_audit(
    predictions: pd.DataFrame,
    model_label: str,
    reference_audit_csv: Path | None = None,
) -> str:
    """Fail closed when an audit from the same run is explicitly provided.

    Model labels are reused across random seeds, so silently selecting an older
    label-matched audit can compare two different checkpoints.  The caller must
    therefore provide the exact same-run audit path when this check is wanted.
    """
    del model_label  # kept in the signature for backward compatibility
    if reference_audit_csv is None:
        return "not_checked_reference_not_provided"
    audit_path = reference_audit_csv
    if not audit_path.exists():
        return "not_checked_reference_missing"
    reference = pd.read_csv(audit_path)[["index", "true_label", "pred_label"]]
    current = predictions[["index", "true_label", "pred_label"]]
    merged = reference.merge(
        current,
        on="index",
        how="outer",
        suffixes=("_reference", "_current"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Current prediction indices do not match the persisted audit.")
    true_match = merged["true_label_reference"] == merged["true_label_current"]
    pred_match = merged["pred_label_reference"] == merged["pred_label_current"]
    if not (true_match & pred_match).all():
        mismatch = merged.loc[~(true_match & pred_match), "index"].tolist()
        raise ValueError(
            "Checkpoint predictions do not reproduce the persisted audit; "
            f"mismatched indices: {mismatch[:10]}"
        )
    return "passed"


def collect_probabilities(
    model: torch.nn.Module,
    split_df: pd.DataFrame,
    transform,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    records = []
    tensors = []
    rows = []

    def flush() -> None:
        if not tensors:
            return
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()
        for row, proba in zip(rows, probabilities):
            pred_id = int(np.argmax(proba))
            records.append(
                {
                    "index": int(row["index"]),
                    "patient_id": str(row["patient_id"]),
                    "path": row["path"],
                    "true_label": row["label"],
                    "pred_label": CLASS_ORDER[pred_id],
                    "prob_lt": float(proba[0]),
                    "prob_yz": float(proba[1]),
                    "prob_zz": float(proba[2]),
                }
            )
        tensors.clear()
        rows.clear()

    for index, row in split_df.reset_index(drop=True).iterrows():
        image = Image.open(row["path"]).convert("RGB")
        tensors.append(transform(image))
        rows.append({**row.to_dict(), "index": index})
        if len(tensors) >= batch_size:
            flush()
    flush()
    return pd.DataFrame(records)


def auc_point_estimates(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["true_label"].map(CLASS_TO_ID).to_numpy()
    y_binary = label_binarize(y, classes=np.arange(3))
    proba = predictions[["prob_lt", "prob_yz", "prob_zz"]].to_numpy()
    estimates = {
        label: float(roc_auc_score(y_binary[:, idx], proba[:, idx]))
        for idx, label in enumerate(CLASS_ORDER)
    }
    estimates["macro_ovr"] = float(roc_auc_score(y_binary, proba, average="macro", multi_class="ovr"))
    estimates["micro_ovr"] = float(roc_auc_score(y_binary, proba, average="micro", multi_class="ovr"))
    return estimates


def patient_bootstrap_auc(
    predictions: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    groups = {patient: sub for patient, sub in predictions.groupby("patient_id", sort=True)}
    patient_ids = np.asarray(list(groups))
    samples = {key: [] for key in [*CLASS_ORDER, "macro_ovr", "micro_ovr"]}
    attempts = 0
    while len(samples["macro_ovr"]) < n_boot and attempts < n_boot * 2:
        attempts += 1
        chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot = pd.concat([groups[patient] for patient in chosen], ignore_index=True)
        try:
            estimates = auc_point_estimates(boot)
        except ValueError:
            continue
        for key, value in estimates.items():
            samples[key].append(value)
    if len(samples["macro_ovr"]) < n_boot:
        raise RuntimeError("Too many invalid bootstrap samples; verify each class is represented.")
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in samples.items()
    }


def plot_roc(predictions: pd.DataFrame, model_label: str, output_dir: Path) -> None:
    y = predictions["true_label"].map(CLASS_TO_ID).to_numpy()
    y_binary = label_binarize(y, classes=np.arange(3))
    proba = predictions[["prob_lt", "prob_yz", "prob_zz"]].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    for idx, (label, color) in enumerate(zip(CLASS_ORDER, COLORS)):
        fpr, tpr, _ = roc_curve(y_binary[:, idx], proba[:, idx])
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{CLASS_CN[label]}（AUC={auc(fpr, tpr):.3f}）")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("假阳性率（1-特异度）")
    ax.set_ylabel("真阳性率（敏感度）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#DDDDDD", linewidth=0.6, alpha=0.8)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / "figures" / f"roc_{model_label}.{suffix}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.output_dir / "data"
    figure_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    device = select_device(args.device)
    model = build_model(args, device)
    split_df = pd.read_csv(args.splits_dir / f"{args.split}.csv", dtype={"patient_id": str})
    transform = get_eval_transforms(crop_border=args.crop_border)
    predictions = collect_probabilities(model, split_df, transform, device, args.batch_size)
    validate_probabilities(predictions)
    audit_validation = validate_against_persisted_audit(
        predictions, args.model_label, args.reference_audit_csv
    )
    checkpoint_sha256 = sha256_file(args.checkpoint_path)
    estimates = auc_point_estimates(predictions)
    intervals = patient_bootstrap_auc(predictions, args.bootstrap, args.seed)

    rows = []
    for key in [*CLASS_ORDER, "macro_ovr", "micro_ovr"]:
        rows.append(
            {
                "model_label": args.model_label,
                "auc_type": key,
                "auc": estimates[key],
                "ci_low": intervals[key][0],
                "ci_high": intervals[key][1],
                "ci_method": f"患者级Bootstrap（{args.bootstrap}次）",
                "n_images": len(predictions),
                "n_patients": predictions["patient_id"].nunique(),
                "checkpoint_sha256": checkpoint_sha256,
                "audit_prediction_validation": audit_validation,
                "attention_model": bool(args.attention),
                "attention_hires": bool(args.attn_hires),
                "crop_border": bool(args.crop_border),
            }
        )
    predictions.to_csv(data_dir / f"probabilities_{args.model_label}.csv", index=False)
    pd.DataFrame(rows).to_csv(data_dir / f"auc_{args.model_label}.csv", index=False)
    metadata = {
        "model_label": args.model_label,
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "split": args.split,
        "n_images": len(predictions),
        "n_patients": int(predictions["patient_id"].nunique()),
        "class_probability_sum_check": "passed",
        "persisted_audit_prediction_check": audit_validation,
        "bootstrap_requested": args.bootstrap,
        "bootstrap_effective": args.bootstrap,
        "bootstrap_seed": args.seed,
        "device": device,
        "model_flags": {
            "attention": bool(args.attention),
            "attn_hires": bool(args.attn_hires),
            "crop_border": bool(args.crop_border),
            "dropout": args.dropout,
        },
    }
    (data_dir / f"auc_metadata_{args.model_label}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_roc(predictions, args.model_label, args.output_dir)
    print(f"Saved probabilities: {data_dir / f'probabilities_{args.model_label}.csv'}")
    print(f"Saved AUC table: {data_dir / f'auc_{args.model_label}.csv'}")
    print(f"Saved ROC figure: {figure_dir / f'roc_{args.model_label}.pdf'}")


if __name__ == "__main__":
    main()
