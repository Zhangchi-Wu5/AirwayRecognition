"""Audit Grad-CAM overlap with dark-border and specular-highlight masks."""
import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from src.attention import build_attention_resnet50
from src.data import get_eval_geometry_transforms, get_eval_transforms, ID_TO_LABEL, LABEL_NAMES_CN
from src.models import build_resnet50
from src.viz import (
    build_pseudo_feature_masks,
    make_gradcam_heatmap,
    pseudo_feature_attention_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pseudo-feature attention on a split.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "checkpoints" / "best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "pseudo_feature_audit")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--attention", action="store_true",
                        help="Load the anatomy-attention ResNet-50 (must match how the checkpoint was trained).")
    parser.add_argument("--attn-hires", action="store_true",
                        help="Build the attention model in hires (layer3, 14x14) mode — match the checkpoint.")
    parser.add_argument("--crop-border", action="store_true",
                        help="Remove the scope black border before inference/scoring (match training).")
    parser.add_argument("--pf-mask", choices=["combined", "specular_highlight", "dark_border"], default="combined",
                        help="Which pseudo-feature mask the high-risk score is based on (match training).")
    parser.add_argument("--risk-threshold", type=float, default=0.30)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick server smoke runs.")
    parser.add_argument("--save-top-k", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dark-threshold", type=int, default=24)
    parser.add_argument("--bright-threshold", type=int, default=235)
    parser.add_argument("--color-spread-threshold", type=int, default=35)
    parser.add_argument("--min-highlight-area", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    split_csv = args.splits_dir / f"{args.split}.csv"
    split_df = pd.read_csv(split_csv)
    if args.max_samples is not None:
        split_df = split_df.head(args.max_samples).copy()

    out_dir = args.output_dir / args.split
    samples_dir = out_dir / "sample_images"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if args.attention:
        model = build_attention_resnet50(num_classes=3, pretrained=False, dropout=args.dropout,
                                         hires=args.attn_hires).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        target_layer = model.gradcam_target_layer()
    else:
        model = build_resnet50(num_classes=3, pretrained=False, dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        target_layer = model.layer4[-1]
    transform = get_eval_transforms(crop_border=args.crop_border)
    display_transform = get_eval_geometry_transforms(crop_border=args.crop_border)

    records = []
    for idx, row in split_df.reset_index(drop=True).iterrows():
        path = Path(row["path"])
        original = Image.open(path).convert("RGB")
        image_tensor = transform(original)

        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            proba = torch.softmax(logits, dim=1)[0].cpu()
            pred_id = int(proba.argmax().item())
            confidence = float(proba[pred_id].item())

        heatmap = make_gradcam_heatmap(
            model=model,
            image_tensor=image_tensor,
            target_class=pred_id,
            target_layer=target_layer,
            device=device,
        )
        # Critical alignment invariant: display image and masks use exactly the same
        # PIL geometry as the tensor from which ``heatmap`` was computed.
        mask_src = display_transform(original)
        if mask_src.size != (heatmap.shape[1], heatmap.shape[0]):
            raise RuntimeError(
                f"Aligned display image has size {mask_src.size}, heatmap is "
                f"{heatmap.shape[1]}x{heatmap.shape[0]}"
            )
        masks = build_pseudo_feature_masks(
            mask_src,
            dark_threshold=args.dark_threshold,
            bright_threshold=args.bright_threshold,
            color_spread_threshold=args.color_spread_threshold,
            min_highlight_area=args.min_highlight_area,
        )
        pseudo_score = pseudo_feature_attention_score(heatmap, masks[args.pf_mask])
        dark_score = pseudo_feature_attention_score(heatmap, masks["dark_border"])
        highlight_score = pseudo_feature_attention_score(heatmap, masks["specular_highlight"])

        dark_area_ratio = float(masks["dark_border"].mean())
        highlight_area_ratio = float(masks["specular_highlight"].mean())
        configured_area_ratio = float(masks[args.pf_mask].mean())
        is_high_risk = pseudo_score >= args.risk_threshold
        records.append({
            "index": idx,
            "path": str(path),
            "patient_id": row.get("patient_id", ""),
            "true_label": row.get("label", ID_TO_LABEL.get(int(row["label_id"]), "")),
            "pred_label": ID_TO_LABEL[pred_id],
            "pred_label_cn": LABEL_NAMES_CN[ID_TO_LABEL[pred_id]],
            "confidence": confidence,
            "pseudo_attention_score": pseudo_score,
            "dark_border_attention_score": dark_score,
            "specular_highlight_attention_score": highlight_score,
            "dark_border_area_ratio": dark_area_ratio,
            "specular_highlight_area_ratio": highlight_area_ratio,
            "configured_mask_area_ratio": configured_area_ratio,
            "dark_border_attention_enrichment": (
                dark_score / dark_area_ratio if dark_area_ratio > 0 else 0.0
            ),
            "specular_highlight_attention_enrichment": (
                highlight_score / highlight_area_ratio if highlight_area_ratio > 0 else 0.0
            ),
            "configured_attention_enrichment": (
                pseudo_score / configured_area_ratio if configured_area_ratio > 0 else 0.0
            ),
            "high_risk": is_high_risk,
        })

        print(
            f"[{idx + 1}/{len(split_df)}] pred={ID_TO_LABEL[pred_id]} conf={confidence:.3f} "
            f"pseudo_score={pseudo_score:.3f} high_risk={is_high_risk} path={path.name}"
        )

    audit_df = pd.DataFrame(records).sort_values("pseudo_attention_score", ascending=False)
    audit_csv = out_dir / f"{args.split}_pseudo_feature_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    top_df = audit_df.head(args.save_top_k)
    for rank, record in enumerate(top_df.to_dict("records"), start=1):
        _save_audit_figure(
            record=record,
            model=model,
            target_layer=target_layer,
            transform=transform,
            device=device,
            output_path=samples_dir / f"rank_{rank:02d}_score_{record['pseudo_attention_score']:.3f}.png",
            args=args,
        )

    high_risk_count = int(audit_df["high_risk"].sum())
    print(f"Saved audit CSV: {audit_csv}")
    print(f"Saved sample figures: {samples_dir}")
    print(f"High-risk samples: {high_risk_count}/{len(audit_df)} (threshold={args.risk_threshold})")


def _save_audit_figure(
    record: dict,
    model: torch.nn.Module,
    target_layer,
    transform,
    device: str,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    original = Image.open(record["path"]).convert("RGB")
    image_tensor = transform(original)
    pred_id = {label: idx for idx, label in ID_TO_LABEL.items()}[record["pred_label"]]
    heatmap = make_gradcam_heatmap(
        model=model,
        image_tensor=image_tensor,
        target_class=pred_id,
        target_layer=target_layer,
        device=device,
    )
    display_image = get_eval_geometry_transforms(
        crop_border=getattr(args, "crop_border", False)
    )(original)
    if display_image.size != (heatmap.shape[1], heatmap.shape[0]):
        raise RuntimeError(
            f"Aligned display image has size {display_image.size}, heatmap is "
            f"{heatmap.shape[1]}x{heatmap.shape[0]}"
        )
    masks = build_pseudo_feature_masks(
        display_image,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        color_spread_threshold=args.color_spread_threshold,
        min_highlight_area=args.min_highlight_area,
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(display_image)
    axes[0].set_title("Original")
    axes[1].imshow(display_image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(masks["combined"], cmap="gray")
    axes[2].set_title("Pseudo-feature mask")
    axes[3].imshow(display_image)
    axes[3].imshow(masks["combined"], cmap="Reds", alpha=0.35)
    axes[3].imshow(heatmap, cmap="jet", alpha=0.35)
    axes[3].set_title("Mask + heatmap")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"true={record['true_label']} pred={record['pred_label']} "
        f"conf={record['confidence']:.3f} pseudo_score={record['pseudo_attention_score']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
