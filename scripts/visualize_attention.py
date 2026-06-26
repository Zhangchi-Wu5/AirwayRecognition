"""Visualize the in-model anatomical attention map A (the SpatialAttention output).

Unlike Grad-CAM (a post-hoc, class-discriminative explanation), A is the attention map
the model computes internally and uses to weight features for classification — i.e. the
"内嵌解剖注意力" described in the disclosure. This script loads a trained attention
checkpoint, extracts A by a single forward pass (return_attn=True), and renders it as a
heatmap overlay for Fig.1.

Run on the server where the checkpoint and dataset live, e.g.:

  python -m scripts.visualize_attention --checkpoint-path checkpoints_reg_crop/best_model.pt \
      --attention --crop-border --num 6
"""
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
import torch.nn.functional as F
from PIL import Image

from src.attention import build_attention_resnet50
from src.data import ID_TO_LABEL, IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from src.viz import setup_chinese_font


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize the in-model anatomical attention map.")
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "attention_vis")
    p.add_argument("--attention", action="store_true",
                   help="Required: the checkpoint must be an anatomy-attention model.")
    p.add_argument("--crop-border", action="store_true", help="Match crop-border preprocessing used in training.")
    p.add_argument("--num", type=int, default=6, help="Number of images to visualize (first N of the split).")
    p.add_argument("--indices", default=None, help="Comma-separated row indices to visualize (overrides --num).")
    p.add_argument("--topk-percent", type=float, default=None,
                   help="If set (e.g. 25), the overlay highlights only the top X%% strongest attention "
                        "pixels (clearer, more honest than the min-max red wash).")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    setup_chinese_font(verbose=False)

    if not args.attention:
        print("Note: --attention not set; attention visualization requires the attention model. Proceeding as attention model.")
    model = build_attention_resnet50(num_classes=3, pretrained=False, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    transform = get_eval_transforms(crop_border=args.crop_border)
    df = pd.read_csv(args.splits_dir / f"{args.split}.csv").reset_index(drop=True)
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        indices = list(range(min(args.num, len(df))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        row = df.iloc[idx]
        original = Image.open(row["path"]).convert("RGB")
        tensor = transform(original)  # (3,224,224), normalized

        with torch.no_grad():
            logits, attn = model(tensor.unsqueeze(0).to(device), return_attn=True)
            proba = torch.softmax(logits, dim=1)[0].cpu()
            pred_id = int(proba.argmax())
            conf = float(proba[pred_id])

        attn_map = attn[0, 0].cpu()  # (7,7)
        attn_up = F.interpolate(attn_map[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        a = attn_up.numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)  # min-max for display contrast
        base = (tensor * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

        if args.topk_percent:
            import numpy as np
            thr = np.percentile(a, 100 - args.topk_percent)
            overlay_alpha = np.where(a >= thr, 0.55, 0.0)
        else:
            overlay_alpha = 0.45

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(base)
        axes[0].set_title("输入图像")
        axes[1].imshow(a, cmap="jet")
        axes[1].set_title("内嵌解剖注意力 A")
        axes[2].imshow(base)
        axes[2].imshow(a, cmap="jet", alpha=overlay_alpha)
        axes[2].set_title("叠加（前%g%%）" % args.topk_percent if args.topk_percent else "叠加")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"true={row['label']}  pred={ID_TO_LABEL[pred_id]}  conf={conf:.3f}", fontsize=11)
        plt.tight_layout()
        out_path = args.output_dir / f"attn_{idx:03d}_{row['label']}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[{idx}] pred={ID_TO_LABEL[pred_id]} conf={conf:.3f} -> {out_path}")

    print(f"Saved {len(indices)} attention visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
