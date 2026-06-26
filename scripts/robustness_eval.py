"""Robustness evaluation under scope rotation and synthetic pseudo-features.

Equivariance and pseudo-feature suppression are not meant to raise accuracy on clean,
in-distribution images (where a plain classifier already does well); their benefit shows
up when the input is rotated or contaminated by glare / occlusion. This script applies
controlled perturbations of increasing strength to the test split and reports accuracy
vs. strength, so that benefit becomes measurable.

Run once per checkpoint (matching the flags it was trained with) and compare the curves,
e.g.:

  python -m scripts.robustness_eval --checkpoint-path checkpoints_baseline/best_model.pt --label baseline
  python -m scripts.robustness_eval --checkpoint-path checkpoints_reg_crop/best_model.pt \
      --attention --crop-border --label full
"""
import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

import pandas as pd
import torch
from PIL import Image

from src.attention import build_attention_resnet50
from src.data import IMAGENET_MEAN, IMAGENET_STD, ID_TO_LABEL, get_eval_transforms
from src.models import build_resnet50
from src.regularization import _rotate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

ROTATIONS = [0, 10, 20, 30, 45]          # degrees
GLARE_COVERAGE = [0.0, 0.05, 0.10, 0.20]  # fraction of area as near-white patches
OCCLUSION_COVERAGE = [0.0, 0.05, 0.10, 0.20]  # fraction of area blacked out


def _denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.cpu() * _STD + _MEAN).clamp(0, 1)


def _renorm(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - _MEAN) / _STD


def apply_rotation(x_norm: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a normalized batch (B,3,H,W) by angle_deg; corners filled with black."""
    if angle_deg == 0:
        return x_norm
    x01 = _denorm(x_norm)
    rotated = _rotate(x01, torch.tensor(float(angle_deg)))
    return _renorm(rotated)


def apply_patches(x_norm: torch.Tensor, coverage: float, value: float, seed: int, patch: int = 20) -> torch.Tensor:
    """Stamp square patches (value in [0,1]) covering ~coverage of the area, deterministically.

    value≈0.99 simulates specular glare; value=0.0 simulates black occlusion.
    """
    if coverage <= 0:
        return x_norm
    x01 = _denorm(x_norm).clone()
    b, c, h, w = x01.shape
    n_patches = max(1, int(round(coverage * h * w / (patch * patch))))
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_patches):
        ys = torch.randint(0, max(1, h - patch), (b,), generator=g)
        xs = torch.randint(0, max(1, w - patch), (b,), generator=g)
        for i in range(b):
            x01[i, :, ys[i]:ys[i] + patch, xs[i]:xs[i] + patch] = value
    return _renorm(x01)


def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: str, batch_size: int = 32) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            xb = X[i:i + batch_size].to(device)
            logits = model(xb)
            correct += (logits.argmax(dim=1).cpu() == y[i:i + batch_size]).sum().item()
    return correct / X.size(0)


def _mean_std(vals: list) -> tuple:
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var ** 0.5


def run_robustness(
    model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, device: str,
    seed: int = 42, n_seeds: int = 1,
) -> list:
    """Return [{perturbation, level, accuracy, std}] over rotation / glare / occlusion.

    Rotation is deterministic (std=0). Glare and occlusion depend on random patch
    placement, so they are averaged over ``n_seeds`` placements and report the std.
    """
    n_seeds = max(1, n_seeds)
    results = []
    for a in ROTATIONS:
        acc = evaluate(model, apply_rotation(X, a), y, device)
        results.append({"perturbation": "rotation", "level": a, "accuracy": acc, "std": 0.0})
    for pert, value, base in [("glare", 0.99, seed), ("occlusion", 0.0, seed + 1000)]:
        cov_list = GLARE_COVERAGE if pert == "glare" else OCCLUSION_COVERAGE
        for cov in cov_list:
            accs = [
                evaluate(model, apply_patches(X, cov, value=value, seed=base + s), y, device)
                for s in range(n_seeds)
            ]
            mean, std = _mean_std(accs)
            results.append({"perturbation": pert, "level": cov, "accuracy": mean, "std": std})
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness evaluation under rotation / glare / occlusion.")
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--attention", action="store_true", help="Load the anatomy-attention model.")
    p.add_argument("--attn-hires", action="store_true",
                   help="Build the attention model in hires (layer3, 14x14) mode — match the checkpoint.")
    p.add_argument("--crop-border", action="store_true", help="Match crop-border preprocessing used in training.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42, help="Base seed for glare/occlusion patch placement.")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of random seeds to average for glare/occlusion (rotation is deterministic).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--label", default="model", help="Name for this checkpoint in the printout.")
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.attention:
        model = build_attention_resnet50(num_classes=3, pretrained=False, dropout=args.dropout,
                                         hires=args.attn_hires).to(device)
    else:
        model = build_resnet50(num_classes=3, pretrained=False, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    transform = get_eval_transforms(crop_border=args.crop_border)
    split_df = pd.read_csv(args.splits_dir / f"{args.split}.csv")
    xs, ys = [], []
    for _, row in split_df.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        xs.append(transform(img))
        ys.append(int(row["label_id"]))
    X = torch.stack(xs)
    y = torch.tensor(ys)

    results = run_robustness(model, X, y, device, seed=args.seed, n_seeds=args.seeds)
    df = pd.DataFrame(results)
    df.insert(0, "label", args.label)

    print(f"\n=== Robustness [{args.label}] on {args.split} "
          f"({X.size(0)} images, device={device}, seeds={args.seeds}) ===")
    for pert in ["rotation", "glare", "occlusion"]:
        sub = df[df["perturbation"] == pert]
        cells = "  ".join(
            f"{lvl}:{acc:.3f}" + (f"±{sd:.3f}" if sd > 0 else "")
            for lvl, acc, sd in zip(sub["level"], sub["accuracy"], sub["std"])
        )
        print(f"{pert:10s} {cells}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
