"""Train and evaluate the bronchoscopy classifier from the command line."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    BronchoscopyDataset,
    build_manifest,
    get_eval_transforms,
    get_train_transforms,
    split_by_patient,
)
from src.attention import build_attention_resnet50
from src.evaluate import collect_predictions, compute_metrics
from src.models import build_resnet50
from src.train import set_seed, train_two_stage
from src.viz import plot_confusion_matrix, plot_training_curves, setup_chinese_font


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs in each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _split_fingerprint(frame: pd.DataFrame) -> str:
    """Hash split membership without depending on machine-specific absolute paths."""
    stable = frame[["patient_id", "label", "label_id"]].copy()
    stable["patient_id"] = stable["patient_id"].astype(str)
    payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the airway recognition model.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "checkpoints" / "best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Patient-split seed. Defaults to --seed for backward compatibility; set a fixed value for multi-seed training.",
    )
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    # Anatomy-attention model + regularization (no manual annotation required).
    parser.add_argument("--attention", action="store_true",
                        help="Use the anatomy-attention ResNet-50 (in-model spatial attention + fusion).")
    parser.add_argument("--reg", action="store_true",
                        help="Enable attention regularization (implies --attention).")
    parser.add_argument("--lambda-eq", type=float, default=0.1, help="Weight for attention equivariance loss.")
    parser.add_argument("--lambda-pf", type=float, default=0.1, help="Weight for pseudo-feature suppression loss.")
    parser.add_argument("--max-angle", type=float, default=15.0, help="Max rotation (deg) for equivariance loss.")
    parser.add_argument("--mask-size", type=int, default=56, help="Resolution for pseudo-feature masks.")
    parser.add_argument("--crop-border", action="store_true",
                        help="Remove the scope black border in preprocessing (recommend with --pf-mask specular_highlight).")
    parser.add_argument("--pf-mask", choices=["combined", "specular_highlight", "dark_border"], default="combined",
                        help="Which pseudo-feature mask the suppression loss targets.")
    parser.add_argument("--attn-hires", action="store_true",
                        help="Compute the attention map at layer3 (14x14) instead of layer4 (7x7) for a finer map.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    setup_chinese_font(verbose=True)
    set_seed(args.seed)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    manifest = build_manifest(args.dataset_dir)
    train_df, val_df, test_df = split_by_patient(manifest, seed=split_seed)
    manifest.to_csv(args.splits_dir / "manifest.csv", index=False)
    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    val_df.to_csv(args.splits_dir / "val.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)

    print(f"Device: {device}")
    print(f"Training seed: {args.seed}; patient-split seed: {split_seed}")
    print(f"Manifest: {len(manifest)} images, {manifest['patient_id'].nunique()} patients")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(df)} images, {df['patient_id'].nunique()} patients, labels={dict(df['label'].value_counts())}")

    use_attention = args.attention or args.reg
    use_reg = args.reg
    train_ds = BronchoscopyDataset(
        train_df, transform=get_train_transforms(crop_border=args.crop_border),
        return_pseudo_mask=use_reg, mask_size=args.mask_size, mask_kind=args.pf_mask,
    )
    val_ds = BronchoscopyDataset(val_df, transform=get_eval_transforms(crop_border=args.crop_border))
    test_ds = BronchoscopyDataset(test_df, transform=get_eval_transforms(crop_border=args.crop_border))
    pin_memory = device == "cuda"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    if use_attention:
        model = build_attention_resnet50(num_classes=3, pretrained=not args.no_pretrained,
                                         dropout=args.dropout, hires=args.attn_hires)
        print(f"Model: AnatomyAttentionResNet50 (reg={'on' if use_reg else 'off'}, hires={args.attn_hires})")
    else:
        model = build_resnet50(num_classes=3, pretrained=not args.no_pretrained, dropout=args.dropout)
        print("Model: ResNet50 (baseline)")
    model = model.to(device)

    reg = None
    if use_reg:
        reg = {"lambda_eq": args.lambda_eq, "lambda_pf": args.lambda_pf, "max_angle": args.max_angle}
        print(f"Regularization: lambda_eq={args.lambda_eq} lambda_pf={args.lambda_pf} "
              f"max_angle={args.max_angle} pf_mask={args.pf_mask}")
    print(f"Preprocessing: crop_border={args.crop_border}")

    def log_epoch(info: dict) -> None:
        print(
            f"Stage {info['stage']} epoch {info['epoch']:02d} | "
            f"train_loss={info['train_loss']:.4f} train_acc={info['train_acc']:.4f} | "
            f"val_loss={info['val_loss']:.4f} val_acc={info['val_acc']:.4f}"
        )

    history = train_two_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=args.checkpoint_path,
        on_epoch_end=log_epoch,
        reg=reg,
    )
    print(f"Best val accuracy: {history['best_val_acc']:.4f}")
    print(f"Best checkpoint: {args.checkpoint_path}")

    history_df = pd.DataFrame({k: v for k, v in history.items() if isinstance(v, list)})
    history_df.to_csv(args.output_dir / "training_history.csv", index=False)
    plot_training_curves(history, args.output_dir / "training_curves.png")

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device=device)
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names=["lt", "yz", "zz"])
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report"])
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["lt", "yz", "zz"],
        output_path=args.output_dir / "confusion_matrix.png",
        title=f"Test confusion matrix (Accuracy={metrics['accuracy']:.4f})",
    )

    metrics_path = args.output_dir / "test_metrics.txt"
    metrics_path.write_text(
        f"accuracy={metrics['accuracy']:.6f}\n\n{metrics['classification_report']}",
        encoding="utf-8",
    )
    predictions = test_df[["patient_id", "label", "label_id", "path"]].copy()
    predictions.insert(0, "index", np.arange(len(predictions)))
    predictions["pred_label_id"] = y_pred
    predictions["pred_label"] = np.asarray(["lt", "yz", "zz"])[y_pred]
    predictions["prob_lt"] = y_proba[:, 0]
    predictions["prob_yz"] = y_proba[:, 1]
    predictions["prob_zz"] = y_proba[:, 2]
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_device": device,
        "resolved_split_seed": split_seed,
        "git_commit": git_commit,
        "n_images": len(manifest),
        "n_patients": int(manifest["patient_id"].nunique()),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "split_fingerprints": {
            "train": _split_fingerprint(train_df),
            "val": _split_fingerprint(val_df),
            "test": _split_fingerprint(test_df),
        },
        "best_val_accuracy": float(history["best_val_acc"]),
        "test_accuracy": float(metrics["accuracy"]),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved probabilities to: {args.output_dir / 'test_predictions.csv'}")
    print(f"Saved run config to: {args.output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
