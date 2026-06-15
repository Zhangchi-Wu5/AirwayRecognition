"""Train and evaluate the bronchoscopy classifier from the command line."""
import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

import matplotlib
matplotlib.use("Agg")
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
from src.evaluate import collect_predictions, compute_metrics
from src.models import build_resnet50
from src.train import set_seed, train_two_stage
from src.viz import plot_confusion_matrix, plot_training_curves, setup_chinese_font


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the airway recognition model.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "checkpoints" / "best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    setup_chinese_font(verbose=True)
    set_seed(args.seed)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.dataset_dir)
    train_df, val_df, test_df = split_by_patient(manifest, seed=args.seed)
    manifest.to_csv(args.splits_dir / "manifest.csv", index=False)
    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    val_df.to_csv(args.splits_dir / "val.csv", index=False)
    test_df.to_csv(args.splits_dir / "test.csv", index=False)

    print(f"Device: {device}")
    print(f"Manifest: {len(manifest)} images, {manifest['patient_id'].nunique()} patients")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(df)} images, {df['patient_id'].nunique()} patients, labels={dict(df['label'].value_counts())}")

    train_ds = BronchoscopyDataset(train_df, transform=get_train_transforms())
    val_ds = BronchoscopyDataset(val_df, transform=get_eval_transforms())
    test_ds = BronchoscopyDataset(test_df, transform=get_eval_transforms())
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
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

    model = build_resnet50(num_classes=3, pretrained=not args.no_pretrained, dropout=args.dropout).to(device)

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
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
