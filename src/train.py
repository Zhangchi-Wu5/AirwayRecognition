"""Training loop with two-stage fine-tuning, early stopping, and checkpointing."""
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import freeze_backbone, unfreeze_all
from src.regularization import equivariance_loss, pseudo_suppression_loss


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze the backbone, preferring a model-provided method (attention model)."""
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    else:
        freeze_backbone(model)


def _unfreeze_all(model: nn.Module) -> None:
    if hasattr(model, "unfreeze_all"):
        model.unfreeze_all()
    else:
        unfreeze_all(model)


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
    reg: Optional[dict] = None,
) -> tuple[float, float]:
    """Train for one epoch; return (avg_loss, accuracy).

    If ``reg`` is provided (dict with ``lambda_eq``, ``lambda_pf``, ``max_angle``),
    the model must support ``forward(x, return_attn=True)`` and the loader must yield
    (images, labels, pseudo_mask) triples. Total loss = CE + lambda_pf * pseudo
    suppression + lambda_eq * attention equivariance.
    """
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in loader:
        if reg is not None:
            images, labels, masks = batch
            masks = masks.to(device)
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if reg is not None:
            logits, attn = model(images, return_attn=True)
            loss = criterion(logits, labels)
            if reg.get("lambda_pf", 0.0) > 0:
                loss = loss + reg["lambda_pf"] * pseudo_suppression_loss(attn, masks)
            if reg.get("lambda_eq", 0.0) > 0:
                loss = loss + reg["lambda_eq"] * equivariance_loss(
                    model, images, attn, max_angle=reg.get("max_angle", 15.0)
                )
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> tuple[float, float]:
    """Evaluate on a loader; return (avg_loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


def train_two_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    stage1_epochs: int = 5,
    stage1_lr: float = 1e-3,
    stage2_epochs: int = 20,
    stage2_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    checkpoint_path: Optional[Path] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
    reg: Optional[dict] = None,
) -> dict:
    """Run two-stage fine-tuning.

    Stage 1: Freeze backbone, train head for stage1_epochs.
    Stage 2: Unfreeze all, train for up to stage2_epochs with early stopping.

    ``reg`` (optional) enables attention regularization — see ``train_one_epoch``.

    Returns: history dict with lists of per-epoch metrics.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = {
        "stage": [], "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    # Stage 1: head only
    _freeze_backbone(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=stage1_lr, weight_decay=weight_decay)
    for epoch in range(1, stage1_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, reg=reg)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        _record(history, 1, epoch, train_loss, train_acc, val_loss, val_acc)
        if on_epoch_end:
            on_epoch_end({"stage": 1, "epoch": epoch, "train_loss": train_loss,
                          "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    # Stage 2: full network
    _unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    best_val_acc = -1.0
    patience_counter = 0
    for epoch in range(1, stage2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, reg=reg)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        _record(history, 2, epoch, train_loss, train_acc, val_loss, val_acc)
        if on_epoch_end:
            on_epoch_end({"stage": 2, "epoch": epoch, "train_loss": train_loss,
                          "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at stage 2 epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break
    history["best_val_acc"] = best_val_acc
    return history


def _record(history, stage, epoch, train_loss, train_acc, val_loss, val_acc):
    history["stage"].append(stage)
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
