"""Evaluation utilities: collect predictions and compute metrics."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on loader; return (y_true, y_pred, y_proba) as numpy arrays."""
    model = model.to(device)
    model.eval()
    all_true, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            proba = torch.softmax(logits, dim=1)
            preds = proba.argmax(dim=1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_proba.append(proba.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute accuracy, confusion matrix, and classification report."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
