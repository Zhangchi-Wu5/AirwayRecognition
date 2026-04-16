"""Tests for evaluation: collect_predictions, compute_metrics."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import collect_predictions, compute_metrics


def make_tiny_loader(n_samples=12, batch_size=4):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, 3, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_tiny_model():
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 3),
    )


def test_collect_predictions_returns_correct_shape():
    model = make_tiny_model()
    loader = make_tiny_loader(n_samples=12, batch_size=4)
    y_true, y_pred, y_proba = collect_predictions(model, loader, device="cpu")
    assert len(y_true) == 12
    assert len(y_pred) == 12
    assert y_proba.shape == (12, 3)
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])  # 2 wrong
    y_proba = np.eye(3)[y_pred]
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names=["lt", "yz", "zz"])
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert abs(metrics["accuracy"] - 4 / 6) < 1e-6
    cm = metrics["confusion_matrix"]
    assert cm.shape == (3, 3)
    assert cm.sum() == 6
