"""Tests for visualization helpers."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path

from src.viz import plot_training_curves, plot_confusion_matrix


def test_plot_training_curves_saves_png(tmp_path):
    history = {
        "epoch": [1, 2, 3],
        "stage": [1, 1, 1],
        "train_loss": [1.0, 0.5, 0.3],
        "train_acc": [0.4, 0.7, 0.9],
        "val_loss": [1.1, 0.6, 0.4],
        "val_acc": [0.3, 0.6, 0.85],
    }
    output = tmp_path / "curves.png"
    plot_training_curves(history, output)
    assert output.exists()
    assert output.stat().st_size > 1000


def test_plot_confusion_matrix_saves_png(tmp_path):
    cm = np.array([[10, 1, 0], [2, 8, 1], [0, 1, 9]])
    output = tmp_path / "cm.png"
    plot_confusion_matrix(cm, class_names=["lt", "yz", "zz"], output_path=output)
    assert output.exists()
    assert output.stat().st_size > 1000
