"""Tests for visualization helpers."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path

from src.viz import (
    build_pseudo_feature_masks,
    detect_dark_border_mask,
    detect_specular_highlight_mask,
    plot_training_curves,
    plot_confusion_matrix,
    pseudo_feature_attention_score,
)


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


def test_detect_dark_border_mask_marks_only_edge_connected_dark_pixels():
    image = np.full((20, 20, 3), 180, dtype=np.uint8)
    image[:2, :, :] = 0
    image[-2:, :, :] = 0
    image[:, :2, :] = 0
    image[:, -2:, :] = 0
    image[9:11, 9:11, :] = 0  # dark airway-like center, not edge-connected

    mask = detect_dark_border_mask(image, dark_threshold=10)

    assert mask.dtype == bool
    assert mask[0, 0]
    assert mask[1, 10]
    assert not mask[10, 10]
    assert not mask[5, 5]


def test_detect_specular_highlight_mask_marks_near_white_not_red_tissue():
    image = np.full((12, 12, 3), [150, 45, 45], dtype=np.uint8)
    image[4:7, 4:7, :] = [252, 250, 248]
    image[8:10, 8:10, :] = [245, 20, 20]

    mask = detect_specular_highlight_mask(
        image,
        bright_threshold=235,
        color_spread_threshold=20,
        min_area=2,
    )

    assert mask[5, 5]
    assert not mask[9, 9]
    assert not mask[0, 0]


def test_build_pseudo_feature_masks_returns_union():
    image = np.full((12, 12, 3), 120, dtype=np.uint8)
    image[:, :1, :] = 0
    image[5:7, 5:7, :] = 255

    masks = build_pseudo_feature_masks(
        image,
        dark_threshold=10,
        bright_threshold=240,
        min_highlight_area=2,
    )

    assert set(masks) == {"dark_border", "specular_highlight", "combined"}
    assert masks["dark_border"][4, 0]
    assert masks["specular_highlight"][5, 5]
    assert masks["combined"][4, 0]
    assert masks["combined"][5, 5]


def test_pseudo_feature_attention_score_measures_heatmap_overlap():
    heatmap = np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    mask = np.array([[False, True], [True, False]])

    score = pseudo_feature_attention_score(heatmap, mask)

    assert score == pytest.approx(0.75)
