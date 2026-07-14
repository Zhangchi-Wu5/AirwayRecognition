"""Tests for BronchoscopyDataset and transforms."""
import pandas as pd
import numpy as np
import pytest
import torch
from PIL import Image
from src.data import (
    BronchoscopyDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_train_transforms,
    get_eval_transforms,
    get_eval_geometry_transforms,
)


@pytest.fixture
def fake_dataset_dir(tmp_path):
    """Create a tiny fake dataset with 3 images of different colors."""
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        img = Image.new("RGB", (320, 240), color=color)
        img.save(tmp_path / f"000000000{i}lt.png")
    return tmp_path


@pytest.fixture
def fake_manifest_df(fake_dataset_dir):
    """Manifest for the 3-image fake dataset."""
    rows = []
    for i in range(3):
        rows.append({
            "patient_id": f"000000000{i}",
            "label": "lt",
            "label_id": 0,
            "path": str(fake_dataset_dir / f"000000000{i}lt.png"),
        })
    return pd.DataFrame(rows)


class TestTransforms:
    def test_train_transforms_output_shape(self, fake_dataset_dir):
        transform = get_train_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out = transform(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_eval_transforms_output_shape(self, fake_dataset_dir):
        transform = get_eval_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_train_transforms_are_random(self, fake_dataset_dir):
        # Two runs should produce different tensors because of augmentation
        transform = get_train_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out1 = transform(img)
        out2 = transform(img)
        assert not torch.allclose(out1, out2), "Train transforms must be stochastic"

    def test_eval_transforms_are_deterministic(self, fake_dataset_dir):
        transform = get_eval_transforms()
        img = Image.open(fake_dataset_dir / "0000000000lt.png")
        out1 = transform(img)
        out2 = transform(img)
        assert torch.allclose(out1, out2), "Eval transforms must be deterministic"

    @pytest.mark.parametrize("crop_border", [False, True])
    def test_eval_display_geometry_matches_model_tensor(self, crop_border):
        image = np.full((180, 320, 3), [140, 70, 60], dtype=np.uint8)
        image[:, :30] = 0
        image[:, -30:] = 0
        pil = Image.fromarray(image)

        display = get_eval_geometry_transforms(crop_border=crop_border)(pil)
        tensor = get_eval_transforms(crop_border=crop_border)(pil)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        denormalized = (tensor * std + mean).permute(1, 2, 0).numpy()

        assert display.size == (224, 224)
        assert np.allclose(denormalized, np.asarray(display) / 255.0, atol=1e-6)


class TestBronchoscopyDataset:
    def test_len(self, fake_manifest_df):
        ds = BronchoscopyDataset(fake_manifest_df, transform=get_eval_transforms())
        assert len(ds) == 3

    def test_getitem_returns_tensor_and_label(self, fake_manifest_df):
        ds = BronchoscopyDataset(fake_manifest_df, transform=get_eval_transforms())
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert label == 0

    def test_label_returns_label_id_not_string(self, fake_manifest_df):
        ds = BronchoscopyDataset(fake_manifest_df, transform=get_eval_transforms())
        _, label = ds[0]
        assert label == 0  # lt → 0
        assert not isinstance(label, str)
