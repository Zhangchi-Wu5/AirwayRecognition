"""Smoke tests: verify the whole data pipeline works end-to-end on real data."""
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    BronchoscopyDataset, build_manifest, get_eval_transforms,
    split_by_patient,
)
from src.models import build_resnet50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="Real dataset not available")
class TestRealDataPipeline:
    def test_manifest_has_expected_counts(self):
        df = build_manifest(DATASET_DIR)
        # Allow some slack because we verified 641 rows earlier
        assert 600 <= len(df) <= 700
        assert set(df["label"].unique()) == {"lt", "yz", "zz"}
        assert 200 <= df["patient_id"].nunique() <= 230

    def test_split_produces_three_nonempty_sets_without_overlap(self):
        df = build_manifest(DATASET_DIR)
        train, val, test = split_by_patient(df, seed=42)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        train_p = set(train["patient_id"])
        val_p = set(val["patient_id"])
        test_p = set(test["patient_id"])
        assert not (train_p & val_p)
        assert not (train_p & test_p)
        assert not (val_p & test_p)

    def test_dataloader_produces_correct_batch(self):
        df = build_manifest(DATASET_DIR)
        train, _, _ = split_by_patient(df, seed=42)
        ds = BronchoscopyDataset(train, transform=get_eval_transforms())
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        assert labels.dtype == torch.int64

    def test_model_forward_on_real_batch(self):
        df = build_manifest(DATASET_DIR)
        train, _, _ = split_by_patient(df, seed=42)
        ds = BronchoscopyDataset(train, transform=get_eval_transforms())
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        images, _ = next(iter(loader))
        model = build_resnet50(num_classes=3, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (2, 3)
