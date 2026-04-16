"""Tests for patient-level data splitting."""
import pandas as pd
import pytest
from src.data import split_by_patient


def make_fake_manifest(n_patients: int) -> pd.DataFrame:
    """Create a fake manifest with n_patients, each having 3 images (lt/yz/zz)."""
    rows = []
    for i in range(n_patients):
        pid = f"P{i:06d}"
        for label, lid in [("lt", 0), ("yz", 1), ("zz", 2)]:
            rows.append({
                "patient_id": pid,
                "label": label,
                "label_id": lid,
                "path": f"/fake/{pid}{label}.png",
            })
    return pd.DataFrame(rows)


class TestSplitByPatient:
    def test_split_sizes_sum_to_total_patients(self):
        manifest = make_fake_manifest(100)
        train, val, test = split_by_patient(manifest, seed=42)
        total_patients = (
            train["patient_id"].nunique()
            + val["patient_id"].nunique()
            + test["patient_id"].nunique()
        )
        assert total_patients == 100

    def test_no_patient_overlap(self):
        manifest = make_fake_manifest(100)
        train, val, test = split_by_patient(manifest, seed=42)
        train_p = set(train["patient_id"])
        val_p = set(val["patient_id"])
        test_p = set(test["patient_id"])
        assert train_p & val_p == set()
        assert train_p & test_p == set()
        assert val_p & test_p == set()

    def test_split_ratio_approximately_70_15_15(self):
        manifest = make_fake_manifest(200)
        train, val, test = split_by_patient(manifest, seed=42)
        n_train = train["patient_id"].nunique()
        n_val = val["patient_id"].nunique()
        n_test = test["patient_id"].nunique()
        assert 130 <= n_train <= 150  # ~70% of 200
        assert 25 <= n_val <= 35      # ~15% of 200
        assert 25 <= n_test <= 35     # ~15% of 200

    def test_reproducibility_with_same_seed(self):
        manifest = make_fake_manifest(50)
        train1, val1, test1 = split_by_patient(manifest, seed=42)
        train2, val2, test2 = split_by_patient(manifest, seed=42)
        assert set(train1["patient_id"]) == set(train2["patient_id"])
        assert set(val1["patient_id"]) == set(val2["patient_id"])
        assert set(test1["patient_id"]) == set(test2["patient_id"])

    def test_all_images_for_patient_stay_together(self):
        manifest = make_fake_manifest(50)
        train, val, test = split_by_patient(manifest, seed=42)
        # For each patient, all 3 images must be in the same split
        for split_df in [train, val, test]:
            counts = split_df.groupby("patient_id").size()
            assert (counts == 3).all(), "Every patient in a split must have all 3 images"
