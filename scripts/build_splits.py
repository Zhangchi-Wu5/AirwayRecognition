"""Generate manifest.csv and train/val/test splits from the real dataset."""
from pathlib import Path
from src.data import build_manifest, split_by_patient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_DIR = PROJECT_ROOT / "data_splits"
SPLITS_DIR.mkdir(exist_ok=True)

# 1. Manifest
manifest = build_manifest(DATASET_DIR)
manifest_path = SPLITS_DIR / "manifest.csv"
manifest.to_csv(manifest_path, index=False)
print(f"Manifest saved: {manifest_path} ({len(manifest)} rows, {manifest['patient_id'].nunique()} patients)")

# 2. Splits
train_df, val_df, test_df = split_by_patient(manifest, seed=42)
train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

print(f"Train: {train_df['patient_id'].nunique()} patients, {len(train_df)} images")
print(f"Val:   {val_df['patient_id'].nunique()} patients, {len(val_df)} images")
print(f"Test:  {test_df['patient_id'].nunique()} patients, {len(test_df)} images")

# 3. Sanity check: no patient overlap
train_p = set(train_df["patient_id"])
val_p = set(val_df["patient_id"])
test_p = set(test_df["patient_id"])
assert not (train_p & val_p), "Patient leakage between train and val!"
assert not (train_p & test_p), "Patient leakage between train and test!"
assert not (val_p & test_p), "Patient leakage between val and test!"
print("Sanity check passed: no patient overlap.")
