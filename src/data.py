"""Data loading, manifest generation, and patient-level splitting."""
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_TO_ID = {"lt": 0, "yz": 1, "zz": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABEL_NAMES_CN = {"lt": "隆突", "yz": "右总支气管", "zz": "左总支气管"}

_FILENAME_PATTERN = re.compile(r"^(\d+)\s*(lt|yz|zz)\.(png|jpg)$", re.IGNORECASE)


def parse_filename(filename: str) -> Optional[dict]:
    """Parse a bronchoscopy image filename into components.

    Returns None if the filename doesn't match the expected pattern.
    Pattern: {patient_id}[optional-space]{lt|yz|zz}.{png|jpg}
    """
    match = _FILENAME_PATTERN.match(filename.strip())
    if match is None:
        return None
    patient_id, label, ext = match.groups()
    return {
        "patient_id": patient_id,
        "label": label.lower(),
        "ext": ext.lower(),
    }


def build_manifest(dataset_dir: Path) -> pd.DataFrame:
    """Scan dataset_dir and build a manifest dataframe.

    Columns: patient_id, label, label_id, path
    Skips unparseable files with a warning.
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    skipped = []
    for file_path in sorted(dataset_dir.iterdir()):
        if not file_path.is_file():
            continue
        parsed = parse_filename(file_path.name)
        if parsed is None:
            skipped.append(file_path.name)
            continue
        rows.append({
            "patient_id": parsed["patient_id"],
            "label": parsed["label"],
            "label_id": LABEL_TO_ID[parsed["label"]],
            "path": str(file_path.resolve()),
        })
    if skipped:
        print(f"[build_manifest] Skipped {len(skipped)} unparseable files:")
        for name in skipped[:10]:
            print(f"  - {name}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return pd.DataFrame(rows)


def split_by_patient(
    manifest: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split manifest into train/val/test by patient_id.

    All images from the same patient go to the same split.
    test_ratio is inferred as 1 - train_ratio - val_ratio.

    Returns: (train_df, val_df, test_df)
    """
    assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
    assert 0 < val_ratio < 1, "val_ratio must be in (0, 1)"
    assert train_ratio + val_ratio < 1, "train + val must leave room for test"

    patient_ids = sorted(manifest["patient_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    train_df = manifest[manifest["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = manifest[manifest["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = manifest[manifest["patient_id"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Augmentations for training.

    NOTE: Horizontal flip is intentionally disabled because yz (right) and
    zz (left) labels would swap under flipping.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic preprocessing for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BronchoscopyDataset(Dataset):
    """PyTorch Dataset that reads images listed in a manifest DataFrame.

    Expects columns: path, label_id.
    """

    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_id"])
