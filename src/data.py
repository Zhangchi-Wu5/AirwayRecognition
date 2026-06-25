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


class CropBlackBorder:
    """Crop the scope's edge-connected black border (mechanical vignette) to the content bbox.

    This removes only non-image content (the black frame outside the optical field),
    NOT anatomical structures or candidate ROIs — so it is compatible with whole-image
    input. Detection runs on a 128x128 thumbnail for speed; the bounding box is mapped
    back to the original resolution before cropping.
    """

    def __init__(self, dark_threshold: int = 24, detect_size: int = 128):
        self.dark_threshold = dark_threshold
        self.detect_size = detect_size

    def __call__(self, image: Image.Image) -> Image.Image:
        import numpy as np
        from src.viz import detect_dark_border_mask

        w, h = image.size
        small = image.resize((self.detect_size, self.detect_size))
        border = detect_dark_border_mask(small, dark_threshold=self.dark_threshold)
        content = ~border
        if not content.any():
            return image
        ys = np.where(content.any(axis=1))[0]
        xs = np.where(content.any(axis=0))[0]
        x0 = int(xs[0] / self.detect_size * w)
        x1 = int((xs[-1] + 1) / self.detect_size * w)
        y0 = int(ys[0] / self.detect_size * h)
        y1 = int((ys[-1] + 1) / self.detect_size * h)
        if (x1 - x0) < 8 or (y1 - y0) < 8:  # skip degenerate / near-empty crops
            return image
        return image.crop((x0, y0, x1, y1))


def get_train_transforms(crop_border: bool = False) -> transforms.Compose:
    """Augmentations for training.

    NOTE: Horizontal flip is intentionally disabled because yz (right) and
    zz (left) labels would swap under flipping.

    When ``crop_border`` is True, the scope black border is removed first (so the
    pseudo-feature suppression loss only needs to handle the un-croppable highlights).
    """
    ops = []
    if crop_border:
        ops.append(CropBlackBorder())
    ops += [
        transforms.Resize(256),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def get_eval_transforms(crop_border: bool = False) -> transforms.Compose:
    """Deterministic preprocessing for validation/test/inference."""
    ops = []
    if crop_border:
        ops.append(CropBlackBorder())
    ops += [
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class BronchoscopyDataset(Dataset):
    """PyTorch Dataset that reads images listed in a manifest DataFrame.

    Expects columns: path, label_id.

    When ``return_pseudo_mask`` is True, each item also returns a (1, mask_size,
    mask_size) pseudo-feature mask aligned with the (augmented) image — used by the
    pseudo-feature suppression loss. Validation/test loaders should keep it False.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform=None,
        return_pseudo_mask: bool = False,
        mask_size: int = 56,
        mask_kind: str = "combined",
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.return_pseudo_mask = return_pseudo_mask
        self.mask_size = mask_size
        self.mask_kind = mask_kind

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_id"])
        if self.return_pseudo_mask:
            from src.regularization import pseudo_mask_from_tensor
            mask = pseudo_mask_from_tensor(
                image, IMAGENET_MEAN, IMAGENET_STD, self.mask_size, kind=self.mask_kind
            )
            return image, label, mask
        return image, label
