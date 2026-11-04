"""Isolated data and input geometry for the new 93-image patient cohort.

Auditing and folds use only Pillow and the standard library. Existing dataset
loaders, old split files, and the original transforms are never modified here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import re


EXPECTED_FINGERPRINT = "7473d977642e4fe9cc8fe3302a0cd5bcd7d073e21ac8b7e30bc4808e95d4caa0"
EXPECTED_IMAGES = 93
EXPECTED_PATIENTS = 31
LABEL_TO_ID = {"lt": 0, "yz": 1, "zz": 2}
METHODS = ("original", "full_frame")
_FILENAME = re.compile(r"^([0-9]+)\s*(lt|yz|zz)\.png$", re.IGNORECASE)


def audit_dataset(root, strict=True):
    """Return image rows and provenance; scan only first-level PNG files.

    ``strict=False`` is solely for small synthetic test fixtures. Formal callers
    must use the default, which locks both cohort size and all source bytes.
    Patient keys normalize leading zeroes for grouping; original IDs are kept.
    Neither file names nor these checks establish clinical label correctness.
    """
    from PIL import Image, ImageFile

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    entries = sorted(root.iterdir(), key=lambda p: p.name)
    files = [p for p in entries if p.is_file() and p.suffix.lower() == ".png"]
    if not files:
        raise ValueError("No first-level PNG images found")
    rows, seen_pairs, seen_hashes, original_ids = [], set(), {}, {}
    previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        for index, path in enumerate(files):
            if path.is_symlink():
                raise ValueError(f"Image symlinks are not accepted: {path.name}")
            match = _FILENAME.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Invalid PNG image name: {path.name}")
            patient_id, label = match.group(1), match.group(2).lower()
            patient_key = str(int(patient_id))
            pair = (patient_key, label)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate normalized patient/class: {patient_key}/{label}")
            seen_pairs.add(pair)
            if original_ids.setdefault(patient_key, patient_id) != patient_id:
                raise ValueError(f"Inconsistent patient ID spelling for normalized key: {patient_key}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError(f"Duplicate image bytes: {seen_hashes[digest]} and {path.name}")
            seen_hashes[digest] = path.name
            try:
                with Image.open(path) as image:
                    if image.format != "PNG":
                        raise ValueError("PNG extension does not contain PNG data")
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
            except Exception as error:
                raise ValueError(f"Image decode failed: {path.name}: {error}") from error
            rows.append({"source_index": index, "patient_id": patient_id,
                         "patient_key": patient_key, "label": label,
                         "label_id": LABEL_TO_ID[label], "path": str(path),
                         "image_name": path.name, "image_sha256": digest,
                         "width": width, "height": height, "mode": mode})
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated

    per_patient = defaultdict(set)
    for row in rows:
        per_patient[row["patient_key"]].add(row["label"])
    if any(labels != set(LABEL_TO_ID) for labels in per_patient.values()):
        raise ValueError("Each patient must have exactly one image of each lt/yz/zz class")
    payload = "".join(row["image_name"] + "\t" + row["image_sha256"] + "\n" for row in rows)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if strict:
        if len(rows) != EXPECTED_IMAGES or len(per_patient) != EXPECTED_PATIENTS:
            raise ValueError("Formal renew cohort requires exactly 93 images and 31 patients")
        if fingerprint != EXPECTED_FINGERPRINT:
            raise ValueError("Formal renew source fingerprint does not match the audited archive")
    metadata = {
        "image_root": str(root), "image_count": len(rows), "patient_count": len(per_patient),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
        "dataset_fingerprint": fingerprint, "strict_cohort": bool(strict),
        "grouping": "numeric patient ID with leading zeroes removed; original ID preserved",
        "scan": "immediate PNG files only; no recursion",
        "ignored_directories": sum(p.is_dir() for p in entries),
        "ignored_non_png_files": sum(p.is_file() and p.suffix.lower() != ".png" for p in entries),
        "clinical_labels_verified": False,
    }
    return rows, metadata


def make_patient_folds(rows, n_folds=5, seed=42):
    """Fixed shuffled patient folds, with each patient evaluated exactly once."""
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("At least two integer folds are required")
    if not rows:
        raise ValueError("Cannot split an empty cohort")
    by_patient = defaultdict(list)
    seen_names = set()
    for row in rows:
        key = str(int(row["patient_id"]))
        if row.get("patient_key") != key:
            raise ValueError("Patient grouping key does not match the original ID")
        if row["image_name"] in seen_names:
            raise ValueError("Duplicate image identity in fold input")
        seen_names.add(row["image_name"])
        by_patient[key].append(row)
    if any(Counter(r["label"] for r in group) != Counter({label: 1 for label in LABEL_TO_ID})
           for group in by_patient.values()):
        raise ValueError("Fold input must have one image per patient/class")
    patients = sorted(by_patient, key=int)
    if len(patients) < n_folds:
        raise ValueError("There are fewer patients than folds")
    random.Random(seed).shuffle(patients)
    quotient, remainder = divmod(len(patients), n_folds)
    folds, offset = [], 0
    for fold in range(n_folds):
        size = quotient + (fold < remainder)
        eval_keys = patients[offset:offset + size]
        eval_set = set(eval_keys)
        train_keys = [key for key in patients if key not in eval_set]
        folds.append({"fold": fold,
                      "train": [row for row in rows if row["patient_key"] not in eval_set],
                      "eval": [row for row in rows if row["patient_key"] in eval_set],
                      "train_patient_keys": train_keys, "eval_patient_keys": eval_keys})
        offset += size
    return folds


@dataclass(frozen=True)
class Letterbox224:
    """Resize the entire RGB rectangle to fit and center it on black padding."""
    size: int = 224

    def __post_init__(self):
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 1:
            raise ValueError("Letterbox size must be a positive integer")

    def __call__(self, image):
        from PIL import Image
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < 1:
            raise ValueError("Cannot letterbox an empty image")
        scale = min(self.size / width, self.size / height)
        resized_size = (max(1, min(self.size, round(width * scale))),
                        max(1, min(self.size, round(height * scale))))
        resized = image.resize(resized_size, resample=Image.Resampling.BILINEAR)
        result = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        result.paste(resized, ((self.size - resized.width) // 2,
                              (self.size - resized.height) // 2))
        return result


@dataclass(frozen=True)
class _FixedRotation:
    """Deterministic geometry preview only; never used by the training factory."""
    angle: float
    expand: bool

    def __call__(self, image):
        from torchvision.transforms import functional as TF
        return TF.rotate(image, self.angle, expand=self.expand, fill=0)


def get_geometry_transforms(method, train=False, angle=None):
    """PIL geometry for real inputs and previews, without color/normalization.

    ``angle`` may replace only the rotation in a training-geometry preview;
    the original method still samples its original random crop. Evaluation is
    deterministic and disallows a supplied angle. No ROI detector is used.
    """
    from torchvision import transforms
    from src.data import get_eval_geometry_transforms, get_train_transforms
    if method not in METHODS:
        raise ValueError(f"Unknown preprocessing method: {method}")
    if angle is not None and (not train or not math.isfinite(float(angle))):
        raise ValueError("A finite preview angle is allowed only with train=True")
    if method == "original":
        if not train:
            return get_eval_geometry_transforms(False)
        operations = list(get_train_transforms(False).transforms[:3])
        if angle is not None:
            operations[-1] = _FixedRotation(float(angle), expand=False)
        return transforms.Compose(operations)
    if not train:
        return transforms.Compose([Letterbox224()])
    rotation = (transforms.RandomRotation(degrees=15, expand=True, fill=0) if angle is None
                else _FixedRotation(float(angle), expand=True))
    return transforms.Compose([transforms.Resize(256), rotation, Letterbox224()])


def get_transforms(method, train=False):
    """Original pipeline, or full-frame geometry with the same color settings."""
    from torchvision import transforms
    from src.data import (IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms,
                          get_train_transforms)
    if method not in METHODS:
        raise ValueError(f"Unknown preprocessing method: {method}")
    if method == "original":
        return get_train_transforms(False) if train else get_eval_transforms(False)
    operations = list(get_geometry_transforms(method, train=train).transforms)
    if train:
        operations.append(transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2))
    operations.extend([transforms.ToTensor(),
                       transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(operations)
