"""Frozen per-image disparity targets and training-only relative-order supervision.

This module does not infer physical depth or fit statistics across images. All
cache reads are restricted to the rows explicitly supplied by the experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import ResNet, BasicBlock

from src.data import IMAGENET_MEAN, IMAGENET_STD

VERSION = "relative_disparity_v1"
VARIANTS = (("original", 1.0), ("brightness", .9), ("brightness", 1.1),
            ("saturation", .9), ("saturation", 1.1))
GRID_SIZE = 14
PAIR_COUNT = 1024
PAIR_SEED = 20261105
MIN_RELATIVE_GAP = .02


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cache_manifest(cache_root):
    root = Path(cache_root)
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if value.get("version") != VERSION or digest(value.get("teacher")) != value.get("teacher_sha256"):
        raise ValueError("Geometry cache teacher/version identity mismatch")
    if value.get("variants") != [list(v) for v in VARIANTS]:
        raise ValueError("Geometry cache photometric variant mismatch")
    return value


def _read_entry(root, row, manifest):
    key = row["image_sha256"]
    entry = manifest["entries"].get(key)
    if not entry or entry.get("image_name") != row["image_name"]:
        raise ValueError(f"Geometry cache missing/wrong image: {row['image_name']}")
    path = Path(root) / f"{key}.npz"
    if entry.get("file") != path.name or file_hash(path) != entry.get("sha256"):
        raise ValueError(f"Geometry cache checksum mismatch: {row['image_name']}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"disparity", "original_size"}:
            raise ValueError("Unexpected geometry cache fields")
        maps = data["disparity"].copy()
        size = tuple(int(v) for v in data["original_size"])
    if (maps.dtype != np.float32 or maps.ndim != 3 or maps.shape[0] != len(VARIANTS)
            or min(maps.shape[1:]) < GRID_SIZE or not np.isfinite(maps).all()
            or size != tuple(entry["original_size"]) or list(maps.shape) != entry["shape"]):
        raise ValueError("Geometry cache shape/finite-value identity mismatch")
    return maps, size, entry


def validate_cache(cache_root, rows):
    """Fingerprint only selected support entries; later cache growth is harmless."""
    manifest = load_cache_manifest(cache_root)
    selected = {}
    for row in rows:
        _, _, entry = _read_entry(cache_root, row, manifest)
        selected[row["image_sha256"]] = entry
    identity = {"version": VERSION, "teacher_sha256": manifest["teacher_sha256"],
                "variants": manifest["variants"], "entries": selected}
    return {"identity_sha256": digest(identity), **identity}


def cache_predictions(rows, cache_root, predict, teacher_identity):
    """predict(PIL RGB)->2-D disparity; safe resumption without fitted statistics."""
    from src.baseline_reproduction import output_lock
    with output_lock(Path(cache_root)):
        return _cache_predictions(rows, cache_root, predict, teacher_identity)


def _cache_predictions(rows, cache_root, predict, teacher_identity):
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    teacher_hash = digest(teacher_identity)
    if (root / "manifest.json").exists():
        manifest = load_cache_manifest(root)
        if manifest["teacher_sha256"] != teacher_hash:
            raise ValueError("Existing cache belongs to a different teacher/config")
    else:
        if any(p.name != ".reproduction.lock" for p in root.iterdir()):
            raise ValueError("Nonempty cache without manifest; use a new cache directory")
        manifest = {"version": VERSION, "teacher": teacher_identity,
                    "teacher_sha256": teacher_hash, "variants": [list(v) for v in VARIANTS],
                    "entries": {}}
        atomic_json(root / "manifest.json", manifest)
    for row in rows:
        key = row["image_sha256"]
        if file_hash(row["path"]) != key:
            raise ValueError(f"Source image checksum mismatch: {row['image_name']}")
        if key in manifest["entries"]:
            _read_entry(root, row, manifest)
            continue
        path = root / f"{key}.npz"
        if path.exists():
            raise ValueError(f"Unrecorded cache file exists: {path.name}")
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB")
        canvas = TF.resize(image, 256)
        maps = []
        for kind, factor in VARIANTS:
            varied = image if kind == "original" else getattr(TF, f"adjust_{kind}")(image, factor)
            values = np.asarray(predict(varied), dtype=np.float32)
            if values.ndim != 2 or not np.isfinite(values).all() or np.ptp(values) <= 1e-8:
                raise ValueError(f"Teacher disparity is nonfinite/constant: {row['image_name']}")
            maps.append(np.asarray(Image.fromarray(values, mode="F").resize(
                canvas.size, Image.Resampling.BILINEAR), dtype=np.float32))
        values = np.stack(maps)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f, disparity=values, original_size=np.array(image.size, dtype=np.int32))
        temporary.replace(path)
        manifest["entries"][key] = {"image_name": row["image_name"], "file": path.name,
            "sha256": file_hash(path), "shape": list(values.shape), "original_size": list(image.size),
            "range": [float(values.min()), float(values.max())]}
        atomic_json(root / "manifest.json", manifest)
    return validate_cache(root, rows)


def joint_geometry(image, disparity, train=True, angle=None):
    """The RGB operations match renew full_frame; all maps share its coordinates."""
    image = image.convert("RGB")
    if train:
        image = TF.resize(image, 256)
    maps = [Image.fromarray(x.astype(np.float32), mode="F").resize(image.size,
            Image.Resampling.BILINEAR) for x in disparity]
    valid = Image.new("F", image.size, 1.0)
    if train:
        rotation = (float(torch.empty(1).uniform_(-15, 15).item()) if angle is None else float(angle))
        image = TF.rotate(image, rotation, expand=True, fill=0)
        maps = [TF.rotate(x, rotation, expand=True, fill=0) for x in maps]
        valid = TF.rotate(valid, rotation, expand=True, fill=0)
    width, height = image.size
    scale = min(224 / width, 224 / height)
    size = (max(1, min(224, round(width * scale))), max(1, min(224, round(height * scale))))
    origin = ((224 - size[0]) // 2, (224 - size[1]) // 2)
    def letterbox(obj, mode):
        canvas = Image.new(mode, (224, 224), 0)
        canvas.paste(obj.resize(size, Image.Resampling.BILINEAR), origin)
        return canvas
    image = letterbox(image, "RGB")
    full_maps = np.stack([np.asarray(letterbox(m, "F"), dtype=np.float32) for m in maps])
    full_valid = np.asarray(letterbox(valid, "F"), dtype=np.float32).copy()
    # Average pooling rejects any cell intersecting padding/rotation fill.
    mask = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(full_valid)[None], GRID_SIZE)[0] > .999
    targets = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(full_maps), GRID_SIZE)
    return image, targets, mask


class GeometryImageDataset:
    def __init__(self, rows, cache_root=None, train=True, seed=42):
        self.rows, self.train, self.seed = list(rows), train, seed
        self.maps = {}
        if cache_root is not None:
            manifest = load_cache_manifest(cache_root)
            for row in self.rows:
                maps, size, _ = _read_entry(cache_root, row, manifest)
                self.maps[row["image_sha256"]] = (maps, size)
        self.color = transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB")
        cached = self.maps.get(row["image_sha256"])
        if cached is not None and cached[1] != image.size:
            raise ValueError("Geometry and source-image dimensions differ")
        maps = cached[0] if cached else np.zeros((len(VARIANTS), 16, 16), dtype=np.float32)
        image, maps, valid = joint_geometry(image, maps, train=self.train)
        if cached is None:
            valid.zero_()
        if self.train:
            image = self.color(image)
        tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        key = int(row["image_sha256"][:12], 16) ^ self.seed
        return {"image": tensor, "label": int(row["label_id"]), "geometry": maps,
                "valid": valid, "sample_seed": key}


class GeometryResNet18(ResNet):
    def __init__(self, num_classes=3, dropout=.3, pretrained=True):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
        # Auxiliary-head initialization must not change classification RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261105)
            self.geometry_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x, return_geometry=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        geometry = self.geometry_head(x) if return_geometry else None
        x = self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))
        return (x, geometry) if return_geometry else x


def build_resnet18_student(pretrained=True, num_classes=3, dropout=.3):
    return GeometryResNet18(num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def relation_loss(predicted, geometry, valid, variant="plain", sample_seeds=None):
    """Logistic ordinal loss; no gradients into teacher, weights or pair selection."""
    if variant not in ("plain", "stable", "shuffled"):
        raise ValueError(f"Unknown geometry variant: {variant}")
    prediction = predicted.flatten(1)
    target = geometry.detach().flatten(2)
    masks = valid.detach().flatten(1).bool()
    if target.shape[0] != prediction.shape[0] or target.shape[2] != prediction.shape[1] or masks.shape != prediction.shape:
        raise ValueError("Predicted and teacher geometry must share batch and grid dimensions")
    n = prediction.shape[1]
    generator = torch.Generator().manual_seed(PAIR_SEED)
    candidates = torch.triu_indices(n, n, offset=1)
    order = torch.randperm(candidates.shape[1], generator=generator)[:PAIR_COUNT]
    pairs = candidates[:, order].to(prediction.device)
    i, j = pairs
    losses, total, kept, changed = [], 0, 0, 0
    for b in range(prediction.shape[0]):
        mask = masks[b] & torch.isfinite(target[b]).all(0)
        if int(mask.sum()) < 2:
            losses.append(prediction[b].sum() * 0)
            continue
        # Per-image scale normalization only; no support-pool/test statistics.
        low, high = torch.quantile(target[b, 0, mask].float(), torch.tensor([.05, .95], device=target.device))
        scale = (high - low).clamp_min(1e-8)
        delta = (target[b, :, i] - target[b, :, j]) / scale
        eligible = mask[i] & mask[j] & (delta[0].abs() >= MIN_RELATIVE_GAP)
        total += int(eligible.sum())
        labels = delta[0].sign()
        if variant != "plain":
            eligible = eligible & (delta.abs() >= MIN_RELATIVE_GAP).all(0) & (delta.sign() == labels[None]).all(0)
        selected = torch.nonzero(eligible).flatten()
        count = len(selected)
        kept += count
        if count == 0:
            losses.append(prediction[b].sum() * 0)
            continue
        y = labels[selected]
        if variant == "shuffled":
            if sample_seeds is None:
                raise ValueError("Shuffled control requires deterministic per-image seeds")
            local = torch.Generator().manual_seed(int(sample_seeds[b]))
            permutation = torch.randperm(count, generator=local).to(y.device)
            shuffled = y[permutation]
            changed += int((shuffled != y).sum())
            y = shuffled
        margin = prediction[b, i[selected]] - prediction[b, j[selected]]
        losses.append(torch.nn.functional.softplus(-y * margin).mean())
    loss = torch.stack(losses).mean()
    return loss, {"valid_pairs": total, "selected_pairs": kept,
                  "selection_rate": kept / total if total else 0.0,
                  "shuffled_changed_pairs": changed}
