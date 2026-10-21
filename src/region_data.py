"""Aligned candidate inputs for the exploratory region-relation experiment.

The ordinary ResNet transforms are retained unchanged. Candidates are extracted
from the *final augmented input*, so masks and features share coordinates. This
does not restore structures removed by the existing random/center crop.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms, get_train_transforms
from src.region_candidates import CandidateConfig, extract_candidates


@contextmanager
def isolated_sample_rng(seed: int):
    """Use a CPU-only RNG stream without advancing the model/worker RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # torch.manual_seed also seeds CUDA; workers must only touch CPU state.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def tensor_to_rgb(image: torch.Tensor) -> Image.Image:
    """Invert ImageNet normalization in the same final input coordinates."""
    mean = image.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = image.new_tensor(IMAGENET_STD)[:, None, None]
    array = ((image.detach().cpu() * std + mean).clamp(0, 1) * 255).round()
    return Image.fromarray(array.to(torch.uint8).permute(1, 2, 0).numpy(), mode="RGB")


class RegionDataset(Dataset):
    def __init__(self, manifest, training=False, candidate_config=None, seed=42):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.training = bool(training)
        self.candidate_config = candidate_config or CandidateConfig()
        self.seed = int(seed)
        self.epoch = 0
        self.transform = (get_train_transforms(crop_border=False) if self.training
                          else get_eval_transforms(crop_border=False))

    def __len__(self):
        return len(self.manifest)

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        epoch = self.epoch if self.training else 0
        token = f"region-input-v1:{self.seed}:{epoch}:{int(index)}".encode("ascii")
        sample_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        with Image.open(row["path"]) as source:
            rgb = source.convert("RGB")
        with isolated_sample_rng(sample_seed):
            image = self.transform(rgb)
            candidates = extract_candidates(tensor_to_rgb(image), self.candidate_config)
        return {
            "image": image,
            "label": int(row["label_id"]),
            "candidates": {name: torch.from_numpy(np.ascontiguousarray(value))
                           for name, value in candidates.items()},
            "index": int(index),
            "case_id": f"{row.get('patient_id', '')}:{Path(row['path']).name}",
        }
