"""Label-free dark-region proposals, not anatomical lumen annotations.

Run on the *post-augmentation* RGB image before tensor normalization. Coordinates
and masks therefore describe exactly the pixels seen by the classifier.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CandidateConfig:
    max_candidates: int = 3
    work_size: int = 56
    dark_quantiles: tuple = (0.15, 0.25, 0.35)
    black_threshold: float = 24.0
    min_area_fraction: float = 0.003
    max_area_fraction: float = 0.25
    min_contrast: float = 0.06
    stability_iou: float = 0.55
    min_stable_levels: int = 2
    ring_radius: int = 3
    min_ring_pixels: int = 4

    def __post_init__(self):
        if self.max_candidates < 1 or self.work_size < 8:
            raise ValueError("max_candidates >= 1 and work_size >= 8 required")
        if not (2 <= self.min_stable_levels <= len(self.dark_quantiles)):
            raise ValueError("min_stable_levels must fit dark_quantiles")
        if any(not 0 < q < 1 for q in self.dark_quantiles):
            raise ValueError("dark_quantiles must be between zero and one")
        if len(set(self.dark_quantiles)) != len(self.dark_quantiles):
            raise ValueError("dark_quantiles must be distinct")
        if not 0 < self.min_area_fraction < self.max_area_fraction < 1:
            raise ValueError("invalid candidate area range")
        if not 0 < self.stability_iou <= 1 or not 0 < self.min_contrast <= 1:
            raise ValueError("invalid stability or contrast threshold")
        if not 0 <= self.black_threshold < 255 or self.ring_radius < 1 or self.min_ring_pixels < 1:
            raise ValueError("invalid border/ring configuration")


def _components(mask):
    """Eight-connected components; intentionally no scipy/opencv dependency."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue, pixels = deque([(int(y), int(x))]), []
        seen[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        result = np.zeros_like(mask, dtype=bool)
        yy, xx = np.asarray(pixels).T
        result[yy, xx] = True
        yield result


def _dilate(mask, radius):
    h, w = mask.shape
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                result |= padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return result


def extract_candidates(image: Image.Image, config: CandidateConfig = CandidateConfig()):
    """Return fixed-size numpy arrays; no labels, filenames or patient IDs used.

    geometry = (cx, cy, area, width, height, contrast). Spatial quantities use
    image coordinates divided by work_size (area by work_size**2); contrast is
    (surround luminance - interior luminance) / 255. quality is a heuristic
    stability/contrast score, NOT a calibrated probability of anatomy.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    s, k = config.work_size, config.max_candidates
    rgb = np.asarray(image.convert("RGB").resize((s, s), Image.Resampling.BILINEAR), dtype=np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    fov = np.ones((s, s), dtype=bool)
    # Only edge-connected near-black components are treated as outside the FOV.
    # An internal dark region is retained even when its intensity is near zero.
    for part in _components(rgb.max(axis=2) <= config.black_threshold):
        if part[0].any() or part[-1].any() or part[:, 0].any() or part[:, -1].any():
            fov[part] = False
    result = {
        "inner_masks": np.zeros((k, s, s), dtype=np.float32),
        "ring_masks": np.zeros((k, s, s), dtype=np.float32),
        "valid": np.zeros(k, dtype=bool),
        "quality": np.zeros(k, dtype=np.float32),
        "geometry": np.zeros((k, 6), dtype=np.float32),
        "fov_mask": fov[None].astype(np.float32),
    }
    n_fov = int(fov.sum())
    if n_fov == 0:
        return result
    # Different quantile names can yield the same threshold on flat images.
    # Such duplicate masks do not constitute independent stability evidence.
    thresholds = np.unique(np.quantile(gray[fov], config.dark_quantiles))
    if len(thresholds) < config.min_stable_levels:
        return result
    levels = []
    for threshold in thresholds:
        levels.append([m for m in _components((gray < threshold) & fov)
                       if max(2, config.min_area_fraction * n_fov) <= m.sum()
                       <= config.max_area_fraction * n_fov])
    proposals = []
    for level, masks in enumerate(levels):
        for mask in masks:
            matches = 1
            for other_level, others in enumerate(levels):
                if other_level == level:
                    continue
                if any((mask & other).sum() / (mask | other).sum() >= config.stability_iou for other in others):
                    matches += 1
            ring = _dilate(mask, config.ring_radius) & fov & ~mask
            if matches < config.min_stable_levels or ring.sum() < config.min_ring_pixels:
                continue
            contrast = float((gray[ring].mean() - gray[mask].mean()) / 255)
            if contrast >= config.min_contrast:
                score = (matches / len(levels)) * min(contrast / 0.30, 1.0)
                proposals.append((score, int(mask.sum()), mask))
    proposals.sort(key=lambda p: (p[0], p[1]), reverse=True)
    chosen = []
    for score, area, mask in proposals:
        if any((mask & other[2]).sum() / min(area, other[1]) > 0.30 for other in chosen):
            continue
        chosen.append((score, area, mask))
        if len(chosen) == k:
            break
    interiors = np.any([c[2] for c in chosen], axis=0) if chosen else np.zeros_like(fov)
    for index, (score, area, mask) in enumerate(chosen):
        ring = _dilate(mask, config.ring_radius) & fov & ~interiors
        if ring.sum() < config.min_ring_pixels:
            continue
        contrast = float((gray[ring].mean() - gray[mask].mean()) / 255)
        if contrast < config.min_contrast:
            continue
        yy, xx = np.where(mask)
        result["inner_masks"][index] = mask
        result["ring_masks"][index] = ring
        result["valid"][index] = True
        result["quality"][index] = min(score, contrast / 0.30)
        result["geometry"][index] = ((xx.mean() + .5) / s, (yy.mean() + .5) / s,
                                     area / (s * s), (xx.max() - xx.min() + 1) / s,
                                     (yy.max() - yy.min() + 1) / s, contrast)
    return result
