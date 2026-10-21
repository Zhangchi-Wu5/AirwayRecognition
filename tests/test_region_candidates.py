"""Synthetic geometry checks for label-free dark-region proposals."""
from dataclasses import asdict
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.region_candidates import CandidateConfig, extract_candidates


def scene():
    noise = np.random.default_rng(17).integers(-12, 13, (56, 56))
    noise = np.repeat(np.repeat(noise, 2, axis=0), 2, axis=1)
    rgb = np.clip(np.array([185, 145, 135])[None, None] + noise[..., None], 0, 255).astype(np.uint8)
    field = Image.new("L", (112, 112))
    ImageDraw.Draw(field).ellipse((3, 3, 109, 109), fill=255)
    rgb[np.asarray(field) == 0] = 0
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.ellipse((27, 39, 45, 57), fill=(18, 15, 12))
    draw.ellipse((66, 51, 84, 69), fill=(25, 22, 20))
    return image


def test_blank_images_have_no_candidates_and_finite_arrays():
    for value in (0, 25, 128, 255):
        result = extract_candidates(Image.new("RGB", (100, 90), (value,) * 3))
        assert not result["valid"].any()
        assert all(np.isfinite(v).all() for v in result.values())


def test_regions_exclude_black_border_and_all_other_interiors():
    result = extract_candidates(scene())
    assert result["valid"].sum() >= 2
    assert result["fov_mask"][0, 0, 0] == 0
    union = result["inner_masks"].any(axis=0)
    assert not (result["ring_masks"].astype(bool) & union).any()
    assert not (result["inner_masks"].astype(bool) & ~result["fov_mask"].astype(bool)).any()
    masks = result["inner_masks"][result["valid"]].astype(bool)
    for i in range(len(masks)):
        for j in range(i):
            assert (masks[i] & masks[j]).sum() / min(masks[i].sum(), masks[j].sum()) <= .3


def test_same_pixels_metadata_and_quantile_roundtrip_do_not_change_results():
    image = scene()
    image.info["patient_id"] = "not-consumed"
    first = extract_candidates(image)
    image.info["label"] = 2
    config = CandidateConfig(**json.loads(json.dumps(asdict(CandidateConfig()))))
    second = extract_candidates(image, config)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_geometry_follows_post_augmentation_pixels():
    original = extract_candidates(scene())
    mirrored = extract_candidates(scene().transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    original_x = sorted(original["geometry"][original["valid"], 0])
    mirrored_x = sorted(1 - mirrored["geometry"][mirrored["valid"], 0])
    np.testing.assert_allclose(original_x, mirrored_x, atol=.02)


def test_configuration_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        CandidateConfig(min_stable_levels=4)
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=0)


def test_duplicate_intensity_thresholds_are_not_stability_evidence():
    image = Image.new("RGB", (56, 56), (180,) * 3)
    ImageDraw.Draw(image).rectangle((20, 20, 28, 28), fill=(25,) * 3)
    # All three default quantiles equal 180; counting them three times would
    # spuriously certify the same component as stable across three thresholds.
    assert not extract_candidates(image)["valid"].any()
