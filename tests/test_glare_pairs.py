"""Checks for paired glare evaluation, including independently known metric values."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from scripts import evaluate_glare_pairs as glare
from src.data import get_eval_transforms


def _scope_image(color=(60, 90, 120)) -> Image.Image:
    pixels = np.zeros((96, 120, 3), dtype=np.uint8)
    pixels[12:84, 15:105] = color
    pixels[35:60, 45:75] = (120, 70, 40)
    return Image.fromarray(pixels)


@pytest.fixture
def manifest(tmp_path) -> pd.DataFrame:
    rows = []
    for label_id, label in enumerate(("lt", "yz", "zz")):
        path = tmp_path / f"image_{label}.png"
        _scope_image((60 + label_id * 15, 90, 120)).save(path)
        rows.append({"path": str(path), "patient_id": f"00{label_id + 1}",
                     "label": label, "label_id": label_id})
    return pd.DataFrame(rows)


def test_manifest_retains_label_order_and_patient_leading_zeroes(manifest) -> None:
    validated = glare.validate_manifest(manifest)
    by_class = validated.sort_values("label_id")
    assert by_class["patient_id"].tolist() == ["001", "002", "003"]
    assert by_class["label_id"].tolist() == [0, 1, 2]
    assert by_class["label"].tolist() == ["lt", "yz", "zz"]
    assert validated["case_id"].is_unique
    reversed_frame = glare.validate_manifest(manifest.iloc[::-1])
    assert dict(zip(validated["path"], validated["case_id"])) == dict(
        zip(reversed_frame["path"], reversed_frame["case_id"])
    )


@pytest.mark.parametrize("fault", ["label", "label_id", "missing_path", "duplicate", "patient"])
def test_manifest_rejects_invalid_inputs(manifest, fault) -> None:
    invalid = manifest.copy()
    if fault == "label":
        invalid.loc[0, "label"] = "zz"
    elif fault == "label_id":
        invalid.loc[0, "label_id"] = 9
    elif fault == "missing_path":
        invalid.loc[0, "path"] += ".missing"
    elif fault == "duplicate":
        invalid = pd.concat([invalid, invalid.iloc[[0]]], ignore_index=True)
    else:
        invalid.loc[0, "patient_id"] = None
    with pytest.raises((ValueError, FileNotFoundError)):
        glare.validate_manifest(invalid)


def test_zero_glare_preserves_exact_pixels() -> None:
    image = _scope_image()
    altered, mask = glare.make_glare_pair(image, coverage=0.0, seed=17)
    np.testing.assert_array_equal(np.asarray(altered), np.asarray(image))
    assert mask.shape == (image.height, image.width)
    assert not mask.any()


def test_glare_is_deterministic_nested_and_confined_to_visible_field() -> None:
    image = _scope_image()
    valid = glare.valid_field_mask(image)
    low, low_mask = glare.make_glare_pair(image, coverage=0.05, seed=17, valid_mask=valid)
    repeat, repeat_mask = glare.make_glare_pair(image, coverage=0.05, seed=17, valid_mask=valid)
    _, high_mask = glare.make_glare_pair(image, coverage=0.20, seed=17, valid_mask=valid)
    np.testing.assert_array_equal(np.asarray(low), np.asarray(repeat))
    np.testing.assert_array_equal(low_mask, repeat_mask)
    assert low_mask.any()
    assert not (low_mask & ~valid).any()
    assert not (low_mask & ~high_mask).any()
    np.testing.assert_array_equal(np.asarray(low)[~valid], np.asarray(image)[~valid])


def test_case_seed_changes_only_with_case_placement_or_global_seed() -> None:
    seeds = {glare.stable_seed(42, "case_a", 0), glare.stable_seed(42, "case_a", 1),
             glare.stable_seed(42, "case_b", 0), glare.stable_seed(43, "case_a", 0)}
    assert len(seeds) == 4
    assert glare.stable_seed(42, "case_a", 0) == glare.stable_seed(42, "case_a", 0)


class _TinyClassifier(torch.nn.Module):
    """A deterministic CPU model with no pretrained weights or batch coupling."""

    def forward(self, inputs):
        means = inputs.mean(dim=(2, 3))
        return torch.stack((means[:, 0], means[:, 1] + 0.1, means[:, 2] - 0.2), dim=1)


def test_fixed_crop_matches_training_and_does_not_reestimate_on_glare() -> None:
    image = _scope_image()
    bbox = glare.clean_crop_bbox(image)
    torch.testing.assert_close(glare.fixed_preprocess(image, bbox), get_eval_transforms(True)(image),
                               rtol=0, atol=0)
    torch.testing.assert_close(glare.fixed_preprocess(image), get_eval_transforms(False)(image),
                               rtol=0, atol=0)
    # Adversarial white border would move a newly estimated crop to the image edge.
    pixels = np.asarray(image).copy()
    pixels[:, :10] = 252
    changed_border = Image.fromarray(pixels)
    assert glare.clean_crop_bbox(changed_border) != bbox
    torch.testing.assert_close(glare.fixed_preprocess(changed_border, bbox),
                               get_eval_transforms(False)(changed_border.crop(bbox)), rtol=0, atol=0)
    assert glare.input_mask_coverage(np.ones((96, 120), dtype=bool), bbox) == 1.0
    assert glare.input_mask_coverage(np.zeros((96, 120), dtype=bool), bbox) == 0.0


def _collect(manifest, batch_size=2, crop_border=None):
    return glare.collect_predictions(
        models={"baseline": _TinyClassifier(), "full": _TinyClassifier()},
        manifest=manifest, coverages=[0, 0.05], placements=2, seed=17,
        batch_size=batch_size, device="cpu",
        crop_border=crop_border or {"baseline": False, "full": False},
    )


def test_predictions_are_paired_reproducible_and_zero_glare_is_exact(manifest) -> None:
    validated = glare.validate_manifest(manifest)
    first = _collect(validated, batch_size=1)
    reordered = _collect(validated.iloc[::-1], batch_size=4)
    pd.testing.assert_frame_equal(first, reordered, rtol=1e-6, atol=1e-7)
    assert len(first) == 3 * 2 * (1 + 2)  # Zero severity has no redundant placements.
    zero = first[first.coverage == 0]
    assert not zero["flipped"].any()
    assert not zero["true_probability_delta"].any()
    assert not zero["input_coverage"].any()
    for label in ("lt", "yz", "zz"):
        np.testing.assert_array_equal(zero[f"clean_prob_{label}"], zero[f"prob_{label}"])
        np.testing.assert_allclose(first[f"delta_prob_{label}"],
                                   first[f"prob_{label}"] - first[f"clean_prob_{label}"], atol=1e-7)
    probabilities = first[["prob_lt", "prob_yz", "prob_zz"]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
    assert first["perturbed_pred"].tolist() == np.array(["lt", "yz", "zz"])[probabilities.argmax(1)].tolist()
    expected_true = probabilities[np.arange(len(first)), first["true_label_id"]]
    clean = first[["clean_prob_lt", "clean_prob_yz", "clean_prob_zz"]].to_numpy()
    np.testing.assert_allclose(first.true_probability_delta,
                               expected_true - clean[np.arange(len(first)), first.true_label_id], atol=1e-7)
    left = first[first.model == "baseline"].reset_index(drop=True)
    right = first[first.model == "full"].reset_index(drop=True)
    pd.testing.assert_frame_equal(left.drop(columns="model"), right.drop(columns="model"))


def test_model_specific_crops_still_share_original_glare_and_frozen_bbox(manifest, monkeypatch) -> None:
    calls = []
    original_bbox = glare.clean_crop_bbox

    def record_bbox(image):
        calls.append(np.asarray(image).copy())
        return original_bbox(image)

    monkeypatch.setattr(glare, "clean_crop_bbox", record_bbox)
    result = _collect(glare.validate_manifest(manifest), crop_border={"baseline": False, "full": True})
    assert len(calls) == len(manifest)  # One estimate per clean image, never per perturbation.
    for _, case_rows in result.groupby("case_id"):
        for _, model_rows in case_rows.groupby("model"):
            assert model_rows.crop_bbox.nunique() == 1
        assert case_rows[case_rows.model == "baseline"].crop_bbox.iloc[0] != case_rows[
            case_rows.model == "full"].crop_bbox.iloc[0]
    paired = result.pivot(index=["case_id", "coverage", "placement"], columns="model",
                          values=["mask_sha256", "perturbation_seed", "actual_coverage"])
    for field in ("mask_sha256", "perturbation_seed", "actual_coverage"):
        assert paired[(field, "baseline")].equals(paired[(field, "full")])


def _metric_frame() -> pd.DataFrame:
    # Four independently specified cases. Only A/B are clean-correct for both models.
    vectors = {
        "baseline": [([.8, .1, .1], [.3, .6, .1]), ([.6, .3, .1], [.5, .4, .1]),
                     ([.1, .7, .2], [.2, .3, .5]), ([.2, .3, .5], [.6, .3, .1])],
        "full": [([.7, .2, .1], [.6, .3, .1]), ([.7, .2, .1], [.3, .6, .1]),
                 ([.6, .3, .1], [.5, .3, .2]), ([.2, .3, .5], [.6, .3, .1])],
    }
    rows = []
    for model, pairs in vectors.items():
        for i, (clean, perturbed) in enumerate(pairs):
            cp, pp = int(np.argmax(clean)), int(np.argmax(perturbed))
            for placement in (0, 1):
                rows.append(dict(
                    model=model, case_id="ABCD"[i], patient_id="p1" if i == 0 else "p2",
                    true_label_id=0, coverage=.1, placement=placement,
                    perturbation_seed=i * 2 + placement, mask_sha256=f"mask-{i}-{placement}",
                    clean_correct=int(cp == 0), perturbed_correct=int(pp == 0),
                    flipped=int(cp != pp), correct_to_wrong=int(cp == 0 and pp != 0),
                    wrong_to_correct=int(cp != 0 and pp == 0),
                    true_probability_delta=perturbed[0] - clean[0],
                    true_probability_drop=clean[0] - perturbed[0],
                    probability_l1_change=sum(abs(a - b) for a, b in zip(clean, perturbed)),
                    actual_coverage=.09, actual_fov_coverage=.12, input_coverage=.11,
                ))
    return pd.DataFrame(rows)


def test_hand_calculated_rates_denominators_and_full_minus_baseline() -> None:
    predictions = _metric_frame()
    summary = glare.summarize_predictions(predictions).set_index("model")
    assert summary.loc["baseline", "flip_rate"] == .75
    assert summary.loc["baseline", "correct_to_wrong_rate"] == .5
    assert summary.loc["baseline", "correct_to_wrong_all_rate"] == .25
    assert summary.loc["baseline", "wrong_to_correct_rate"] == .5
    assert summary.loc["baseline", "true_probability_delta_mean"] == pytest.approx(-.025)
    assert summary.loc["full", "correct_to_wrong_rate"] == pytest.approx(1 / 3)
    assert summary.loc["full", "true_probability_delta_mean"] == pytest.approx(-.05)
    assert (summary.n_observations == 8).all()
    comparison = glare.compare_models(predictions, n_bootstrap=0).iloc[0]
    assert comparison.flip_rate_difference == -.25
    assert comparison.correct_to_wrong_rate_difference == pytest.approx(-1 / 6)
    assert comparison.perturbed_accuracy_difference == .25
    assert comparison.true_probability_delta_mean_difference == pytest.approx(-.025)
    assert comparison.common_clean_correct_to_wrong_rate_difference == 0
    assert comparison.n_common_clean_correct_images == 2
    assert comparison.n_common_clean_correct_observations == 4
    assert comparison.bootstrap_method == "disabled"
    assert comparison.flip_rate_bootstrap_valid == 0
    assert np.isnan(comparison.flip_rate_ci_low)


def test_no_clean_correct_cases_reports_undefined_rate() -> None:
    predictions = _metric_frame().query("case_id == 'D'")
    summary = glare.summarize_predictions(predictions)
    assert summary.correct_to_wrong_rate.isna().all()
    assert summary.correct_to_wrong_all_rate.eq(0).all()
    result = glare.compare_models(predictions, n_bootstrap=10).iloc[0]
    assert np.isnan(result.correct_to_wrong_rate_difference)
    assert np.isnan(result.common_clean_correct_to_wrong_rate_difference)
    assert result.n_common_clean_correct_images == 0
    assert result.bootstrap_method == "insufficient_patients"


def test_bootstrap_preserves_patient_clusters_and_all_placements(monkeypatch) -> None:
    class KnownPatientDraws:
        def multinomial(self, n, pvals, size):
            assert n == 2 and size == 3
            assert list(pvals) == [.5, .5]
            # All p1 (A), then both patients (A/B/C/D), then all p2 (B/C/D).
            return np.array([[2, 0], [1, 1], [0, 2]])

    monkeypatch.setattr(glare.np.random, "default_rng", lambda seed: KnownPatientDraws())
    result = glare.compare_models(_metric_frame(), n_bootstrap=3).iloc[0]
    # Full minus baseline flip rates for these complete clusters: -1, -.25, 0.
    assert result.flip_rate_ci_low == pytest.approx(-.9625)
    assert result.flip_rate_ci_high == pytest.approx(-.0125)
    assert result.flip_rate_bootstrap_valid == 3
    assert result.n_observations == 8


def test_comparison_rejects_missing_or_mismatched_pairs() -> None:
    predictions = _metric_frame()
    with pytest.raises(ValueError, match="matched"):
        glare.compare_models(predictions.iloc[1:], n_bootstrap=0)
    predictions.loc[0, "mask_sha256"] = "different-original-perturbation"
    with pytest.raises(ValueError, match="mask_sha256"):
        glare.compare_models(predictions, n_bootstrap=0)


def test_cli_cpu_outputs_are_complete_and_refuse_overwriting(manifest, tmp_path, monkeypatch) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    manifest.to_csv(split_dir / "test.csv", index=False)
    checkpoint = tmp_path / "stub.pt"
    checkpoint.write_bytes(b"stub-checkpoint-for-provenance")
    output_dir = tmp_path / "results"
    monkeypatch.setattr(glare, "load_models", lambda args, device: {
        "baseline": _TinyClassifier(), "full": _TinyClassifier()})
    arguments = ["--baseline-checkpoint", str(checkpoint), "--full-checkpoint", str(checkpoint),
                 "--splits-dir", str(split_dir), "--output-dir", str(output_dir),
                 "--device", "cpu", "--coverages", "0", ".05", "--placements", "2",
                 "--bootstrap", "0", "--batch-size", "2"]
    glare.main(arguments)
    assert {p.name for p in output_dir.iterdir()} == {
        "per_image_predictions.csv", "summary.csv", "paired_comparison.csv", "summary.md", "metadata.json"}
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["completed"] is True
    assert metadata["class_order"] == ["lt", "yz", "zz"]
    assert metadata["n_prediction_rows"] == 18
    assert metadata["bootstrap"]["replicates"] == 0
    before = (output_dir / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="absent or empty"):
        glare.main(arguments)
    assert (output_dir / "metadata.json").read_bytes() == before
