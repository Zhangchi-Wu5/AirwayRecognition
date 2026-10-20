"""Analytical tests for paired EQ rotation evaluation; no downloads or GPU training."""
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from scripts import evaluate_rotation_pairs as rotation


class TinyAttention(torch.nn.Module):
    def forward(self, inputs, return_attn=False):
        means = inputs.mean((2, 3))
        logits = torch.stack((means[:, 0], means[:, 1] + .2, means[:, 2] - .1), dim=1)
        attention = F.adaptive_avg_pool2d(inputs[:, :1], (7, 7)).sigmoid()
        return (logits, attention) if return_attn else logits


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for label_id, label in enumerate(rotation.CLASS_ORDER):
        pixels = np.zeros((80, 100, 3), dtype=np.uint8)
        pixels[10:70, 12:88] = (70 + label_id * 20, 90, 130)
        pixels[25:50, 25:60] = (140, 80, 40)
        path = tmp_path / f"{label_id:03}_{label}.png"
        Image.fromarray(pixels).save(path)
        rows.append(dict(path=str(path), patient_id=f"{label_id:03}",
                         label=label, label_id=label_id))
    return rotation.validate_manifest(pd.DataFrame(rows))


def test_zero_rotation_is_exact_object_identity():
    inputs = torch.randn(2, 3, 17, 19)
    rotated = rotation.rotate_inputs(inputs, 0)
    assert rotated is inputs
    assert torch.equal(rotated, inputs)


def test_rotation_direction_matches_independent_quarter_turn_oracle():
    # The helper is checked against an independent index permutation; the
    # diagnostic CLI intentionally allows only small angles.
    inputs = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    actual = rotation.rotate_inputs(inputs, 90)
    expected = torch.rot90(inputs, 1, dims=(-2, -1))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=3e-6)


@pytest.mark.parametrize("angles", [[], [0], [1, 2], [0, 5, 5], [0, float("nan")], [0, 16]])
def test_rejects_unplanned_or_invalid_angles(angles):
    with pytest.raises(ValueError):
        rotation.validate_angles(angles)


def test_attention_rotation_oracle_and_valid_mask():
    # Native map equals work grid, so the known target is an exact oracle.
    attention = torch.linspace(.1, .9, 56 * 56).reshape(1, 1, 56, 56)
    rotated = rotation.rotate_inputs(attention, 15)
    metrics = rotation.attention_diagnostics(attention, rotated, 15)
    assert metrics["attention_equivariance_mse"][0] == 0
    valid = rotation.rotate_inputs(torch.ones_like(attention), 15) > .99
    assert .8 < metrics["attention_valid_fraction"][0] < 1
    changed = rotated.clone()
    changed[~valid] = 100
    metrics_changed = rotation.attention_diagnostics(attention, changed, 15)
    assert metrics_changed["attention_equivariance_mse"][0] == 0
    torch.testing.assert_close(
        torch.as_tensor(metrics["attention_valid_fraction"]), valid.float().flatten(1).mean(1))


def test_constant_attention_can_look_equivariant_but_statistics_show_degeneracy():
    attention = torch.full((2, 1, 7, 7), .3)
    metrics = rotation.attention_diagnostics(attention, attention, 15)
    assert np.max(metrics["attention_equivariance_mse"]) < 1e-6
    np.testing.assert_allclose(metrics["clean_attention_std"], 0, atol=1e-15)
    np.testing.assert_allclose(metrics["clean_attention_entropy"], 1, atol=1e-12)
    zero = rotation.attention_diagnostics(attention * 0, attention * 0, 0)
    assert np.isnan(zero["clean_attention_entropy"]).all()
    np.testing.assert_array_equal(zero["attention_equivariance_mse"], 0)
    np.testing.assert_array_equal(zero["clean_attention_mean"], 0)


def test_negative_or_nonfinite_attention_is_rejected():
    attention = torch.ones(1, 1, 7, 7)
    with pytest.raises(ValueError):
        rotation.attention_diagnostics(attention, -attention, 5)
    attention[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        rotation.attention_diagnostics(attention, attention, 5)


def test_predictions_match_case_reordering_and_batch_size_and_identity(manifest):
    def collect(frame, batch):
        return rotation.collect_predictions(
            {"eq_off": TinyAttention(), "eq_on": TinyAttention()},
            frame, angles=[-5, 0, 5], batch_size=batch, device="cpu")
    first = collect(manifest, 1)
    reordered = collect(manifest.iloc[::-1], 3)
    pd.testing.assert_frame_equal(first, reordered, rtol=1e-6, atol=1e-7)
    assert len(first) == 3 * 3 * 2
    zero = first[first.angle_deg == 0]
    assert not zero.flipped.any()
    np.testing.assert_array_equal(zero.attention_equivariance_mse, 0)
    np.testing.assert_array_equal(zero.true_probability_delta, 0)
    np.testing.assert_array_equal(zero.probability_l1_change, 0)
    for label in rotation.CLASS_ORDER:
        np.testing.assert_array_equal(zero[f"clean_prob_{label}"], zero[f"prob_{label}"])
    left = first[first.model == "eq_off"].drop(columns="model").reset_index(drop=True)
    right = first[first.model == "eq_on"].drop(columns="model").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def analytical_predictions():
    # p1 has A, p2 has B/C: unequal image counts detect accidental image bootstrap.
    # Both models clean-correct on A/B; only eq_off is clean-correct on C.
    rows = []
    correct, wrong = np.array([.8, .1, .1]), np.array([.1, .8, .1])
    for model in rotation.MODEL_ORDER:
        for case in "ABC":
            clean = wrong if model == "eq_on" and case == "C" else correct
            for angle in [-5., 0., 5.]:
                if angle == 0:
                    probs = clean
                elif model == "eq_off":
                    probs = correct if case == "B" else wrong
                else:
                    probs = correct if case == "A" or (case == "B" and angle == 5.) else wrong
                base = dict(case_id=case, patient_id="p1" if case == "A" else "p2",
                            true_label="lt", true_label_id=0, image_name=case, angle_deg=angle)
                base.update({column: 0. for column in rotation.ATTENTION_COLUMNS})
                rows.append(rotation.prediction_row(base, model, clean, probs))
    return pd.DataFrame(rows)


def test_known_rates_common_cohort_and_probability_differences():
    predictions = analytical_predictions()
    summary = rotation.summarize_predictions(predictions)
    pooled = summary[summary.scope == "nonzero_mean"].set_index("model")
    assert pooled.loc["eq_off", "rotated_accuracy"] == pytest.approx(1/3)
    assert pooled.loc["eq_on", "rotated_accuracy"] == .5
    assert pooled.loc["eq_off", "common_clean_correct_to_wrong_rate"] == .5
    assert pooled.loc["eq_on", "common_clean_correct_to_wrong_rate"] == .25
    assert pooled.loc["eq_off", "correct_to_wrong_rate"] == pytest.approx(2/3)
    assert pooled.loc["eq_on", "correct_to_wrong_rate"] == .25
    assert pooled.loc["eq_off", "true_probability_drop_mean"] == pytest.approx(.7 * 2/3)
    assert pooled.loc["eq_on", "true_probability_drop_mean"] == pytest.approx(.7 / 6)
    np.testing.assert_allclose(summary.true_probability_delta_mean, -summary.true_probability_drop_mean)
    comparison = rotation.compare_models(predictions, 0).iloc[0]
    assert comparison.rotated_accuracy_difference == pytest.approx(1/6)
    assert comparison.common_clean_correct_to_wrong_rate_difference == -.25
    assert comparison.n_common_clean_correct_images == 2
    assert np.isnan(comparison.rotated_accuracy_ci_low)
    assert comparison.bootstrap_method == "disabled"


def test_patient_bootstrap_matches_two_patient_oracle_and_keeps_angles_together():
    result = rotation.compare_models(analytical_predictions(), 10000, seed=31).iloc[0]
    # Sampling two patients gives: p1 twice (diff 1), both (1/6), p2 twice (-1/4).
    assert result.rotated_accuracy_ci_low == -.25
    assert result.rotated_accuracy_ci_high == 1
    assert result.common_clean_correct_to_wrong_rate_ci_low == -1
    assert result.common_clean_correct_to_wrong_rate_ci_high == .5
    assert result.rotated_accuracy_bootstrap_valid == 10000
    assert result.n_patients == 2
    assert result.n_images == 3
    assert result.n_observations == 6
    reordered = rotation.compare_models(analytical_predictions().iloc[::-1], 10000, seed=31)
    pd.testing.assert_series_equal(result, reordered.iloc[0])


def test_zero_denominators_and_single_patient_remain_undefined():
    predictions = analytical_predictions()
    predictions["patient_id"] = "one"
    result = rotation.compare_models(predictions, 30).iloc[0]
    assert result.bootstrap_method == "insufficient_patients"
    assert result.rotated_accuracy_bootstrap_valid == 0
    assert np.isnan(result.rotated_accuracy_ci_low)
    # eq_off has no clean errors, so its error-to-correct conditional rate is undefined.
    assert np.isnan(result.eq_off_wrong_to_correct_rate)


@pytest.mark.parametrize("fault", ["missing_pair", "duplicate", "metadata", "clean_changed", "zero_changed", "missing_angle"])
def test_invalid_pairing_cannot_be_summarized(fault):
    frame = analytical_predictions()
    if fault == "missing_pair":
        frame = frame.iloc[:-1]
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif fault == "metadata":
        frame.loc[0, "patient_id"] = "different"
    elif fault == "clean_changed":
        frame.loc[0, "clean_prob_lt"] = .5
    elif fault == "zero_changed":
        frame.loc[frame.angle_deg.eq(0), "prob_lt"] = .5
    else:
        frame = frame[~(frame.case_id.eq("A") & frame.angle_deg.eq(5))]
    with pytest.raises(ValueError):
        rotation.summarize_predictions(frame)


def test_cli_rejects_test_split(tmp_path):
    with pytest.raises(SystemExit):
        rotation.parse_args(["--eq-off-checkpoint", "a", "--eq-on-checkpoint", "b",
                             "--splits-dir", str(tmp_path), "--output-dir", "out", "--split", "test"])


def test_complete_cli_output_with_tiny_models(manifest, tmp_path, monkeypatch):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    manifest.drop(columns="case_id").to_csv(split_dir / "val.csv", index=False)
    off, on = tmp_path / "off.pt", tmp_path / "on.pt"
    off.write_bytes(b"test-checkpoint-off")
    on.write_bytes(b"test-checkpoint-on")
    monkeypatch.setattr(rotation, "load_models", lambda args, device: {
        "eq_off": TinyAttention(), "eq_on": TinyAttention()})
    output = tmp_path / "results"
    args = ["--eq-off-checkpoint", str(off), "--eq-on-checkpoint", str(on),
            "--splits-dir", str(split_dir), "--output-dir", str(output),
            "--device", "cpu", "--batch-size", "2", "--angles", "-5", "0", "5", "--bootstrap", "20"]
    rotation.main(args)
    assert {path.name for path in output.iterdir()} == {
        "per_image_predictions.csv", "summary.csv", "paired_comparison.csv", "metadata.json", "summary.md"}
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["completed"]
    assert metadata["n_prediction_rows"] == 18
    assert metadata["n_patients"] == 3
    assert metadata["perturbation"]["padding_rgb"] == [.485, .456, .406]
    assert metadata["training_provenance"]["eq_off"]["config"] is None
    with pytest.raises(ValueError, match="will not be overwritten"):
        rotation.main(args)


def test_training_metadata_checks_nested_regularization_and_initialization(tmp_path):
    paths = {}
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("patient_id,path\n001,example.png\n")
    for name, weight in [("eq_off", 0.), ("eq_on", .1)]:
        directory = tmp_path / name
        directory.mkdir()
        checkpoint = directory / "last_model.pt"
        checkpoint.write_bytes(name.encode())
        config = {
            "protocol_version": "eq_sham_control_v1", "completed": True, "eq_auxiliary_forward": True,
            "fixed_architecture": "AnatomyAttentionResNet50_layer4_7x7",
            "primary_checkpoint": "last_model.pt", "crop_border": True,
            "initial_state_sha256": "same", "arguments": {"dropout": .3},
            "regularization": {"lambda_eq": weight, "lambda_pf": .1, "max_angle": 15},
            "checkpoint_sha256": {"last_model.pt": rotation.sha256_file(checkpoint)},
            "splits": {"val": {"csv_sha256": rotation.sha256_file(val_csv)}},
        }
        (directory / "run_config.json").write_text(json.dumps(config))
        paths[name] = checkpoint
    rotation._training_provenance(paths, .3, val_csv)
    different_val = tmp_path / "different_val.csv"
    different_val.write_text("patient_id,path\n002,another.png\n")
    with pytest.raises(ValueError, match="Validation split differs"):
        rotation._training_provenance(paths, .3, different_val)
    with pytest.raises(ValueError, match="dropout"):
        rotation._training_provenance(paths, .5, val_csv)
    config_path = paths["eq_on"].parent / "run_config.json"
    changed = json.loads(config_path.read_text())
    changed["initial_state_sha256"] = "different"
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="initial_state_sha256"):
        rotation._training_provenance(paths, .3)
    changed["initial_state_sha256"] = "same"
    changed["regularization"]["lambda_eq"] = 0
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="must be positive"):
        rotation._training_provenance(paths, .3)
