"""BN recalibration checks: leakage boundaries, frozen weights, and paired effects."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from scripts import diagnose_bn_stats as diagnostic
from scripts import evaluate_glare_pairs as glare


class TinyBNClassifier(torch.nn.Module):
    """Records execution modes without downloading or constructing a large model."""

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(p=.75)
        self.bn = torch.nn.BatchNorm2d(3, momentum=.17)
        self.classifier = torch.nn.Linear(3, 3)
        self.register_buffer("non_bn_marker", torch.tensor([2., 7.]))
        self.observed_modes = []
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(3))
            self.classifier.bias.zero_()

    def forward(self, images):
        self.observed_modes.append((self.training, self.dropout.training, self.bn.training))
        features = self.bn(self.dropout(images))
        return self.classifier(features.mean(dim=(2, 3)))


def _manifest(tmp_path, prefix, colors=(40, 80, 160)):
    rows = []
    for i, color in enumerate(colors):
        path = tmp_path / f"{prefix}_{i}.png"
        pixels = np.zeros((64, 80, 3), dtype=np.uint8)
        pixels[6:-6, 8:-8] = (color, min(color + 10, 255), max(color - 10, 0))
        Image.fromarray(pixels).save(path)
        rows.append(dict(path=str(path), patient_id=f"{prefix}_patient_{i}",
                         label=("lt", "yz", "zz")[i % 3], label_id=i % 3))
    return glare.validate_manifest(pd.DataFrame(rows))


def _mode_snapshot(model):
    return {name: module.training for name, module in model.named_modules()}


def test_calibration_changes_only_bn_statistics_and_restores_mixed_modes(tmp_path):
    train = _manifest(tmp_path, "train")
    model = TinyBNClassifier().train()
    model.bn.eval()  # Preserve heterogeneous module modes, not only model.training.
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    modes = _mode_snapshot(model)
    changes, integrity = diagnostic.recalibrate_bn(model, train, 2, "cpu")

    assert _mode_snapshot(model) == modes
    assert model.bn.momentum == .17
    assert all(mode == (False, False, True) for mode in model.observed_modes)
    assert len(model.observed_modes) == 2
    allowed = {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    after = model.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in allowed)
    for key in before.keys() - allowed:
        assert torch.equal(before[key], after[key]), key
    assert all(parameter.grad is None for parameter in model.parameters())
    assert integrity["parameters_unchanged"] is True
    assert integrity["non_bn_buffers_unchanged"] is True
    assert integrity["parameter_sha256_before"] == integrity["parameter_sha256_after"]
    assert integrity["non_bn_buffer_sha256_before"] == integrity["non_bn_buffer_sha256_after"]
    assert integrity["n_images"] == 3
    assert integrity["n_batches"] == 2
    assert changes["module"].tolist() == ["bn"]
    assert int(changes.iloc[0].batches_after) == 2


def test_last_batch_is_weighted_by_images_not_counted_as_full_batch(tmp_path, monkeypatch):
    train = _manifest(tmp_path, "weighted", colors=(10, 20, 90))
    # Sorting must keep the deliberately uneven last image last.
    train["case_id"] = ["A", "B", "C"]

    def center_tensor(image):
        pixel = np.asarray(image)[image.height // 2, image.width // 2].copy()
        return torch.tensor(pixel, dtype=torch.float32)[:, None, None].expand(3, 2, 2).clone()

    monkeypatch.setattr(diagnostic, "get_eval_transforms", lambda crop_border=False: center_tensor)
    model = TinyBNClassifier()
    diagnostic.recalibrate_bn(model, train, 2, "cpu")
    # Images have first-channel values 10, 20, 90: exact weighted mean is 40,
    # while equal weighting of two batches would incorrectly give 52.5.
    torch.testing.assert_close(model.bn.running_mean, torch.tensor([40., 50., 30.]))
    # This algorithm averages per-batch unbiased variances, not pooled variance.
    torch.testing.assert_close(model.bn.running_var, torch.full((3,), (200 / 7) * (2 / 3)))
    assert model.bn.num_batches_tracked.item() == 2


def test_dropout_is_disabled_and_recalibration_order_is_repeatable(tmp_path):
    train = _manifest(tmp_path, "repeat")
    first = TinyBNClassifier().train()
    second = copy.deepcopy(first)
    torch.manual_seed(1)
    diagnostic.recalibrate_bn(first, train, 2, "cpu")
    torch.manual_seed(789)
    diagnostic.recalibrate_bn(second, train.iloc[::-1], 2, "cpu")
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key]), key


def test_modes_and_momentum_are_restored_after_forward_error(tmp_path):
    class FailingClassifier(TinyBNClassifier):
        def forward(self, images):
            super().forward(images)
            raise RuntimeError("deliberate calibration failure")

    model = FailingClassifier().train()
    model.classifier.eval()
    model.bn.momentum = None
    modes = _mode_snapshot(model)
    with pytest.raises(RuntimeError, match="deliberate"):
        diagnostic.recalibrate_bn(model, _manifest(tmp_path, "failure"), 2, "cpu")
    assert _mode_snapshot(model) == modes
    assert model.bn.momentum is None


@pytest.mark.parametrize("state", ["parameter", "non_bn_buffer"])
def test_integrity_guard_rejects_forward_mutating_frozen_state(tmp_path, state):
    class MutatingClassifier(TinyBNClassifier):
        def forward(self, images):
            logits = super().forward(images)
            if state == "parameter":
                self.classifier.bias.add_(1)
            else:
                self.non_bn_marker.add_(1)
            return logits

    model = MutatingClassifier().train()
    modes = _mode_snapshot(model)
    with pytest.raises(RuntimeError, match="parameters or non-BN buffers"):
        diagnostic.recalibrate_bn(model, _manifest(tmp_path, "mutation"), 2, "cpu")
    assert _mode_snapshot(model) == modes
    assert model.bn.momentum == .17


def test_calibration_never_needs_or_uses_labels(tmp_path):
    train = _manifest(tmp_path, "unlabelled")
    with_labels = TinyBNClassifier()
    without_labels = copy.deepcopy(with_labels)
    diagnostic.recalibrate_bn(with_labels, train, 2, "cpu")
    diagnostic.recalibrate_bn(without_labels, train[["case_id", "path"]], 2, "cpu")
    for key in with_labels.state_dict():
        assert torch.equal(with_labels.state_dict()[key], without_labels.state_dict()[key]), key


@pytest.mark.parametrize("fault", ["patient", "path", "resolved_path"])
def test_train_and_validation_overlap_is_rejected(tmp_path, fault):
    train = _manifest(tmp_path, "train")
    val = _manifest(tmp_path, "val")
    diagnostic.validate_split_separation(train, val)
    if fault == "patient":
        val.loc[0, "patient_id"] = train.loc[1, "patient_id"]
    elif fault == "path":
        val.loc[0, "path"] = train.loc[1, "path"]
    else:
        path = tmp_path / "linked_image.png"
        path.symlink_to(train.loc[1, "path"])
        val.loc[0, "path"] = str(path)
    with pytest.raises(ValueError, match="(?i)(overlap|leak|shared)"):
        diagnostic.validate_split_separation(train, val)


def test_calibration_rejects_no_tracked_bn_empty_data_and_invalid_batch(tmp_path):
    train = _manifest(tmp_path, "invalid")
    with pytest.raises(ValueError):
        diagnostic.recalibrate_bn(torch.nn.Linear(3, 3), train, 2, "cpu")
    model = TinyBNClassifier()
    model.bn = torch.nn.BatchNorm2d(3, track_running_stats=False)
    with pytest.raises(ValueError):
        diagnostic.recalibrate_bn(model, train, 2, "cpu")
    with pytest.raises(ValueError):
        diagnostic.recalibrate_bn(TinyBNClassifier(), train.iloc[:0], 2, "cpu")
    with pytest.raises(ValueError):
        diagnostic.recalibrate_bn(TinyBNClassifier(), train, 0, "cpu")


def _effect_predictions():
    # The models have intentionally different before/after clean-correct cohorts.
    # Baseline common cohort = A/B; full common cohort = B/C.
    vectors = {
        "baseline": {
            "before": [([.8, .1, .1], [.3, .6, .1]), ([.8, .1, .1], [.6, .3, .1]),
                       ([.1, .8, .1], [.2, .7, .1])],
            "after": [([.7, .2, .1], [.6, .3, .1]), ([.7, .2, .1], [.6, .3, .1]),
                      ([.7, .2, .1], [.3, .6, .1])],
        },
        "full": {
            "before": [([.1, .8, .1], [.2, .7, .1]), ([.8, .1, .1], [.6, .3, .1]),
                       ([.8, .1, .1], [.6, .3, .1])],
            "after": [([.1, .8, .1], [.2, .7, .1]), ([.7, .2, .1], [.3, .6, .1]),
                      ([.7, .2, .1], [.3, .6, .1])],
        },
    }
    results = {}
    for state in ("before", "after"):
        rows = []
        for model, states in vectors.items():
            for i, (clean, changed) in enumerate(states[state]):
                for placement in (0, 1):
                    base = dict(case_id="ABC"[i], patient_id=f"p{i}", true_label_id=0,
                                coverage=.1, placement=placement, perturbation_seed=i * 2 + placement,
                                mask_sha256=f"mask-{i}-{placement}", actual_coverage=.09,
                                actual_fov_coverage=.12, input_coverage=.11)
                    rows.append(glare._prediction_row(base, model, np.asarray(clean), np.asarray(changed)))
        results[state] = pd.DataFrame(rows)
    return results["before"], results["after"]


def test_effects_are_after_minus_before_within_each_model_not_across_models():
    before, after = _effect_predictions()
    effects = diagnostic.compare_bn_effects(before, after, n_bootstrap=0).set_index("model")
    assert set(effects.index) == {"baseline", "full"}
    assert effects.comparison.eq("recalibrated-minus-original").all()
    assert effects.loc["baseline", "common_clean_correct_to_wrong_rate_difference"] == -.5
    assert effects.loc["full", "common_clean_correct_to_wrong_rate_difference"] == 1
    assert effects.loc["baseline", "perturbed_accuracy_difference"] == pytest.approx(1 / 3)
    assert effects.loc["full", "perturbed_accuracy_difference"] == pytest.approx(-2 / 3)
    assert effects.n_common_clean_correct_images.eq(2).all()
    assert effects.n_common_clean_correct_observations.eq(4).all()
    assert effects.bootstrap_method.eq("disabled").all()
    assert effects.flip_rate_ci_low.isna().all()


def test_effects_reject_mismatched_glare_masks():
    before, after = _effect_predictions()
    after.loc[0, "mask_sha256"] = "different-perturbation"
    with pytest.raises(ValueError, match="mask_sha256"):
        diagnostic.compare_bn_effects(before, after, n_bootstrap=0)


def _cli_fixture(tmp_path, monkeypatch):
    train = _manifest(tmp_path, "train")
    val = _manifest(tmp_path, "val")
    splits = tmp_path / "splits"
    splits.mkdir()
    train.to_csv(splits / "train.csv", index=False)
    val.to_csv(splits / "val.csv", index=False)
    # No test.csv: selecting val by default must not read the held-out test split.
    checkpoint = tmp_path / "stub.pt"
    checkpoint.write_bytes(b"synthetic provenance placeholder")
    models = {name: TinyBNClassifier() for name in ("baseline", "full")}
    monkeypatch.setattr(diagnostic, "load_models", lambda args, device: models)
    output = tmp_path / "bn_results"
    args = ["--baseline-checkpoint", str(checkpoint), "--full-checkpoint", str(checkpoint),
            "--splits-dir", str(splits), "--output-dir", str(output), "--device", "cpu",
            "--batch-size", "2", "--coverages", "0", ".05", "--placements", "1",
            "--bootstrap", "0"]
    return args, output, train, val, models


def test_cli_uses_train_for_calibration_and_val_for_paired_evaluation(tmp_path, monkeypatch):
    args, output, train, val, models = _cli_fixture(tmp_path, monkeypatch)
    actual_recalibrate = diagnostic.recalibrate_bn
    actual_collect = diagnostic.collect_predictions
    calibration_cases, evaluation_cases, collected = [], [], []

    def record_calibration(model, train_manifest, *args, **kwargs):
        calibration_cases.append(set(train_manifest.case_id))
        return actual_recalibrate(model, train_manifest, *args, **kwargs)

    def record_predictions(*args, **kwargs):
        manifest = kwargs.get("manifest", args[1] if len(args) > 1 else None)
        evaluation_cases.append(set(manifest.case_id))
        result = actual_collect(*args, **kwargs)
        collected.append(result.copy())
        return result

    monkeypatch.setattr(diagnostic, "recalibrate_bn", record_calibration)
    monkeypatch.setattr(diagnostic, "collect_predictions", record_predictions)
    diagnostic.main(args)
    assert calibration_cases == [set(train.case_id)] * 2
    assert evaluation_cases == [set(val.case_id)] * 2
    assert len(collected) == 2
    pairing = ["case_id", "model", "coverage", "placement", "mask_sha256", "perturbation_seed"]
    pd.testing.assert_frame_equal(collected[0][pairing], collected[1][pairing])
    marker = output / "metadata.json"
    metadata = json.loads(marker.read_text())
    assert metadata["completed"] is True
    assert metadata["comparison"]["evaluation_split"] == "val"
    assert metadata["calibration"]["source"] == "train.csv only"
    assert metadata["calibration"]["checkpoint_saved"] is False
    assert metadata["splits"]["train"]["n_images"] == len(train)
    assert metadata["splits"]["val"]["n_images"] == len(val)
    assert {path.name for path in output.iterdir()} == {
        "original", "recalibrated", "bn_changes.csv", "bn_effect.csv", "summary.md", "metadata.json"}
    for phase in ("original", "recalibrated"):
        assert {path.name for path in (output / phase).iterdir()} == {
            "per_image_predictions.csv", "summary.csv", "paired_comparison.csv", "summary.md"}
    assert (tmp_path / "stub.pt").read_bytes() == b"synthetic provenance placeholder"
    before = marker.read_bytes()
    with pytest.raises(ValueError, match="absent or empty"):
        diagnostic.main(args)
    assert marker.read_bytes() == before


def test_cli_refuses_test_set_selection(tmp_path, monkeypatch):
    args, output, *_ = _cli_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        diagnostic.main(args + ["--split", "test"])
    assert not (output / "metadata.json").exists()


def test_failed_calibration_does_not_write_completion_marker(tmp_path, monkeypatch):
    args, output, *_ = _cli_fixture(tmp_path, monkeypatch)

    def failure(*args, **kwargs):
        raise RuntimeError("deliberate calibration failure")

    monkeypatch.setattr(diagnostic, "recalibrate_bn", failure)
    with pytest.raises(RuntimeError, match="deliberate"):
        diagnostic.main(args)
    assert not (output / "metadata.json").exists()
