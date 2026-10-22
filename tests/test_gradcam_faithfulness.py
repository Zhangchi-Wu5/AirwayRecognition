"""Known-answer tests for an inference-only, patient-paired heatmap screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from scripts import screen_gradcam_faithfulness as screen
from scripts import run_region_relation_ablation as runner
from tests.test_region_relation_training import _args, _splits


class SpatialToy(nn.Module):
    """The positive-class logit depends only on the red-channel spatial mean."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 0, 0] = 2.0

    def forward(self, images, candidates=None):
        value = self.bn(self.conv(images)).mean((1, 2, 3))
        return torch.stack((value, -value, value * 0 - 20), 1)


class TinyCamRegionModel(nn.Module):
    """Cheap trainable model retaining the actual loader's layer4 interface."""
    def __init__(self, variant="baseline", **kwargs):
        super().__init__()
        self.variant = variant
        self.backbone = nn.Module()
        self.backbone.layer4 = nn.Sequential(
            nn.Conv2d(3, 4, 1, stride=8), nn.BatchNorm2d(4), nn.ReLU()
        )
        self.fc = nn.Linear(4, 3)
        with torch.no_grad():
            self.backbone.layer4[0].weight.fill_(.2)
            self.backbone.layer4[0].bias.fill_(.4)
            self.fc.weight.fill_(-.1)
            self.fc.weight[0].fill_(.5)
            self.fc.bias.copy_(torch.tensor([3., 0., -3.]))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.layer4[1].eval()
        return self

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(self, images, candidates=None):
        return self.fc(self.backbone.layer4(images).mean((2, 3)))


def test_cam_matches_known_spatial_dependence_and_preserves_model():
    model = SpatialToy().eval()
    images = torch.zeros(2, 3, 8, 8)
    images[0, 0, 2:5, 1:4] = .8
    images[1, 0] = torch.arange(64).reshape(8, 8) / 64
    original_images = images.clone()
    before = screen.state_digest(model)
    # Existing gradients must not be cleared or overwritten by diagnostics.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    with torch.no_grad():
        result = screen.gradcam(model, model.conv, images)
    assert result["statuses"] == ["ok", "ok"]
    assert result["targets"].tolist() == [0, 0]
    expected = images[:, 0].numpy().copy()
    expected /= expected.max(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(result["maps"], expected, atol=1e-6)
    assert torch.equal(images, original_images)
    assert screen.state_digest(model) == before
    assert all(torch.equal(parameter.grad, old) for parameter, old in zip(model.parameters(), gradients))
    assert not model.conv._forward_hooks
    assert not model.conv._backward_hooks


def test_cam_still_works_with_all_parameters_frozen():
    model = SpatialToy().eval().requires_grad_(False)
    image = torch.zeros(1, 3, 8, 8)
    image[0, 0, 3, 4] = 1
    result = screen.gradcam(model, model.conv, image)
    assert result["statuses"] == ["ok"]
    assert np.unravel_index(result["maps"][0].argmax(), (8, 8)) == (3, 4)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("intensity,status", [(0., "zero_cam"), (1., "flat_cam")])
def test_zero_and_flat_heatmaps_are_explicit(intensity, status):
    model = SpatialToy().eval()
    result = screen.gradcam(model, model.conv, torch.full((1, 3, 8, 8), intensity))
    assert result["statuses"] == [status]
    assert np.isfinite(result["maps"]).all()


def test_gradcam_removes_hook_when_model_forward_fails():
    class Broken(SpatialToy):
        def forward(self, images, candidates=None):
            self.conv(images)
            raise RuntimeError("intentional downstream failure")
    model = Broken().eval()
    before = screen.state_digest(model)
    with pytest.raises(RuntimeError, match="intentional downstream"):
        screen.gradcam(model, model.conv, torch.ones(1, 3, 8, 8))
    assert not model.conv._forward_hooks
    assert screen.state_digest(model) == before


def test_target_probability_remains_original_class_after_prediction_flip():
    logits = torch.tensor([[0., 2., -1.]], dtype=torch.float64)
    original_targets = torch.tensor([0])
    probability, logodds = screen.target_scores(logits, original_targets)
    assert probability.item() == pytest.approx(1 / (1 + np.exp(2) + np.exp(-1)))
    assert probability.item() < logits.softmax(1).max().item()
    assert logodds.item() == pytest.approx(-np.log(np.exp(2) + np.exp(-1)))
    extreme = torch.tensor([[1000., 900., -1000.]])
    probability, logodds = screen.target_scores(extreme, original_targets)
    assert probability.item() == 1.0
    assert logodds.item() == pytest.approx(100.)
    assert torch.isfinite(logodds).all()


def test_square_window_means_and_matched_random_controls():
    cam = np.arange(100, dtype=float).reshape(10, 10)
    border = np.zeros((10, 10), dtype=bool)
    border[:, :2] = True
    means = screen.window_means(cam, 4)
    assert means.shape == (7, 7)
    assert means[1, 3] == pytest.approx(cam[1:5, 3:7].mean())
    first = screen.choose_windows(cam, border, .16, 20, "case-A", 42)
    second = screen.choose_windows(cam[::-1], border, .16, 20, "case-A", 42)
    assert first["status"] == "ok"
    assert first["side"] == 4 and first["actual_coverage"] == .16
    assert first["eligible_count"] == 35
    assert first == screen.choose_windows(cam, border, .16, 20, "case-A", 42)
    randoms = lambda plan: [window for window in plan["windows"] if window["region"] == "random"]
    assert randoms(first) == randoms(second)
    assert randoms(first) != randoms(screen.choose_windows(cam, border, .16, 20, "case-B", 42))
    for window in first["windows"]:
        y, x = window["y"], window["x"]
        assert border[y:y + 4, x:x + 4].mean() <= screen.MAX_BORDER_FRACTION
    high, low = first["windows"][:2]
    assert (high["y"], high["x"]) == (6, 6)
    assert (low["y"], low["x"]) == (0, 2)


def test_window_failure_is_reported_instead_of_constructing_controls():
    border = np.zeros((10, 10), dtype=bool)
    flat = screen.choose_windows(np.ones((10, 10)), border, .1, 3, "a", 42)
    blocked = screen.choose_windows(np.arange(100).reshape(10, 10), ~border, .1, 3, "a", 42)
    nonfinite = screen.choose_windows(np.full((10, 10), np.nan), border, .1, 3, "a", 42)
    assert flat["status"] == "flat_cam" and flat["windows"] == []
    assert blocked["status"] == "no_eligible_window" and blocked["eligible_count"] == 0
    assert nonfinite["status"] == "nonfinite_cam" and nonfinite["windows"] == []


def test_cam_signal_only_in_excluded_border_has_no_rankable_windows():
    cam = np.zeros((10, 10))
    cam[:, :2] = 1
    border = cam.astype(bool)
    plan = screen.choose_windows(cam, border, .16, 3, "a", 42)
    assert plan["eligible_count"] == 35
    assert plan["status"] == "flat_eligible_window_scores"
    assert plan["windows"] == []


@pytest.mark.parametrize("fill", screen.FILLS)
def test_perturbation_changes_only_the_exact_square_and_never_input(fill):
    rgb = np.random.default_rng(123).random((40, 40, 3), dtype=np.float32)
    original = rgb.copy()
    border = np.zeros((40, 40), bool)
    changed, fallback = screen.perturb_rgb(rgb, {"y": 8, "x": 9}, 10, fill, border)
    outside = np.ones((40, 40), bool)
    outside[8:18, 9:19] = False
    assert np.array_equal(rgb, original)
    assert np.array_equal(changed[outside], original[outside])
    assert np.any(changed[~outside] != original[~outside])
    assert np.isfinite(changed).all() and changed.min() >= 0 and changed.max() <= 1
    assert fallback == "none"
    if fill == "blur":
        cached, _ = screen.perturb_rgb(rgb, {"y": 8, "x": 9}, 10, fill, border, screen.blur_rgb(rgb))
        assert np.array_equal(changed, cached)


def test_ring_fill_excludes_border_and_records_fallbacks():
    rgb = np.ones((30, 30, 3), np.float32) * .6
    border = np.zeros((30, 30), bool)
    border[:, :10] = True
    rgb[border] = 0
    rgb[10:14, 10:14] = 1
    changed, fallback = screen.perturb_rgb(rgb, {"y": 10, "x": 10}, 4, "ring_mean", border)
    np.testing.assert_allclose(changed[10:14, 10:14], .6, atol=2e-6)
    assert fallback == "none"
    # Only the window itself is nonborder, so the local ring has no eligible pixels.
    border[:] = True
    border[10:14, 10:14] = False
    changed, fallback = screen.perturb_rgb(rgb, {"y": 10, "x": 10}, 4, "ring_mean", border)
    assert fallback == "image_nonborder_mean"
    np.testing.assert_allclose(changed[10:14, 10:14], 1.)
    border[:] = True
    changed, fallback = screen.perturb_rgb(rgb, {"y": 10, "x": 10}, 4, "ring_mean", border)
    assert fallback == "full_image_mean"
    np.testing.assert_allclose(changed[10:14, 10:14], np.broadcast_to(rgb.mean((0, 1)), (4, 4, 3)))


def _effect_frames(seeds=(42,)):
    clean, effects = [], []
    # Two images from the same patient deliberately get the same effect.
    for seed in seeds:
        for case, patient, value in (("a", "p1", 0.), ("b", "p1", 0.), ("c", "p2", 1.)):
            identity = {"seed": seed, "case_id": case, "patient_id": patient}
            clean.append(identity)
            effects.append({**identity, "coverage": .1, "actual_coverage": .1, "fill": "blur",
                            **{name: value for name in screen.EFFECTS}})
    return pd.DataFrame(effects), pd.DataFrame(clean)


def test_patient_bootstrap_known_answer_and_repeated_seeds_do_not_shrink_interval():
    effects, clean = _effect_frames()
    one = screen.summarize_effects(effects, clean, 2000, 42)
    assert np.allclose(one["mean"], 1 / 3)
    # Sampling 2 patient clusters gives possible means exactly 0, 1/3, 1.
    assert np.allclose(one.ci_low, 0.) and np.allclose(one.ci_high, 1.)
    effects, clean = _effect_frames((42, 2026, 3407))
    repeated = screen.summarize_effects(effects, clean, 2000, 42)
    combined = repeated[repeated.scope == "across_seeds_common_images"]
    np.testing.assert_allclose(combined[["mean", "ci_low", "ci_high"]], one[["mean", "ci_low", "ci_high"]])
    assert set(combined.n_images) == {3} and set(combined.n_patients) == {2}
    assert set(combined.n_training_seeds) == {3}


def test_multiseed_summary_uses_common_evaluable_cohort_and_counts_exclusions():
    effects, clean = _effect_frames((42, 2026))
    effects = effects[~((effects.seed == 2026) & (effects.case_id == "c"))]
    summary = screen.summarize_effects(effects, clean, 200, 42)
    combined = summary[summary.scope == "across_seeds_common_images"]
    assert set(combined.n_images) == {2} and set(combined.n_excluded_images) == {1}
    assert set(combined.n_patients) == {1} and np.allclose(combined["mean"], 0.)
    with pytest.raises(ValueError, match="identical images"):
        screen.summarize_effects(effects, clean.iloc[:-1], 200, 42)


@pytest.fixture
def completed_baseline(tmp_path, monkeypatch):
    splits, images = _splits(tmp_path)
    monkeypatch.setattr(runner, "build_region_relation_resnet50", TinyCamRegionModel)
    monkeypatch.setattr(screen, "build_region_relation_resnet50", TinyCamRegionModel)
    args = _args(tmp_path, splits, images, arms=["baseline"], stage1_epochs=0, stage2_epochs=1)
    seed_dir = runner.run(args)[0]
    return seed_dir, splits, images


def _resign_config(seed_dir, update):
    path = seed_dir / "baseline" / "run_config.json"
    config = json.loads(path.read_text())
    update(config)
    path.write_text(json.dumps(config))
    pair_path = seed_dir / "pair_metadata.json"
    pair = json.loads(pair_path.read_text())
    pair["variants"]["baseline"]["run_config_sha256"] = screen.sha256_file(path)
    pair_path.write_text(json.dumps(pair))


def test_real_training_provenance_and_inference_only_end_to_end(completed_baseline, tmp_path, monkeypatch):
    seed_dir, splits, images = completed_baseline
    # No nonbaseline weight file exists; training metadata and final baseline suffice.
    assert not (seed_dir / "relations" / "last_model.pt").exists()
    for name in ("001lt.png", "001yz.png", "002zz.png"):
        (images / name).unlink()
    opened = []
    actual_open = Image.open
    def recording_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", recording_open)
    args = screen.parse_args([
        "--run-dirs", str(seed_dir), "--splits-dir", str(splits), "--image-root", str(images),
        "--output-dir", str(tmp_path / "screen"), "--device", "cpu", "--batch-size", "3",
        "--coverages", ".1", "--random-placements", "2", "--bootstrap", "20", "--render-per-class", "0",
    ])
    before = screen.sha256_file(seed_dir / "baseline" / "last_model.pt")
    output = screen.run(args)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["completed"] is True and metadata["n_images"] == 2
    assert metadata["training_performed"] is False and metadata["weights_saved"] is False
    assert metadata["test_images_loaded"] is False
    assert set(opened) == {"003lt.png", "003yz.png"}
    assert before == screen.sha256_file(seed_dir / "baseline" / "last_model.pt")
    assert metadata["runs"][0]["state_sha256_before"] == metadata["runs"][0]["state_sha256_after"]
    clean = pd.read_csv(output / "per_image.csv")
    assert len(clean) == 2 and set(clean.cam_status) == {"ok"}
    perturbations = pd.read_csv(output / "perturbations.csv")
    assert len(perturbations) == 2 * 2 * (2 + 2)
    effects = pd.read_csv(output / "paired_effects.csv")
    assert len(effects) == 4
    for row in effects.itertuples():
        sample = perturbations[(perturbations.case_id == row.case_id) & (perturbations.fill == row.fill)]
        expected = sample.loc[sample.region == "high", "prob_drop"].iloc[0] - sample.loc[sample.region == "random", "prob_drop"].mean()
        assert row.high_minus_random_prob_drop == pytest.approx(expected)
    with pytest.raises(ValueError, match="not overwritten"):
        screen.run(args)


def test_all_unevaluable_maps_still_keep_all_original_rows(completed_baseline, tmp_path, monkeypatch):
    seed_dir, splits, images = completed_baseline
    actual_gradcam = screen.gradcam
    def zero_maps(*args, **kwargs):
        result = actual_gradcam(*args, **kwargs)
        result["maps"][:] = 0
        result["statuses"] = ["zero_cam"] * len(result["maps"])
        return result
    monkeypatch.setattr(screen, "gradcam", zero_maps)
    args = screen.parse_args([
        "--run-dirs", str(seed_dir), "--splits-dir", str(splits), "--image-root", str(images),
        "--output-dir", str(tmp_path / "all_zero"), "--device", "cpu", "--coverages", ".1", ".2",
        "--random-placements", "2", "--bootstrap", "20", "--render-per-class", "0",
    ])
    output = screen.run(args)
    assert len(pd.read_csv(output / "per_image.csv")) == 2
    windows = pd.read_csv(output / "window_status.csv")
    assert len(windows) == 4 and set(windows.status) == {"zero_cam"}
    assert pd.read_csv(output / "summary.csv").empty
    assert pd.read_csv(output / "paired_effects.csv").empty
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["completed"] is True and metadata["n_perturbations"] == 0


def test_changed_training_source_provenance_is_rejected(completed_baseline):
    seed_dir, splits, _ = completed_baseline
    _resign_config(seed_dir, lambda config: config["source_sha256"].update({"src/data.py": "0" * 64}))
    with pytest.raises(ValueError, match="Training source changed: src/data.py"):
        screen.validate_baseline_run(seed_dir, screen.read_identity_splits(splits))


def test_incomplete_config_and_stale_weights_rejected(completed_baseline):
    seed_dir, splits, _ = completed_baseline
    frames = screen.read_identity_splits(splits)
    _resign_config(seed_dir, lambda config: config.update(completed=False))
    with pytest.raises(ValueError, match="incomplete"):
        screen.validate_baseline_run(seed_dir, frames)
    _resign_config(seed_dir, lambda config: config.update(completed=True))
    with (seed_dir / "baseline" / "last_model.pt").open("ab") as stream:
        stream.write(b"stale checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint hash"):
        screen.validate_baseline_run(seed_dir, frames)


def test_checkpoint_semantics_verified_even_if_file_hashes_match(completed_baseline):
    seed_dir, splits, _ = completed_baseline
    checkpoint_path = seed_dir / "baseline" / "last_model.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["seed"] += 1
    torch.save(checkpoint, checkpoint_path)
    digest = screen.sha256_file(checkpoint_path)
    _resign_config(seed_dir, lambda config: config["checkpoint_sha256"].update({"last_model.pt": digest}))
    pair_path = seed_dir / "pair_metadata.json"
    pair = json.loads(pair_path.read_text())
    pair["variants"]["baseline"]["checkpoint_sha256"] = digest
    pair_path.write_text(json.dumps(pair))
    validated = screen.validate_baseline_run(seed_dir, screen.read_identity_splits(splits))
    with pytest.raises(ValueError, match="Checkpoint seed mismatch"):
        screen.load_baseline(validated, torch.device("cpu"))
