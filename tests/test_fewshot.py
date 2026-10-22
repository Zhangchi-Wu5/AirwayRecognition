"""Patient-aware sampling, leakage checks and offline linear-probe integration."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src import fewshot
from scripts import run_resnet_fewshot as runner


def _fixtures(tmp_path, n_train=5):
    split_dir, image_dir = tmp_path / "splits", tmp_path / "images"
    split_dir.mkdir()
    image_dir.mkdir()
    for split, patients in (("train", range(n_train)), ("val", range(100, 103)), ("test", range(200, 203))):
        rows = []
        for patient in patients:
            for label, label_id in fewshot.LABEL_TO_ID.items():
                # A missing class must not exclude this patient from other classes.
                if split == "train" and patient == 0 and label == "zz":
                    continue
                name = f"{patient:04d}{label}.png"
                rows.append({"patient_id": f"{patient:04d}", "label": label, "label_id": label_id,
                             "path": f"/old/machine/{name}"})
                # Planning hashes raw bytes; it must not decode even train images.
                if split != "test":
                    (image_dir / name).write_bytes(name.encode())
        fewshot.write_csv(split_dir / f"{split}.csv", rows)
    return split_dir, image_dir


def _args(tmp_path, splits, images, **kwargs):
    result = SimpleNamespace(splits_dir=splits, image_root=images, output_root=tmp_path / "out",
                             shots=[1, 2, 3], repeats=4, sampling_seed=20261022, device="cpu",
                             batch_size=4, num_workers=0, prepare_only=False)
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


def _mutate_csv(path, mutate):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    fewshot.write_csv(path, rows)


def test_nested_exact_k_patient_unique_missing_class_and_reproducibility(tmp_path):
    splits, images = _fixtures(tmp_path)
    frames, _ = fewshot.load_splits(splits, images)
    plan = fewshot.make_support_plan(frames["train"], [1, 2, 3], repeats=20)
    assert plan == fewshot.make_support_plan(frames["train"], [3, 1, 2], repeats=20)
    assert len(plan) == 61
    assert sum(r["budget"] == "full" for r in plan) == 1
    assert plan[-1]["rows"] == frames["train"]
    assert {row["patient_id"] for row in frames["train"]} >= {"0000"}
    assert any(row["patient_id"] == "0000" for run in plan[:-1] for row in run["rows"])
    for repeat in range(20):
        previous = set()
        for run in [r for r in plan if r["repeat"] == repeat]:
            selected = {row["source_index"] for row in run["rows"]}
            assert len(selected) == run["shot"] * 3
            assert previous <= selected
            previous = selected
            for label in fewshot.LABEL_TO_ID:
                group = [r for r in run["rows"] if r["label"] == label]
                assert len(group) == len({r["patient_id"] for r in group}) == run["shot"]


def test_multiple_images_of_patient_sampled_once(tmp_path):
    splits, images = _fixtures(tmp_path)
    frames, _ = fewshot.load_splits(splits, images)
    additional = dict(frames["train"][0], image_name="extra.png", source_index=999)
    train = frames["train"] + [additional]
    plan = fewshot.make_support_plan(train, [1, 3], repeats=10)
    for run in plan[:-1]:
        keys = [(r["patient_id"], r["label"]) for r in run["rows"]]
        assert len(keys) == len(set(keys))


def test_split_loading_preserves_ids_rebases_paths_never_reads_test(tmp_path, monkeypatch):
    splits, images = _fixtures(tmp_path)
    original_hash = fewshot.sha256_file
    accessed = []
    def checked_hash(path):
        accessed.append(Path(path))
        assert not Path(path).name.startswith("020")
        return original_hash(path)
    monkeypatch.setattr(fewshot, "sha256_file", checked_hash)
    frames, metadata = fewshot.load_splits(splits, images)
    assert frames["train"][0]["patient_id"] == "0000"
    assert all(Path(r["path"]).parent == images for r in frames["train"])
    assert not Path(frames["test"][0]["path"]).exists()
    assert metadata["test"]["image_access"] == "identity_only_no_image_access"
    assert "images" not in metadata["test"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(label_id="2"), "label mapping"),
    (lambda rows: rows[0].update(label="unknown"), "label mapping"),
    (lambda rows: rows.append(rows[0]), "duplicate image"),
    (lambda rows: rows[0].update(patient_id=""), "missing identity"),
])
def test_malformed_training_manifest_rejected(tmp_path, mutation, match):
    splits, images = _fixtures(tmp_path)
    _mutate_csv(splits / "train.csv", mutation)
    with pytest.raises(ValueError, match=match):
        fewshot.load_splits(splits, images)


def test_patient_overlap_rejected_including_test(tmp_path):
    splits, images = _fixtures(tmp_path)
    _mutate_csv(splits / "test.csv", lambda rows: rows[0].update(patient_id="0000"))
    with pytest.raises(ValueError, match="Patient overlap"):
        fewshot.load_splits(splits, images)


def test_rebased_path_collision_rejected(tmp_path):
    splits, images = _fixtures(tmp_path)
    _mutate_csv(splits / "test.csv", lambda rows: rows[0].update(path="/another/server/0000lt.png"))
    with pytest.raises(ValueError, match="Resolved image path overlap"):
        fewshot.load_splits(splits, images)


def test_identical_content_across_train_val_rejected(tmp_path):
    splits, images = _fixtures(tmp_path)
    (images / "0100lt.png").write_bytes((images / "0000lt.png").read_bytes())
    with pytest.raises(ValueError, match="Identical image content"):
        fewshot.load_splits(splits, images)


def test_prepare_metadata_no_overwrite_and_validates_before_output(tmp_path):
    splits, images = _fixtures(tmp_path)
    args = _args(tmp_path, splits, images)
    frames, runs, protocol = fewshot.prepare_experiment(args)
    assert protocol["status"] == "prepared_not_run"
    assert protocol["full_reference_runs"] == 1
    assert len(list((args.output_root / "supports").glob("*.csv"))) == len(runs) == 13
    assert json.loads((args.output_root / "protocol.json").read_text())["test_images_accessed"] is False
    assert "0000" in (args.output_root / "supports" / "full.csv").read_text()
    with pytest.raises(ValueError, match="overwrite"):
        fewshot.prepare_experiment(args)
    args.output_root = tmp_path / "invalid_plan"
    args.shots = [100]
    with pytest.raises(ValueError, match="cannot supply"):
        fewshot.prepare_experiment(args)
    assert not args.output_root.exists()


def test_prepare_cli_never_imports_numeric_or_image_libraries(tmp_path):
    splits, images = _fixtures(tmp_path)
    code = """
import builtins, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'numpy', 'sklearn', 'pandas', 'PIL', 'matplotlib'}:
        raise AssertionError('Unexpected import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts.run_resnet_fewshot import main
main(sys.argv[1:])
"""
    result = subprocess.run([sys.executable, "-c", code, "--splits-dir", str(splits), "--image-root", str(images),
                             "--output-root", str(tmp_path / "stdlib_plan"), "--shots", "1", "2", "--repeats", "2", "--prepare-only"],
                            cwd=fewshot.PROJECT_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "Plan only" in result.stdout


def test_cli_defaults_and_no_local_checkpoint_argument():
    args = runner.parse_args(["--output-root", "/tmp/never-created-fewshot-test"])
    assert args.shots == [1, 5, 10, 20] and args.repeats == 20
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-root", "/tmp/never-created-fewshot-test", "--checkpoint", "local.pth"])


def test_per_image_normalization_uses_no_fitted_statistics():
    import numpy as np
    original = np.array([[3., 4.], [12., 5.]])
    alone = fewshot.l2_normalize(original[:1])
    combined = fewshot.l2_normalize(original)
    np.testing.assert_array_equal(alone, combined[:1])
    np.testing.assert_allclose(np.linalg.norm(combined, axis=1), 1)
    with pytest.raises(ValueError, match="Zero feature"):
        fewshot.l2_normalize(np.zeros((1, 2)))


def test_head_convergence_warning_is_failure(monkeypatch):
    import warnings
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    def fail_fit(self, x, y):
        self.n_iter_ = np.array([1])
        warnings.warn("forced nonconvergence", ConvergenceWarning)
        return self
    monkeypatch.setattr(LogisticRegression, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        fewshot.fit_head(np.eye(3), np.array([0, 1, 2]))


def test_offline_pipeline_reads_only_support_labels_and_full_runs_once(tmp_path, monkeypatch):
    import numpy as np
    splits, images = _fixtures(tmp_path)
    args = _args(tmp_path, splits, images, shots=[1, 2], repeats=2)
    frames, runs, protocol = fewshot.prepare_experiment(args)
    fitted_sizes = []
    real_fit = fewshot.fit_head
    def spy_fit(x, y):
        fitted_sizes.append((len(x), len(y)))
        return real_fit(x, y)
    monkeypatch.setattr(fewshot, "fit_head", spy_fit)
    def tiny_extractor(frames, args):
        # No torch or download; unique synthetic features with known separation.
        assert set(frames) == {"train", "val", "test"}
        result = {}
        for split in ("train", "val"):
            result[split] = np.array([[float(r["label_id"] == k) + .001 * (i + 1) for k in range(3)]
                                     for i, r in enumerate(frames[split])])
        return result, {"device_used": "synthetic_test", "pretrained_state_sha256": "test-only"}
    results, summary = fewshot.execute_experiment(args, frames, runs, protocol, extractor=tiny_extractor)
    assert fitted_sizes == [(3, 3), (6, 6), (3, 3), (6, 6), (14, 14)]
    assert all(r["accuracy"] == 1 for r in results)
    assert len(results) == 5
    assert all(r["sample_sd"] is None for r in summary if r["budget"] == "full")
    assert json.loads((args.output_root / "protocol.json").read_text())["status"] == "completed"
    assert len(list((args.output_root / "heads").glob("*.npz"))) == 5
    assert (args.output_root / "learning_curve.png").exists()
    assert (args.output_root / "learning_curve.pdf").exists()
    cache = np.load(args.output_root / "features" / "train.npz", allow_pickle=False)
    assert cache["patient_id"][0] == "0000"
    np.testing.assert_allclose(np.linalg.norm(cache["features"], axis=1), 1)
    with (args.output_root / "predictions.csv").open() as handle:
        predicted = list(csv.DictReader(handle))
    assert len(predicted) == 5 * len(frames["val"])
    assert {p["patient_id"] for p in predicted} == {"0100", "0101", "0102"}


def test_failed_extraction_is_recorded_not_reported_as_complete(tmp_path):
    splits, images = _fixtures(tmp_path)
    args = _args(tmp_path, splits, images)
    frames, runs, protocol = fewshot.prepare_experiment(args)
    def fail(frames, args):
        raise RuntimeError("simulated unavailable weights")
    with pytest.raises(RuntimeError, match="unavailable weights"):
        fewshot.execute_experiment(args, frames, runs, protocol, extractor=fail)
    saved = json.loads((args.output_root / "protocol.json").read_text())
    assert saved["status"] == "failed"
    assert saved["failure"]["message"] == "simulated unavailable weights"
    assert "失败" in (args.output_root / "REPORT.md").read_text()


def test_extraction_freezes_eval_and_never_opens_test_images(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    import torchvision.models
    splits, images = _fixtures(tmp_path)
    # Replace planning-only bytes by real unique RGB images for extraction.
    for i, path in enumerate(sorted(images.iterdir())):
        Image.new("RGB", (30, 32), (10 + i, 20 + i, 30 + i)).save(path)
    frames, _ = fewshot.load_splits(splits, images)
    args = _args(tmp_path, splits, images)
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
            self.bn = nn.BatchNorm2d(4)
            self.fc = nn.Linear(2048, 3)
            self.seen = 0
        def forward(self, value):
            assert not self.training and not self.bn.training
            assert not torch.is_grad_enabled()
            self.seen += len(value)
            feature = self.bn(self.conv(value)).mean((2, 3)).repeat(1, 512)
            return self.fc(feature)
    model = TinyBackbone()
    original_bn = model.bn.running_mean.clone()
    def tiny_factory(*, weights):
        assert weights == torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        return model
    monkeypatch.setattr(torchvision.models, "resnet50", tiny_factory)
    original_open = Image.open
    def checked_open(path, *args, **kwargs):
        assert not Path(path).name.startswith("020")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", checked_open)
    features, info = fewshot.extract_features(frames, args)
    assert model.seen == len(frames["train"]) + len(frames["val"])
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert torch.equal(model.bn.running_mean, original_bn)
    assert features["train"].shape == (len(frames["train"]), 2048)
    np.testing.assert_allclose(np.linalg.norm(features["val"], axis=1), 1)
    assert info["device_used"] == "cpu"
    assert len(info["pretrained_state_sha256"]) == 64
