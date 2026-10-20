"""Tests of the intervention and provenance, without downloading model weights."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import run_rotation_ablation as runner
from src.train import set_seed, train_one_epoch


class TinyAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.attention = nn.Conv2d(4, 1, 1)
        self.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(8, 3))
        self.forward_count = 0

    def forward(self, x, return_attn=False):
        self.forward_count += 1
        feature = self.backbone(x)
        attention = self.attention(feature).sigmoid()
        vector = torch.cat([feature.mean((2, 3)), (feature * attention).mean((2, 3))], 1)
        logits = self.fc(vector)
        return (logits, attention) if return_attn else logits

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = True


def _single_step(model, images, labels, masks, weight, sham=True):
    generator = torch.Generator().manual_seed(123)
    loader = DataLoader(TensorDataset(images, labels, masks), batch_size=len(images), generator=generator)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    metrics = {}
    set_seed(909)
    loss, accuracy = train_one_epoch(
        model, loader, optimizer, nn.CrossEntropyLoss(), device="cpu",
        reg={"lambda_pf": .1, "lambda_eq": weight, "max_angle": 15., "eq_auxiliary_forward": sham},
        component_metrics=metrics,
    )
    return loss, accuracy, metrics, torch.get_rng_state().clone()


def test_sham_matches_auxiliary_bn_and_rng_but_eq_changes_gradients():
    set_seed(12)
    original = TinyAttention()
    off, on = copy.deepcopy(original), copy.deepcopy(original)
    images = torch.randn(3, 3, 16, 16)
    labels = torch.tensor([0, 1, 2])
    masks = (torch.rand(3, 1, 16, 16) > .8).float()
    off_result = _single_step(off, images, labels, masks, 0.)
    on_result = _single_step(on, images, labels, masks, .1)
    assert off.forward_count == on.forward_count == 2
    assert off.backbone[1].num_batches_tracked.item() == on.backbone[1].num_batches_tracked.item() == 2
    assert torch.equal(off.backbone[1].running_mean, on.backbone[1].running_mean)
    assert torch.equal(off.backbone[1].running_var, on.backbone[1].running_var)
    assert torch.equal(off_result[3], on_result[3])
    assert off_result[2] == on_result[2]
    assert on_result[0] == pytest.approx(off_result[0] + .1 * off_result[2]["eq_loss"], abs=1e-6)
    assert not torch.equal(off.attention.weight.grad, on.attention.weight.grad)
    assert not torch.equal(off.attention.weight, on.attention.weight)


def test_default_zero_eq_keeps_historical_single_forward():
    set_seed(10)
    model = TinyAttention()
    images, labels, masks = torch.randn(2, 3, 16, 16), torch.tensor([0, 1]), torch.zeros(2, 1, 16, 16)
    loader = DataLoader(TensorDataset(images, labels, masks), batch_size=2)
    train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=.01), nn.CrossEntropyLoss(),
                    device="cpu", reg={"lambda_pf": .1, "lambda_eq": 0.})
    assert model.forward_count == 1
    assert model.backbone[1].num_batches_tracked.item() == 1


def _splits(tmp_path: Path):
    image_root = tmp_path / "images"
    split_root = tmp_path / "splits"
    image_root.mkdir()
    split_root.mkdir()
    rows = {
        "train": [("001", "lt", 0), ("001", "yz", 1), ("002", "lt", 0), ("002", "zz", 2)],
        "val": [("003", "lt", 0), ("003", "yz", 1)],
        "test": [("004", "zz", 2)],
    }
    for split, cases in rows.items():
        records = []
        for patient, label, label_id in cases:
            name = f"{patient}{label}.png"
            records.append({"patient_id": patient, "label": label, "label_id": label_id,
                            "path": f"/not/available/on/this/computer/{name}"})
            if split != "test":
                image = np.full((32, 32, 3), 90 + 20 * label_id, dtype=np.uint8)
                image[7:20, 10:23, 0] = 200
                Image.fromarray(image).save(image_root / name)
        pd.DataFrame(records).to_csv(split_root / f"{split}.csv", index=False)
    return split_root, image_root


def test_split_validation_preserves_ids_files_and_never_opens_test_images(tmp_path):
    splits, images = _splits(tmp_path)
    before = {path.name: path.read_bytes() for path in splits.iterdir()}
    train, val, metadata = runner.load_existing_splits(splits, images)
    assert train.patient_id.tolist() == ["001", "001", "002", "002"]
    assert set(val.patient_id) == {"003"}
    assert metadata["test"]["members"][0]["patient_id"] == "004"
    assert metadata["test"]["image_access"] == "paths_resolved_images_not_opened"
    assert before == {path.name: path.read_bytes() for path in splits.iterdir()}


def test_split_validation_rejects_patient_overlap(tmp_path):
    splits, images = _splits(tmp_path)
    test = pd.read_csv(splits / "test.csv", dtype=str)
    test["patient_id"] = "001"
    test.to_csv(splits / "test.csv", index=False)
    with pytest.raises(ValueError, match="Patient overlap"):
        runner.load_existing_splits(splits, images)


@pytest.mark.parametrize("split", ["val", "test"])
def test_split_validation_rejects_relocated_image_overlap(tmp_path, split):
    splits, images = _splits(tmp_path)
    frame = pd.read_csv(splits / f"{split}.csv", dtype=str)
    frame.loc[0, "path"] = "/a/different/legacy/location/001lt.png"
    frame.to_csv(splits / f"{split}.csv", index=False)
    with pytest.raises(ValueError, match="Resolved image path overlap"):
        runner.load_existing_splits(splits, images)


def test_state_fingerprint_includes_parameters_and_buffers():
    set_seed(10)
    model = TinyAttention()
    clone = copy.deepcopy(model)
    assert runner.state_fingerprint(model) == runner.state_fingerprint(clone)
    clone.backbone[1].num_batches_tracked.add_(1)
    assert runner.state_fingerprint(model) != runner.state_fingerprint(clone)


def test_runner_fixed_budget_artifacts_and_no_test_inference(tmp_path, monkeypatch):
    splits, images = _splits(tmp_path)
    monkeypatch.setattr(runner, "build_attention_resnet50", TinyAttention)
    output = tmp_path / "experiment"
    args = runner.parse_args([
        "--splits-dir", str(splits), "--image-root", str(images), "--output-root", str(output),
        "--device", "cpu", "--num-workers", "0", "--batch-size", "2", "--stage1-epochs", "0",
        "--stage2-epochs", "2", "--no-pretrained", "--skip-evaluation",
    ])
    completed = runner.run(args)
    assert completed == [output / "seed_42"]
    pair = json.loads((completed[0] / "pair_metadata.json").read_text())
    assert pair["initial_states_identical"] is True
    assert pair["training_input_streams_identical"] is True
    assert pair["test_images_loaded"] is False
    assert pair["primary_checkpoint"] == "last_model.pt"
    initial_hashes = []
    for arm in runner.ARMS:
        arm_dir = completed[0] / arm
        config = json.loads((arm_dir / "run_config.json").read_text())
        assert config["completed"] and config["completed_epochs"] == 2
        assert config["checkpoint_sha256"]["last_model.pt"] == runner.sha256_file(arm_dir / "last_model.pt")
        assert pair["arms"][arm]["run_config_sha256"] == runner.sha256_file(arm_dir / "run_config.json")
        assert config["regularization"]["eq_auxiliary_forward"] is True
        initial_hashes.append(config["initial_state_sha256"])
        history = pd.read_csv(arm_dir / "training_history.csv")
        assert len(history) == 2
        assert {"ce_loss", "pf_loss", "eq_loss"}.issubset(history.columns)
        state = torch.load(arm_dir / "last_model.pt", weights_only=True)
        assert state["backbone.1.num_batches_tracked"].item() == 8
    assert initial_hashes[0] == initial_hashes[1]
    before = (output / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="not overwritten"):
        runner.run(args)
    assert (output / "protocol.json").read_bytes() == before


@pytest.mark.parametrize("num_workers", [0, 4])
def test_augmented_stream_matches_with_sham_and_active_eq(tmp_path, num_workers):
    splits, images = _splits(tmp_path)
    train, val, _ = runner.load_existing_splits(splits, images)
    args = runner.parse_args(["--num-workers", str(num_workers), "--batch-size", "2", "--device", "cpu"])
    traces = []
    for weight in (0., .1):
        set_seed(42)
        model = TinyAttention()
        loader, _ = runner.make_loaders(train, val, args, 42, torch.device("cpu"))
        train_one_epoch(
            model, loader, torch.optim.SGD(model.parameters(), lr=.01), nn.CrossEntropyLoss(),
            device="cpu", reg={"lambda_pf": .1, "lambda_eq": weight, "max_angle": 15.,
                               "eq_auxiliary_forward": True},
        )
        traces.append(loader.epoch_traces)
    assert traces[0] == traces[1]
    assert traces[0][0]["n_images"] == 4


def test_default_evaluation_uses_only_val_and_last_checkpoints(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda args: [tmp_path / "seed_42"])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: calls.append(command))
    runner.main(["--splits-dir", str(tmp_path / "splits"), "--image-root", str(tmp_path / "images")])
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--split") + 1] == "val"
    assert command[command.index("--eq-off-checkpoint") + 1].endswith("eq_off/last_model.pt")
    assert command[command.index("--eq-on-checkpoint") + 1].endswith("eq_on/last_model.pt")


def test_invalid_epochs_and_duplicate_seeds_rejected():
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage2-epochs", "0"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--seeds", "42", "42"])
