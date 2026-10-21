"""Provenance and stream isolation tests using real images and a tiny model."""
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from scripts import run_region_relation_ablation as runner
from src.data import get_eval_transforms
from src.region_candidates import CandidateConfig, extract_candidates
from src.region_data import RegionDataset, tensor_to_rgb


class TinyRegionModel(nn.Module):
    def __init__(self, variant="baseline", dropout=.3, **kwargs):
        super().__init__()
        self.variant = variant
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4, 3))
        self.extra = nn.Linear(1, 3)
        nn.init.zeros_(self.extra.weight)
        nn.init.zeros_(self.extra.bias)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone[1].eval()
        return self

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(self, images, candidates):
        logits = self.head(self.backbone(images.mean((2, 3))))
        if self.variant != "baseline":
            # Additional model RNG must not affect the next transformed sample.
            torch.rand(17)
            logits = logits + self.extra(candidates["quality"].sum(1, keepdim=True))
        return logits


def _splits(tmp_path):
    images, splits = tmp_path / "images", tmp_path / "splits"
    images.mkdir()
    splits.mkdir()
    definitions = {"train": [("001", "lt", 0), ("001", "yz", 1), ("002", "zz", 2)],
                   "val": [("003", "lt", 0), ("003", "yz", 1)],
                   "test": [("004", "zz", 2)]}
    yy, xx = np.mgrid[:260, :310]
    for split, cases in definitions.items():
        rows = []
        for patient, label, label_id in cases:
            name = f"{patient}{label}.png"
            rows.append({"patient_id": patient, "label": label, "label_id": label_id,
                         "path": f"/missing-server-path/{name}"})
            if split != "test":
                image = np.stack([(xx // 2 + label_id * 13) % 200 + 30,
                                  (yy // 2 + 35) % 200 + 20,
                                  (xx // 4 + yy // 4) % 120 + 30], -1).astype(np.uint8)
                image[(xx - 140) ** 2 + (yy - 140) ** 2 < 30 ** 2] = 15
                Image.fromarray(image).save(images / name)
        pd.DataFrame(rows).to_csv(splits / f"{split}.csv", index=False)
    return splits, images


def _args(tmp_path, splits, images, **overrides):
    args = runner.parse_args([
        "--splits-dir", str(splits), "--image-root", str(images),
        "--output-root", str(tmp_path / "experiment"), "--device", "cpu",
        "--num-workers", "0", "--batch-size", "2", "--stage1-epochs", "1",
        "--stage2-epochs", "1", "--no-pretrained", "--skip-evaluation",
    ])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_dataset_alignment_and_no_change_to_eval_transform(tmp_path):
    splits, images = _splits(tmp_path)
    train, val, _ = runner.load_existing_splits(splits, images)
    dataset = RegionDataset(val)
    sample = dataset[0]
    with Image.open(val.iloc[0].path) as source:
        expected = get_eval_transforms(crop_border=False)(source.convert("RGB"))
    assert torch.equal(sample["image"], expected)
    candidates = extract_candidates(tensor_to_rgb(expected), CandidateConfig())
    for name, values in candidates.items():
        assert np.array_equal(sample["candidates"][name].numpy(), values)
    assert sample["index"] == 0
    assert sample["label"] == 0


def test_augmentation_isolated_from_global_rng_and_epoch_dependent(tmp_path):
    splits, images = _splits(tmp_path)
    train, _, _ = runner.load_existing_splits(splits, images)
    dataset = RegionDataset(train, training=True, seed=42)
    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = dataset[0]
    assert random.getstate() == states[0]
    assert all(np.array_equal(x, y) for x, y in zip(np.random.get_state(), states[1]))
    assert torch.equal(torch.get_rng_state(), states[2])
    random.random()
    np.random.rand(32)
    torch.rand(100)
    assert torch.equal(first["image"], dataset[0]["image"])
    dataset.set_epoch(1)
    assert not torch.equal(first["image"], dataset[0]["image"])
    dataset.set_epoch(0)
    assert torch.equal(first["image"], dataset[0]["image"])


def test_worker_count_does_not_change_transformed_stream(tmp_path):
    splits, images = _splits(tmp_path)
    train, val, _ = runner.load_existing_splits(splits, images)
    streams = []
    for workers in (0, 2):
        args = _args(tmp_path, splits, images, num_workers=workers)
        loader, _ = runner.make_loaders(train, val, args, 42, torch.device("cpu"), CandidateConfig())
        loader.dataset.set_epoch(2)
        streams.append(list(loader))
    for left, right in zip(*streams):
        for name in ("image", "label", "index"):
            assert torch.equal(left[name], right[name])
        for name in left["candidates"]:
            assert torch.equal(left["candidates"][name], right["candidates"][name])


def test_four_arms_share_inputs_and_initialization_and_reload(tmp_path, monkeypatch):
    splits, images = _splits(tmp_path)
    monkeypatch.setattr(runner, "build_region_relation_resnet50", TinyRegionModel)
    args = _args(tmp_path, splits, images)
    completed = runner.run(args)
    seed_dir = completed[0]
    pair = json.loads((seed_dir / "pair_metadata.json").read_text())
    assert pair["completed"] and pair["initial_states_identical"] and pair["training_input_streams_identical"]
    assert pair["test_images_loaded"] is False
    assert set(pair["variants"]) == set(runner.ARMS)
    initial, traces = [], []
    for variant in runner.ARMS:
        arm_dir = seed_dir / variant
        config = json.loads((arm_dir / "run_config.json").read_text())
        assert config["completed"] and config["checkpoint_reload_verified"]
        assert config["batchnorm_buffers_unchanged"]
        assert config["completed_epochs"] == 2
        assert config["loss"] == "cross_entropy_only"
        assert config["crop_border"] is False
        assert config["test_images_loaded"] is False
        assert config["primary_checkpoint"] == "last_model.pt"
        assert len(pd.read_csv(arm_dir / "history.csv")) == 2
        assert config["checkpoint_sha256"]["last_model.pt"] == runner.sha256_file(arm_dir / "last_model.pt")
        assert pair["variants"][variant]["run_config_sha256"] == runner.sha256_file(arm_dir / "run_config.json")
        state = torch.load(arm_dir / "last_model.pt", weights_only=True)
        assert state["model_state"]["backbone.1.num_batches_tracked"].item() == 0
        assert state["variant"] == variant
        assert state["split_membership_sha256"]["val"] == config["splits"]["val"]["membership_sha256"]
        initial.append(config["initial_state_sha256"])
        traces.append(config["training_input_traces"])
    assert len(set(initial)) == 1
    assert all(trace == traces[0] for trace in traces)
    assert traces[0][0]["n_images"] == 3
    assert traces[0][0]["augmented_input_sha256"] != traces[0][1]["augmented_input_sha256"]
    before = (args.output_root / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="not overwritten"):
        runner.run(args)
    assert (args.output_root / "protocol.json").read_bytes() == before


def test_patient_overlap_rejected_before_output_created(tmp_path):
    splits, images = _splits(tmp_path)
    test = pd.read_csv(splits / "test.csv", dtype=str)
    test["patient_id"] = "001"
    test.to_csv(splits / "test.csv", index=False)
    args = _args(tmp_path, splits, images)
    with pytest.raises(ValueError, match="Patient overlap"):
        runner.run(args)
    assert not args.output_root.exists()


def test_reload_failure_does_not_mark_arm_or_seed_complete(tmp_path, monkeypatch):
    splits, images = _splits(tmp_path)
    monkeypatch.setattr(runner, "build_region_relation_resnet50", TinyRegionModel)
    real_load = torch.load
    def corrupt_reload(*args, **kwargs):
        checkpoint = real_load(*args, **kwargs)
        checkpoint["model_state"]["head.1.bias"].add_(1)
        return checkpoint
    monkeypatch.setattr(runner.torch, "load", corrupt_reload)
    args = _args(tmp_path, splits, images, arms=["baseline"], stage1_epochs=0)
    with pytest.raises(RuntimeError, match="reload changed"):
        runner.run(args)
    seed_dir = args.output_root / "seed_42"
    config = json.loads((seed_dir / "baseline" / "run_config.json").read_text())
    assert config["completed"] is False
    assert not (seed_dir / "pair_metadata.json").exists()


def test_main_dispatches_validation_evaluator(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda args: [tmp_path / "seed_42"])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: calls.append(command))
    runner.main(["--splits-dir", str(tmp_path / "splits"), "--image-root", str(tmp_path / "images")])
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("-m") + 1] == "scripts.evaluate_region_relations"
    assert command[command.index("--run-dir") + 1].endswith("seed_42")
    assert command[command.index("--output-dir") + 1].endswith("region_val")
    assert "test" not in command


@pytest.mark.parametrize("argv", [
    ["--seeds", "42", "42"], ["--stage2-epochs", "0"], ["--dropout", "nan"],
    ["--stage1-lr", "-1"], ["--arms", "regions", "regions"], ["--bootstrap", "-1"],
    ["--arms", "baseline"],
])
def test_invalid_configuration_rejected(argv):
    with pytest.raises(SystemExit):
        runner.parse_args(argv)
