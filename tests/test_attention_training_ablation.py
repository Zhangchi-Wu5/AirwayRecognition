"""Training controls: freeze persistence, matched execution, provenance and reload."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import run_attention_training_ablation as runner
from src import attention_training_control as control
from src.attention import SpatialAttention
from src.train import set_seed, train_two_stage
from tests.test_rotation_ablation_training import _splits


@pytest.fixture
def tiny_model(monkeypatch):
    """Keep the production policy, forward, stage transitions and state loader."""
    def small_base(self, num_classes=3, pretrained=True, dropout=.3, hires=False):
        nn.Module.__init__(self)
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(8, num_classes))
        self.hires = False
    monkeypatch.setattr(control.AnatomyAttentionResNet50, "__init__", small_base)
    return control.build_controlled_attention_resnet50


def test_identical_initial_states_logits_and_half_attention(tiny_model):
    set_seed(18)
    fixed = tiny_model("fixed", pretrained=False)
    set_seed(18)
    learned = tiny_model("learned", pretrained=False)
    assert runner.state_fingerprint(fixed) == runner.state_fingerprint(learned)
    images = torch.randn(2, 3, 16, 16)
    fixed.eval()
    learned.eval()
    logits_f, attn_f = fixed(images, return_attn=True)
    logits_l, attn_l = learned(images, return_attn=True)
    assert torch.equal(logits_f, logits_l)
    assert torch.equal(attn_f, torch.full_like(attn_f, .5))
    assert torch.equal(attn_f, attn_l)
    for stage in ("freeze_backbone", "unfreeze_all"):
        getattr(fixed, stage)()
        getattr(learned, stage)()
        assert not fixed.attention.conv.weight.requires_grad
        assert learned.attention.conv.weight.requires_grad


def test_two_stage_training_preserves_fixed_and_changes_learned(tiny_model, tmp_path):
    set_seed(123)
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 0])
    masks = torch.ones(4, 1, 16, 16)
    results = []
    for policy in ("fixed", "learned"):
        set_seed(42)
        model = tiny_model(policy, pretrained=False)
        train = DataLoader(TensorDataset(images, labels, masks), batch_size=2,
                           generator=torch.Generator().manual_seed(11))
        val = DataLoader(TensorDataset(images, labels), batch_size=2,
                         generator=torch.Generator().manual_seed(12))
        tracer = runner.ExecutionTracer(model, torch.device("cpu"))
        checks = []
        def epoch(info):
            tracer.finish_epoch(len(train), len(val))
            checks.append(model.assert_attention_policy())
        history = train_two_stage(
            model, train, val, device="cpu", stage1_epochs=1, stage2_epochs=1,
            checkpoint_path=tmp_path / f"{policy}.pt",
            reg={"lambda_eq": 0., "lambda_pf": 0., "max_angle": 15., "eq_auxiliary_forward": True},
            on_epoch_end=epoch, track_loss_components=True,
        )
        tracer.close()
        assert len(history["epoch"]) == 2
        assert model.backbone[1].num_batches_tracked.item() == 8
        results.append((model, tracer.epochs, checks))
    fixed, learned = results[0][0], results[1][0]
    assert torch.count_nonzero(fixed.attention.conv.weight).item() == 0
    assert torch.count_nonzero(learned.attention.conv.weight).item() > 0
    assert results[0][1] == results[1][1]
    assert all(check["count_nonzero"] == 0 and not check["requires_grad"] for check in results[0][2])
    reloaded = tiny_model("fixed", pretrained=False)
    reloaded.load_state_dict(torch.load(tmp_path / "fixed.pt", weights_only=True))
    reloaded.unfreeze_all()
    assert reloaded.assert_attention_policy()["requires_grad"] is False
    invalid = copy.deepcopy(reloaded.state_dict())
    invalid["attention.conv.weight"].fill_(1.)
    with pytest.raises(RuntimeError, match="Fixed attention"):
        reloaded.load_state_dict(invalid)


def test_tracer_rejects_missing_auxiliary_forward(tiny_model):
    model = tiny_model("fixed", pretrained=False)
    tracer = runner.ExecutionTracer(model, torch.device("cpu"))
    model(torch.zeros(2, 3, 16, 16))
    with pytest.raises(RuntimeError, match="Forward schedule"):
        tracer.finish_epoch(1, 0)
    tracer.close()


def test_loss_policies_do_not_attach_regularization_to_primary_arms():
    args = runner.parse_args(["--lambda-eq", ".7", "--lambda-pf", ".3", "--include-regularized"])
    for arm in runner.ARMS:
        reg = runner.regularization_for_arm(arm, args)
        assert reg == {"lambda_eq": 0., "lambda_pf": 0., "max_angle": 15., "eq_auxiliary_forward": True}
    bridge = runner.regularization_for_arm("learned_regularized", args)
    assert bridge["lambda_eq"] == .7 and bridge["lambda_pf"] == .3
    with pytest.raises(ValueError, match="Unknown arm"):
        runner.regularization_for_arm("fixed_regularized", args)


def test_runner_records_completed_matched_training_and_reload(tmp_path, tiny_model):
    splits, images = _splits(tmp_path)
    output = tmp_path / "controls"
    args = runner.parse_args([
        "--splits-dir", str(splits), "--image-root", str(images), "--output-root", str(output),
        "--device", "cpu", "--num-workers", "0", "--batch-size", "2",
        "--stage1-epochs", "1", "--stage2-epochs", "1", "--no-pretrained",
        "--include-regularized", "--skip-evaluation",
    ])
    completed = runner.run(args)
    assert completed == [output / "seed_42"]
    pair = json.loads((completed[0] / "pair_metadata.json").read_text())
    assert pair["protocol_version"] == "attention_training_control_v1"
    assert pair["initial_states_identical"]
    assert pair["training_input_streams_identical"]
    assert pair["execution_traces_identical"]
    assert pair["test_images_loaded"] is False
    assert pair["primary_contrast"] == "learned_ce_minus_fixed_ce"
    assert set(pair["arms"]) == {"fixed_ce", "learned_ce", "learned_regularized"}
    for arm, metadata in pair["arms"].items():
        folder = completed[0] / arm
        config = json.loads((folder / "run_config.json").read_text())
        execution = json.loads((folder / "execution_trace.json").read_text())
        assert config["completed"] and config["completed_epochs"] == 2
        assert config["checkpoint_reload_verified"]
        assert config["execution_traces"] == execution["epochs"]
        assert metadata["run_config_sha256"] == runner.sha256_file(folder / "run_config.json")
        assert metadata["checkpoint_sha256"] == runner.sha256_file(folder / "last_model.pt")
        assert metadata["execution_trace_sha256"] == runner.sha256_file(folder / "execution_trace.json")
        assert "src/attention_training_control.py" in config["source_sha256"]
        assert "scripts/run_attention_training_ablation.py" in config["source_sha256"]
        assert all(epoch["train_forward_count"] == 4 and epoch["eval_forward_count"] == 1
                   for epoch in execution["epochs"])
        assert config["attention_control"] == runner.attention_control(arm)
        for field in ("lambda_eq", "lambda_pf"):
            assert config["regularization"][field] == (.1 if arm == "learned_regularized" else 0.)
        if arm == "fixed_ce":
            assert all(item["count_nonzero"] == 0 and not item["requires_grad"]
                       for item in config["attention_parameter_checks"])
    before = (output / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="not overwritten"):
        runner.run(args)
    assert before == (output / "protocol.json").read_bytes()


def test_main_uses_new_evaluation_with_run_dir_and_validation_output(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda args: [tmp_path / "seed_42"])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: calls.append(command))
    runner.main(["--splits-dir", str(tmp_path / "splits"), "--image-root", str(tmp_path / "images")])
    assert len(calls) == 1
    command = calls[0]
    assert command[2] == "scripts.evaluate_attention_training_pairs"
    assert command[command.index("--run-dir") + 1] == str(tmp_path / "seed_42")
    assert command[command.index("--output-dir") + 1].endswith("/attention_training_val")
    assert "--split" not in command


@pytest.mark.parametrize("args", [
    ["--seeds", "42", "42"], ["--stage2-epochs", "0"], ["--lambda-pf", "0"],
    ["--lambda-eq", "nan"], ["--max-angle", "16"], ["--evaluation-seed", "-1"],
])
def test_invalid_protocol_arguments_rejected(args):
    with pytest.raises(SystemExit):
        runner.parse_args(args)
