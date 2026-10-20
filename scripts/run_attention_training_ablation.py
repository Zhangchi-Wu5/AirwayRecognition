"""Matched training-time learned versus fixed spatial attention, on existing splits.

Primary: learned_ce minus fixed_ce, with EQ=PF=0 in both arms. An optional third
arm adds BOTH EQ and PF; its difference is a joint-regularization contrast.
All arms retain the rotated auxiliary forward to match BN/RNG execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from scripts.run_rotation_ablation import (
    PROJECT_ROOT, _source_metadata as _legacy_source_metadata, _write_json,
    load_existing_splits, make_loaders, sha256_file, state_fingerprint,
)
import numpy as np
import torch

from src.attention_training_control import build_controlled_attention_resnet50
from src.train import set_seed, train_two_stage

PROTOCOL_VERSION = "attention_training_control_v1"
ARMS = ("fixed_ce", "learned_ce")
REGULARIZED_ARM = "learned_regularized"
ATTENTION_POLICIES = {
    "fixed_ce": "fixed_constant_half", "learned_ce": "learned_zero_initialized",
    "learned_regularized": "learned_zero_initialized",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits_patent")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "attention_training_ablation_runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--include-regularized", action="store_true")
    parser.add_argument("--lambda-pf", type=float, default=.1, help="Optional third arm only")
    parser.add_argument("--lambda-eq", type=float, default=.1, help="Optional third arm only")
    parser.add_argument("--max-angle", type=float, default=15.)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 1 for s in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32-2]")
    if args.batch_size < 2 or args.num_workers < 0 or args.stage1_epochs < 0 or args.stage2_epochs < 1:
        parser.error("batch-size >= 2, num-workers/stage1-epochs >= 0 and stage2-epochs >= 1 required")
    if any(not np.isfinite(value) or value <= 0 for value in
           (args.stage1_lr, args.stage2_lr, args.lambda_eq, args.lambda_pf, args.max_angle)):
        parser.error("learning rates, lambda-eq, lambda-pf and max-angle must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1 or args.max_angle > 15:
        parser.error("dropout must be in [0,1); max-angle must not exceed 15")
    if args.bootstrap < 0 or not 0 <= args.evaluation_seed < 2**32:
        parser.error("bootstrap must be nonnegative and evaluation-seed in [0, 2**32-1]")
    return args


def attention_control(arm: str) -> dict:
    if arm not in (*ARMS, REGULARIZED_ARM):
        raise ValueError(f"Unknown arm {arm}")
    return {
        "policy": "fixed" if arm == "fixed_ce" else "learned",
        "initialization": "zero_spatial_convolution", "initial_constant": .5,
        "trainable": arm != "fixed_ce", "kernel_size": 3, "hires": False,
    }


def regularization_for_arm(arm: str, args) -> dict:
    attention_control(arm)  # validate the arm before applying its policy
    return {
        "lambda_eq": args.lambda_eq if arm == REGULARIZED_ARM else 0.,
        "lambda_pf": args.lambda_pf if arm == REGULARIZED_ARM else 0.,
        "max_angle": args.max_angle, "eq_auxiliary_forward": True,
    }


def _rng_fingerprint(device: torch.device) -> str:
    """Read, without advancing, Python/NumPy/CPU/requested-device RNG states."""
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("ascii"))
    np_state = np.random.get_state()
    digest.update(np_state[0].encode("ascii"))
    digest.update(np_state[1].tobytes())
    digest.update(repr(np_state[2:]).encode("ascii"))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if device.type == "cuda":
        digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


class ExecutionTracer:
    """Read-only forward/RNG/BN trace; never compares BN running means/variances."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model, self.device = model, device
        self.epochs = []
        self.bn_modules = {name: module for name, module in model.named_modules()
                           if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        self.handles = [model.register_forward_pre_hook(self._before),
                        model.register_forward_hook(self._after)]
        for name, module in self.bn_modules.items():
            self.handles.append(module.register_forward_pre_hook(self._bn_hook(name)))
        self._reset()

    def _reset(self):
        self.digest = hashlib.sha256()
        self.train_count = self.eval_count = 0
        self.bn_counts = {name: {"train": 0, "eval": 0} for name in self.bn_modules}

    def _before(self, module, inputs):
        mode = "train" if module.training else "eval"
        self.digest.update(f"{mode}:before:{_rng_fingerprint(self.device)}".encode("ascii"))
        if module.training:
            self.train_count += 1
        else:
            self.eval_count += 1

    def _after(self, module, inputs, output):
        self.digest.update(f"after:{_rng_fingerprint(self.device)}".encode("ascii"))

    def _bn_hook(self, name):
        def count(module, inputs):
            self.bn_counts[name]["train" if module.training else "eval"] += 1
        return count

    def finish_epoch(self, train_batches: int, val_batches: int) -> dict:
        if self.train_count != 2 * train_batches or self.eval_count != val_batches:
            raise RuntimeError("Forward schedule differs from two train forwards and one validation forward per batch")
        expected_bn = {"train": 2 * train_batches, "eval": val_batches}
        if any(counts != expected_bn for counts in self.bn_counts.values()):
            raise RuntimeError("BatchNorm forward counts do not match the prescribed schedule")
        entry = {
            "training_epoch": len(self.epochs) + 1,
            "model_forward_count": self.train_count + self.eval_count,
            "train_forward_count": self.train_count, "eval_forward_count": self.eval_count,
            "forward_rng_sha256": self.digest.hexdigest(),
            "end_rng_sha256": _rng_fingerprint(self.device),
            "bn_forward_counts": self.bn_counts,
            "bn_num_batches_tracked": {name: int(module.num_batches_tracked.item())
                                       for name, module in self.bn_modules.items()
                                       if module.num_batches_tracked is not None},
        }
        self.epochs.append(entry)
        self._reset()
        return entry

    def close(self):
        for handle in self.handles:
            handle.remove()


def _source_metadata() -> dict:
    metadata = _legacy_source_metadata()
    for name in (
        "src/attention_training_control.py", "scripts/run_attention_training_ablation.py",
        "scripts/evaluate_attention_training_pairs.py",
    ):
        path = PROJECT_ROOT / name
        if path.is_file():
            metadata["source_sha256"][name] = sha256_file(path)
    return metadata


def run(args) -> list[Path]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError("output-root must be absent or empty; existing runs are not overwritten.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the server environment or --device cpu for a smoke test")
        torch.cuda.get_device_properties(device)
    train, val, splits = load_existing_splits(args.splits_dir, args.image_root)
    if len(train) < 2:
        raise ValueError("At least two training images are required")
    arms = (*ARMS, REGULARIZED_ARM) if args.include_regularized else ARMS
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": list(arms), "splits": splits,
        "fixed_architecture": "AnatomyAttentionResNet50_layer4_7x7",
        "crop_border": True, "pf_mask": "specular_highlight", "mask_size": 56,
        "attention_initialization": "zero_spatial_convolution_A_equals_0.5_in_all_arms",
        "eq_auxiliary_forward": True,
        "batchnorm_policy": "train mode including frozen-backbone stage; same forward counts, not identical learned statistics",
        "rng_policy": "same seed, initial tensors, loader streams and auxiliary schedule; forward RNG states traced",
        "training_budget": "fixed epochs; early stopping disabled",
        "primary_checkpoint": "last_model.pt", "secondary_checkpoint": "best_val.pt",
        "primary_contrast": "learned_ce_minus_fixed_ce",
        "secondary_contrast": "learned_regularized_minus_learned_ce" if args.include_regularized else None,
        "evaluation_split": "val", "test_images_loaded": False,
        "interpretation": (
            "Training-time attention learnability (spatial pattern and amplitude together), conditional on "
            "identical two-branch head, constant-half initialization and matched auxiliary forwards. "
            "The optional third arm tests joint EQ+PF regularization, not EQ alone. "
            "The fixed arm is not the original plain ResNet baseline."
        ),
        **_source_metadata(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "protocol.json", protocol)
    completed_seeds = []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        expected_initial = expected_inputs = expected_execution = None
        arm_metadata = {}
        for arm in arms:
            arm_dir = seed_dir / arm
            arm_dir.mkdir()
            set_seed(seed)
            control = attention_control(arm)
            model = build_controlled_attention_resnet50(
                attention_policy=control["policy"], pretrained=not args.no_pretrained, dropout=args.dropout,
            )
            model.assert_attention_policy()
            initial_hash = state_fingerprint(model)
            if expected_initial is None:
                expected_initial = initial_hash
            elif expected_initial != initial_hash:
                raise RuntimeError("Arms do not have identical initial parameters and buffers")
            train_loader, val_loader = make_loaders(train, val, args, seed, device)
            reg = regularization_for_arm(arm, args)
            config = {
                **protocol, "seed": seed, "arm": arm, "regularization": reg,
                "attention_policy": ATTENTION_POLICIES[arm], "attention_control": control,
                "initial_state_sha256": initial_hash, "completed": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "resolved_device": str(device),
            }
            _write_json(arm_dir / "run_config.json", config)
            tracer = ExecutionTracer(model, device)
            attention_checks = []

            def log_epoch(info):
                if not all(np.isfinite(info[name]) for name in ("train_loss", "train_acc", "val_loss", "val_acc")):
                    raise FloatingPointError(f"Non-finite metric for seed={seed} arm={arm}")
                trace = tracer.finish_epoch(len(train_loader), len(val_loader))
                attention_checks.append({"training_epoch": trace["training_epoch"], **model.assert_attention_policy()})
                print(f"seed={seed} {arm} stage={info['stage']} epoch={info['epoch']} "
                      f"train_loss={info['train_loss']:.6f} val_acc={info['val_acc']:.4f}", flush=True)

            print(f"Training seed={seed} arm={arm}; attention={control['policy']}; "
                  f"EQ={reg['lambda_eq']} PF={reg['lambda_pf']}; matched auxiliary forward", flush=True)
            try:
                history = train_two_stage(
                    model, train_loader, val_loader, device=str(device),
                    stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                    stage1_lr=args.stage1_lr, stage2_lr=args.stage2_lr, weight_decay=args.weight_decay,
                    early_stopping_patience=args.stage2_epochs + 1,
                    checkpoint_path=arm_dir / "best_val.pt", on_epoch_end=log_epoch,
                    reg=reg, track_loss_components=True,
                )
            finally:
                tracer.close()
            expected_epochs = args.stage1_epochs + args.stage2_epochs
            if len(history["epoch"]) != expected_epochs or len(tracer.epochs) != expected_epochs:
                raise RuntimeError("Fixed epoch budget was not completed")
            traces = train_loader.epoch_traces
            if len(traces) != expected_epochs:
                raise RuntimeError("Input trace does not match the epoch budget")
            import pandas as pd
            pd.DataFrame(traces).to_csv(arm_dir / "training_input_trace.csv", index=False)
            execution = {"schema_version": 1, "epochs": tracer.epochs,
                         "policy": "read-only per-forward RNG states and BN call counts; no running-value equality claim"}
            _write_json(arm_dir / "execution_trace.json", execution)
            if expected_inputs is None:
                expected_inputs, expected_execution = traces, tracer.epochs
            elif traces != expected_inputs or tracer.epochs != expected_execution:
                raise RuntimeError("Arms received different augmented inputs or RNG/BN forward schedules; inspect traces")
            model.assert_attention_policy()
            final_hash = state_fingerprint(model)
            torch.save(model.state_dict(), arm_dir / "last_model.pt")
            pd.DataFrame({key: value for key, value in history.items() if isinstance(value, list)}).to_csv(
                arm_dir / "training_history.csv", index=False,
            )
            # Rebuild under the declared policy; requires_grad is not in state_dict.
            reloaded = build_controlled_attention_resnet50(
                attention_policy=control["policy"], pretrained=False, dropout=args.dropout,
            )
            reloaded.load_state_dict(torch.load(arm_dir / "last_model.pt", map_location="cpu", weights_only=True))
            if state_fingerprint(reloaded) != final_hash:
                raise RuntimeError("Saved checkpoint failed policy-preserving reload verification")
            reloaded.assert_attention_policy()
            del reloaded
            config.update(
                completed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                completed_epochs=expected_epochs, best_val_accuracy=float(history["best_val_acc"]),
                last_val_accuracy=float(history["val_acc"][-1]), final_state_sha256=final_hash,
                training_input_trace_sha256=sha256_file(arm_dir / "training_input_trace.csv"),
                training_input_traces=traces,
                execution_trace_sha256=sha256_file(arm_dir / "execution_trace.json"),
                execution_traces=tracer.epochs, attention_parameter_checks=attention_checks,
                checkpoint_reload_verified=True,
                checkpoint_sha256={name: sha256_file(arm_dir / name) for name in ("last_model.pt", "best_val.pt")},
            )
            _write_json(arm_dir / "run_config.json", config)
            arm_metadata[arm] = {
                "attention_policy": ATTENTION_POLICIES[arm], "attention_control": control,
                "initial_state_sha256": initial_hash, "checkpoint_path": str(arm_dir / "last_model.pt"),
                "checkpoint_sha256": config["checkpoint_sha256"]["last_model.pt"],
                "run_config_sha256": sha256_file(arm_dir / "run_config.json"),
                "training_input_trace_sha256": config["training_input_trace_sha256"],
                "execution_trace_sha256": config["execution_trace_sha256"],
                "lambda_eq": reg["lambda_eq"], "lambda_pf": reg["lambda_pf"],
            }
            del model, tracer, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _write_json(seed_dir / "pair_metadata.json", {
            "completed": True, "protocol_version": PROTOCOL_VERSION, "seed": seed,
            "initial_states_identical": len({item["initial_state_sha256"] for item in arm_metadata.values()}) == 1,
            "training_input_streams_identical": len({item["training_input_trace_sha256"] for item in arm_metadata.values()}) == 1,
            "execution_traces_identical": len({item["execution_trace_sha256"] for item in arm_metadata.values()}) == 1,
            "primary_checkpoint": "last_model.pt", "primary_contrast": "learned_ce_minus_fixed_ce",
            "arms": arm_metadata, "test_images_loaded": False,
        })
        completed_seeds.append(seed_dir)
        print(f"Completed attention training controls: {seed_dir}", flush=True)
    return completed_seeds


def main(argv=None):
    args = parse_args(argv)
    completed = run(args)
    if not args.skip_evaluation:
        for seed_dir in completed:
            command = [
                sys.executable, "-m", "scripts.evaluate_attention_training_pairs",
                "--run-dir", str(seed_dir), "--splits-dir", str(args.splits_dir),
                "--output-dir", str(seed_dir / "attention_training_val"),
                "--device", args.device, "--batch-size", str(args.batch_size),
                "--bootstrap", str(args.bootstrap), "--seed", str(args.evaluation_seed),
            ]
            if args.image_root is not None:
                command.extend(["--image-root", str(args.image_root)])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
