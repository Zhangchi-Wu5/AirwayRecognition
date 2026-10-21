"""Fixed-budget, validation-only development trial of candidate-region relations.

Every arm uses the same ordinary ResNet image transforms and initialization.
This runner never evaluates test images and does not establish patent novelty.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

from scripts.evaluate_glare_pairs import sha256_file
from scripts.run_rotation_ablation import load_existing_splits, state_fingerprint
from src.region_candidates import CandidateConfig
from src.region_data import RegionDataset
from src.region_relations import build_region_relation_resnet50
from src.train import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "region_relation_v1"
ARMS = ("baseline", "regions", "relations", "no_geometry_relations")


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits_patent")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "region_relation_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", "--lr", dest="stage2_lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 1 for s in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32-2]")
    if len(set(args.arms)) != len(args.arms):
        parser.error("arms must be distinct")
    if set(args.arms) != set(ARMS) and not args.skip_evaluation:
        parser.error("automatic evaluation requires all four arms; add --skip-evaluation for a subset smoke test")
    if args.batch_size < 1 or args.num_workers < 0 or args.stage1_epochs < 0 or args.stage2_epochs < 1:
        parser.error("batch-size >= 1, num-workers/stage1-epochs >= 0 and stage2-epochs >= 1 required")
    if any(not np.isfinite(v) or v <= 0 for v in (args.stage1_lr, args.stage2_lr)):
        parser.error("learning rates must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0,1)")
    if args.bootstrap < 0 or not 0 <= args.evaluation_seed < 2**32:
        parser.error("bootstrap must be nonnegative and evaluation-seed must be in [0,2**32)")
    return args


def _source_metadata():
    files = (
        "src/region_candidates.py", "src/region_relations.py", "src/region_data.py",
        "src/data.py", "src/models.py", "src/train.py", "scripts/run_region_relation_ablation.py",
        "scripts/evaluate_region_relations.py", "scripts/run_rotation_ablation.py",
        "scripts/evaluate_glare_pairs.py",
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {
        "git_commit": commit, "git_worktree_dirty": dirty,
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in files},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": str(torchvision.__version__), "numpy": np.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
    }


def make_loaders(train, val, args, seed, device, candidate_config):
    train_dataset = RegionDataset(train, training=True, candidate_config=candidate_config, seed=seed)
    val_dataset = RegionDataset(val, training=False, candidate_config=candidate_config, seed=seed)
    common = {"batch_size": args.batch_size, "num_workers": args.num_workers,
              "pin_memory": device.type == "cuda", "persistent_workers": False}
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(val_dataset, shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **common)
    return train_loader, val_loader


def move_batch(batch, device):
    return (batch["image"].to(device), batch["label"].to(device),
            {key: value.to(device) for key, value in batch["candidates"].items()})


def _hash_tensor(digest, name, value):
    tensor = value.detach().cpu().contiguous()
    digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
    digest.update(tensor.numpy().tobytes())


def _epoch(model, loader, device, optimizer=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    digest = hashlib.sha256(f"epoch:{epoch}".encode())
    total_loss = total_correct = total_count = n_batches = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                digest.update(f"batch:{n_batches}".encode())
                for name in ("image", "label", "index"):
                    _hash_tensor(digest, name, batch[name])
                for name in sorted(batch["candidates"]):
                    _hash_tensor(digest, f"candidates.{name}", batch["candidates"][name])
                digest.update(json.dumps(batch["case_id"]).encode())
            images, labels, candidates = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, candidates)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite logits or cross-entropy loss")
            if training:
                loss.backward()
                optimizer.step()
            total_count += len(labels)
            total_loss += float(loss.detach()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            n_batches += 1
    if not total_count:
        raise ValueError("Training and validation loaders must be non-empty")
    trace = {"training_epoch": epoch, "n_batches": n_batches, "n_images": total_count,
             "augmented_input_sha256": digest.hexdigest()}
    return total_loss / total_count, total_correct / total_count, trace


def _bn_buffers(model):
    return {f"{name}.{key}": value.detach().cpu().clone()
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def _checkpoint(model, config, epoch):
    return {
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "variant": config["variant"], "model_config": config["model_config"],
        "candidate_config": config["candidate_config"], "protocol_version": PROTOCOL_VERSION,
        "seed": config["seed"], "epoch": epoch,
        "split_membership_sha256": {name: data["membership_sha256"] for name, data in config["splits"].items()},
    }


def train_arm(model, train_loader, val_loader, args, device, arm_dir, config):
    model.to(device)
    initial_bn = _bn_buffers(model)
    history, traces = [], []
    global_epoch = 0
    best_val_accuracy = -1.0
    for stage, epochs, lr in ((1, args.stage1_epochs, args.stage1_lr),
                              (2, args.stage2_epochs, args.stage2_lr)):
        if stage == 1:
            model.freeze_backbone()
        else:
            model.unfreeze_all()
        if not epochs:
            continue
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if stage == 2 else None
        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_loader.dataset.set_epoch(global_epoch)
            train_loss, train_acc, trace = _epoch(model, train_loader, device, optimizer, global_epoch)
            val_loss, val_acc, _ = _epoch(model, val_loader, device)
            record = {"stage": stage, "epoch": stage_epoch, "training_epoch": global_epoch,
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "train_loss": train_loss, "train_acc": train_acc,
                      "val_loss": val_loss, "val_acc": val_acc}
            history.append(record)
            traces.append(trace)
            if scheduler is not None:
                scheduler.step()
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(_checkpoint(model, config, global_epoch), arm_dir / "best_val.pt")
            print(f"seed={config['seed']} {config['variant']} stage={stage} epoch={stage_epoch} "
                  f"train_loss={train_loss:.6f} val_acc={val_acc:.4f}", flush=True)
            pd.DataFrame(history).to_csv(arm_dir / "history.csv", index=False)
    final_bn = _bn_buffers(model)
    if initial_bn.keys() != final_bn.keys() or any(not torch.equal(value, final_bn[name]) for name, value in initial_bn.items()):
        raise RuntimeError("Frozen BatchNorm running statistics changed during training")
    torch.save(_checkpoint(model, config, global_epoch), arm_dir / "last_model.pt")
    pd.DataFrame(traces).to_csv(arm_dir / "training_input_trace.csv", index=False)
    # Verify portable checkpoint loading and predictions before completion flags.
    checkpoint = torch.load(arm_dir / "last_model.pt", map_location="cpu", weights_only=True)
    model_config = {**config["model_config"], "pretrained": False}
    reloaded = build_region_relation_resnet50(**model_config)
    reloaded.load_state_dict(checkpoint["model_state"], strict=True)
    final_hash = state_fingerprint(model)
    if state_fingerprint(reloaded) != final_hash:
        raise RuntimeError("Checkpoint reload changed model tensors")
    reloaded.to(device).eval()
    model.eval()
    probe = move_batch(next(iter(val_loader)), device)
    with torch.no_grad():
        expected = model(probe[0], probe[2])
        actual = reloaded(probe[0], probe[2])
    if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Checkpoint reload predictions differ")
    del reloaded
    return history, traces, final_hash, best_val_accuracy


def run(args):
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError("output-root must be absent or empty; existing runs are not overwritten")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the server or explicitly --device cpu for a smoke test")
        torch.cuda.get_device_properties(device)
    train, val, splits = load_existing_splits(args.splits_dir, args.image_root)
    candidate_config = CandidateConfig()
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "arguments": arguments, "splits": splits,
        "candidate_config": asdict(candidate_config), "crop_border": False,
        "image_transforms": "src.data ordinary train/eval transforms; candidates extracted after augmentation",
        "field_of_view_limit": "Existing random/center crop is retained; candidates may be truncated",
        "batchnorm_policy": "running statistics frozen in both stages; affine parameters trainable in stage 2",
        "loss": "cross_entropy_only", "pf_loss": False, "eq_loss": False,
        "rng_policy": "isolated per(seed,epoch,index) CPU augmentation RNG; identical loader seed per arm",
        "training_budget": "fixed stage1+stage2 epochs; no early stopping",
        "primary_checkpoint": "last_model.pt", "secondary_checkpoint": "best_val.pt",
        "evaluation_split": "val", "test_images_loaded": False,
        "interpretation": "Exploratory internal development; candidates are not verified anatomical segmentations",
        **_source_metadata(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "protocol.json", protocol)
    completed = []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        expected_initial = expected_traces = None
        variants = {}
        for variant in args.arms:
            set_seed(seed)
            model_config = {"variant": variant, "pretrained": not args.no_pretrained,
                            "num_classes": 3, "dropout": args.dropout,
                            "max_candidates": candidate_config.max_candidates}
            model = build_region_relation_resnet50(**model_config)
            initial_hash = state_fingerprint(model)
            if expected_initial is None:
                expected_initial = initial_hash
            elif initial_hash != expected_initial:
                raise RuntimeError("Arms do not share identical initialized parameters and buffers")
            train_loader, val_loader = make_loaders(train, val, args, seed, device, candidate_config)
            arm_dir = seed_dir / variant
            arm_dir.mkdir()
            config = {**protocol, "variant": variant, "seed": seed, "model_config": model_config,
                      "initial_state_sha256": initial_hash, "completed": False,
                      "started_at_utc": datetime.now(timezone.utc).isoformat(), "resolved_device": str(device)}
            _write_json(arm_dir / "run_config.json", config)
            history, traces, final_hash, best = train_arm(model, train_loader, val_loader, args, device, arm_dir, config)
            if len(history) != args.stage1_epochs + args.stage2_epochs:
                raise RuntimeError("Fixed training budget was not completed")
            if expected_traces is None:
                expected_traces = traces
            elif traces != expected_traces:
                raise RuntimeError("Arms received different transformed training images/candidates/order")
            config.update(
                completed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                completed_epochs=len(history), final_state_sha256=final_hash,
                best_val_accuracy=best, last_val_accuracy=history[-1]["val_acc"],
                training_input_traces=traces,
                training_input_trace_sha256=sha256_file(arm_dir / "training_input_trace.csv"),
                checkpoint_sha256={name: sha256_file(arm_dir / name) for name in ("last_model.pt", "best_val.pt")},
                checkpoint_reload_verified=True, batchnorm_buffers_unchanged=True,
            )
            _write_json(arm_dir / "run_config.json", config)
            variants[variant] = {
                "initial_state_sha256": initial_hash, "checkpoint_sha256": config["checkpoint_sha256"]["last_model.pt"],
                "run_config_sha256": sha256_file(arm_dir / "run_config.json"),
                "training_input_trace_sha256": config["training_input_trace_sha256"],
            }
            del model, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _write_json(seed_dir / "pair_metadata.json", {
            "completed": True, "protocol_version": PROTOCOL_VERSION, "seed": seed,
            "primary_checkpoint": "last_model.pt", "evaluation_split": "val", "test_images_loaded": False,
            "initial_states_identical": True, "training_input_streams_identical": True, "variants": variants,
        })
        completed.append(seed_dir)
        print(f"Completed fixed-budget candidate-region trial: {seed_dir}", flush=True)
    return completed


def main(argv=None):
    args = parse_args(argv)
    completed = run(args)
    if not args.skip_evaluation:
        for seed_dir in completed:
            command = [sys.executable, "-m", "scripts.evaluate_region_relations", "--run-dir", str(seed_dir),
                       "--splits-dir", str(args.splits_dir), "--output-dir", str(seed_dir / "region_val"),
                       "--device", args.device, "--batch-size", str(args.batch_size),
                       "--bootstrap", str(args.bootstrap), "--seed", str(args.evaluation_seed)]
            if args.image_root is not None:
                command += ["--image-root", str(args.image_root)]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
