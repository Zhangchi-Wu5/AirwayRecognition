"""Controlled EQ-loss ablation on fixed patient splits; never infer on test images.

Both arms execute the auxiliary rotated-image forward. Only its loss coefficient
changes: eq_off=0 (sham) versus eq_on=lambda_eq. The primary checkpoint is the
last epoch of a fixed training budget, not the best validation checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision

from scripts.evaluate_glare_pairs import sha256_file, validate_manifest
from src.attention import build_attention_resnet50
from src.data import BronchoscopyDataset, LABEL_TO_ID, get_eval_transforms, get_train_transforms
from src.train import set_seed, train_two_stage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "eq_sham_control_v1"
ARMS = ("eq_off", "eq_on")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def state_fingerprint(model: torch.nn.Module) -> str:
    """Hash tensor contents, independent of torch.save archive names/metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _identity_frame(path: Path) -> pd.DataFrame:
    """Read CSV identities only, including test; do not access image contents."""
    frame = pd.read_csv(path, dtype=str)
    required = {"patient_id", "label", "label_id", "path"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain non-empty patient_id,label,label_id,path columns")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing identities or labels")
    for column in required:
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{path} contains empty {column}")
    expected = frame.label.map(LABEL_TO_ID)
    if expected.isna().any() or not pd.to_numeric(frame.label_id, errors="coerce").eq(expected).all():
        raise ValueError(f"{path} has inconsistent labels")
    frame["label_id"] = expected.astype(int)
    frame["image_name"] = frame.path.str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    if frame.path.duplicated().any() or frame.duplicated(["patient_id", "label", "image_name"]).any():
        raise ValueError(f"{path} contains duplicate image identities")
    return frame


def load_existing_splits(splits_dir: Path, image_root: Path | None = None):
    """Validate all patient identities, resolving images only for train and val."""
    frames = {name: _identity_frame(splits_dir / f"{name}.csv") for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(frames[left].patient_id) & set(frames[right].patient_id):
            raise ValueError(f"Patient overlap between {left} and {right}")
        if set(frames[left].path) & set(frames[right].path):
            raise ValueError(f"Image path overlap between {left} and {right}")
    split_metadata = {}
    for name, frame in frames.items():
        members = frame[["patient_id", "label", "label_id", "image_name"]].sort_values(
            ["patient_id", "label", "image_name"]
        ).to_dict("records")
        split_metadata[name] = {
            "csv_sha256": sha256_file(splits_dir / f"{name}.csv"),
            "membership_sha256": hashlib.sha256(
                json.dumps(members, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "n_images": len(frame),
            "n_patients": int(frame.patient_id.nunique()),
            "class_counts": {key: int(value) for key, value in frame.label.value_counts().items()},
            "members": members,
            "image_access": "paths_resolved_images_not_opened" if name == "test" else "training_or_validation",
        }
    train = validate_manifest(frames["train"], image_root)
    val = validate_manifest(frames["val"], image_root)
    # Legacy absolute paths can differ while --image-root maps them to the same
    # file. Check actual mapping, including test identities, without opening or
    # checking existence of any test image.
    mapped_paths = {"train": set(train.path), "val": set(val.path)}
    mapped_paths["test"] = {
        str((Path(image_root) / row.image_name if image_root is not None else Path(row.path)).expanduser().resolve())
        for row in frames["test"].itertuples(index=False)
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if mapped_paths[left] & mapped_paths[right]:
            raise ValueError(f"Resolved image path overlap between {left} and {right}")
    return train, val, split_metadata


class TracedLoader:
    """Hash every transformed input, label and mask without retaining images."""

    def __init__(self, loader):
        self.loader = loader
        self.epoch_traces = []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        digest = hashlib.sha256()
        n_batches = n_images = 0
        for batch in self.loader:
            digest.update(str(n_batches).encode("ascii"))
            for value in batch:
                tensor = value.detach().cpu().contiguous()
                digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            n_batches += 1
            n_images += len(batch[0])
            yield batch
        self.epoch_traces.append({
            "training_epoch": len(self.epoch_traces) + 1, "n_batches": n_batches,
            "n_images": n_images, "augmented_input_sha256": digest.hexdigest(),
        })


def make_loaders(train: pd.DataFrame, val: pd.DataFrame, args, seed: int, device: torch.device):
    train_dataset = BronchoscopyDataset(
        train, transform=get_train_transforms(crop_border=True),
        return_pseudo_mask=True, mask_size=56, mask_kind="specular_highlight",
    )
    val_dataset = BronchoscopyDataset(val, transform=get_eval_transforms(crop_border=True))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker, generator=generator,
    )
    # Dedicated validation generator avoids incidental use of the model RNG.
    val_generator = torch.Generator().manual_seed(seed + 1)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker, generator=val_generator,
    )
    return TracedLoader(train_loader), val_loader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits_patent")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "rotation_ablation_runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-pf", type=float, default=.1)
    parser.add_argument("--lambda-eq", type=float, default=.1)
    parser.add_argument("--max-angle", type=float, default=15.)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true", help="Train only; useful for an offline smoke test")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--evaluation-seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 1 for s in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32-2]")
    if args.batch_size < 2 or args.num_workers < 0 or args.stage1_epochs < 0 or args.stage2_epochs < 1:
        parser.error("batch-size >= 2, num-workers/stage1-epochs >= 0 and stage2-epochs >= 1 required")
    if any(not np.isfinite(v) or v <= 0 for v in
           (args.stage1_lr, args.stage2_lr, args.lambda_eq, args.max_angle)):
        parser.error("learning rates, lambda-eq and max-angle must be finite and positive")
    if any(not np.isfinite(v) or v < 0 for v in (args.weight_decay, args.lambda_pf)):
        parser.error("weight-decay and lambda-pf must be finite and nonnegative")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1 or args.max_angle > 15:
        parser.error("dropout must be in [0,1); max-angle must not exceed the small-angle protocol limit 15")
    if args.bootstrap < 0 or args.evaluation_seed < 0:
        parser.error("bootstrap and evaluation-seed must be nonnegative")
    return args


def _source_metadata() -> dict:
    files = (
        "src/train.py", "src/regularization.py", "src/attention.py", "src/data.py",
        "src/viz.py", "scripts/run_rotation_ablation.py", "scripts/evaluate_glare_pairs.py",
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {
        "git_commit": commit, "git_worktree_dirty": dirty,
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in files},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "numpy": np.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
    }


def run(args) -> list[Path]:
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError("output-root must be absent or empty. Use a new path; existing runs are not overwritten.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; activate the server environment or explicitly use --device cpu for a smoke test")
        # Resolve the requested GPU now, before creating output or downloading weights.
        torch.cuda.get_device_properties(device)
    train, val, splits = load_existing_splits(args.splits_dir, args.image_root)
    if len(train) < 2:
        raise ValueError("At least two training images are required")
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    source = _source_metadata()
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "arguments": arguments, "splits": splits,
        "fixed_architecture": "AnatomyAttentionResNet50_layer4_7x7",
        "crop_border": True, "pf_mask": "specular_highlight", "mask_size": 56,
        "eq_auxiliary_forward": True,
        "batchnorm_policy": "train mode in both arms, including frozen-backbone stage; identical forward counts",
        "rng_policy": "same seed, initial model, loader generators and auxiliary forward schedule per pair",
        "training_budget": "fixed epochs; early stopping disabled by patience greater than stage2_epochs",
        "primary_checkpoint": "last_model.pt", "secondary_checkpoint": "best_val.pt",
        "evaluation_split": "val", "test_images_loaded": False,
        "interpretation": "EQ loss contribution conditional on cropped attention+PF and matched auxiliary forward; not a reproduction of old checkpoints",
        **source,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "protocol.json", protocol)
    completed_seeds = []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        expected_initial = None
        expected_input_traces = None
        arm_metadata = {}
        for arm in ARMS:
            arm_dir = seed_dir / arm
            arm_dir.mkdir()
            set_seed(seed)
            model = build_attention_resnet50(pretrained=not args.no_pretrained, dropout=args.dropout, hires=False)
            initial_hash = state_fingerprint(model)
            if expected_initial is None:
                expected_initial = initial_hash
            elif initial_hash != expected_initial:
                raise RuntimeError("Arms do not have identical initialized parameters and buffers")
            train_loader, val_loader = make_loaders(train, val, args, seed, device)
            eq_weight = 0. if arm == "eq_off" else args.lambda_eq
            regularization = {"lambda_eq": eq_weight, "lambda_pf": args.lambda_pf,
                              "max_angle": args.max_angle, "eq_auxiliary_forward": True}
            config = {
                **protocol, "seed": seed, "arm": arm, "regularization": regularization,
                "initial_state_sha256": initial_hash, "completed": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "resolved_device": str(device),
            }
            _write_json(arm_dir / "run_config.json", config)
            print(f"Training seed={seed} arm={arm} EQ={eq_weight}; matched auxiliary forward, fixed budget", flush=True)

            def log_epoch(info):
                if not all(np.isfinite(info[name]) for name in ("train_loss", "train_acc", "val_loss", "val_acc")):
                    raise FloatingPointError(f"Non-finite training/validation metric for seed={seed} arm={arm}")
                print(f"seed={seed} {arm} stage={info['stage']} epoch={info['epoch']} "
                      f"train_loss={info['train_loss']:.6f} val_acc={info['val_acc']:.4f}", flush=True)

            history = train_two_stage(
                model, train_loader, val_loader, device=str(device),
                stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                stage1_lr=args.stage1_lr, stage2_lr=args.stage2_lr, weight_decay=args.weight_decay,
                early_stopping_patience=args.stage2_epochs + 1,
                checkpoint_path=arm_dir / "best_val.pt", on_epoch_end=log_epoch, reg=regularization,
                track_loss_components=True,
            )
            expected_epochs = args.stage1_epochs + args.stage2_epochs
            if len(history["epoch"]) != expected_epochs:
                raise RuntimeError("Fixed epoch budget was not completed")
            traces = train_loader.epoch_traces
            pd.DataFrame(traces).to_csv(arm_dir / "training_input_trace.csv", index=False)
            if len(traces) != expected_epochs:
                raise RuntimeError("Training input trace does not match the fixed epoch budget")
            if expected_input_traces is None:
                expected_input_traces = traces
            elif traces != expected_input_traces:
                raise RuntimeError("Paired arms received different input/augmentation streams; inspect training_input_trace.csv")
            torch.save(model.state_dict(), arm_dir / "last_model.pt")
            pd.DataFrame({key: value for key, value in history.items() if isinstance(value, list)}).to_csv(
                arm_dir / "training_history.csv", index=False
            )
            config.update(
                completed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                completed_epochs=expected_epochs, best_val_accuracy=float(history["best_val_acc"]),
                last_val_accuracy=float(history["val_acc"][-1]),
                final_state_sha256=state_fingerprint(model),
                training_input_trace_sha256=sha256_file(arm_dir / "training_input_trace.csv"),
                training_input_traces=traces,
                checkpoint_sha256={name: sha256_file(arm_dir / name) for name in ("last_model.pt", "best_val.pt")},
            )
            _write_json(arm_dir / "run_config.json", config)
            arm_metadata[arm] = {
                "initial_state_sha256": initial_hash,
                "checkpoint_path": str(arm_dir / "last_model.pt"),
                "checkpoint_sha256": config["checkpoint_sha256"]["last_model.pt"],
                "run_config_sha256": sha256_file(arm_dir / "run_config.json"),
                "training_input_trace_sha256": config["training_input_trace_sha256"],
                "lambda_eq": eq_weight,
            }
            del model, train_loader, val_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _write_json(seed_dir / "pair_metadata.json", {
            "completed": True, "protocol_version": PROTOCOL_VERSION, "seed": seed,
            "initial_states_identical": len({item["initial_state_sha256"] for item in arm_metadata.values()}) == 1,
            "training_input_streams_identical": len({item["training_input_trace_sha256"] for item in arm_metadata.values()}) == 1,
            "primary_checkpoint": "last_model.pt", "arms": arm_metadata,
            "test_images_loaded": False,
        })
        completed_seeds.append(seed_dir)
        print(f"Completed training pair: {seed_dir}", flush=True)
    return completed_seeds


def main(argv=None):
    args = parse_args(argv)
    completed_seeds = run(args)
    if not args.skip_evaluation:
        for seed_dir in completed_seeds:
            command = [
                sys.executable, "-m", "scripts.evaluate_rotation_pairs",
                "--eq-off-checkpoint", str(seed_dir / "eq_off" / "last_model.pt"),
                "--eq-on-checkpoint", str(seed_dir / "eq_on" / "last_model.pt"),
                "--splits-dir", str(args.splits_dir), "--split", "val",
                "--output-dir", str(seed_dir / "rotation_val"),
                "--device", args.device, "--batch-size", str(args.batch_size),
                "--angles", "-15", "-10", "-5", "0", "5", "10", "15",
                "--bootstrap", str(args.bootstrap), "--seed", str(args.evaluation_seed),
                "--dropout", str(args.dropout),
            ]
            if args.image_root is not None:
                command.extend(["--image-root", str(args.image_root)])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
