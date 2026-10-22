"""Reproducible fine-tuning of the existing frozen-probe support sets.

Preparation is standard-library-only. Only train-support and validation images
are passed to loaders; the test CSV is used solely for split identity checks.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from src.fewshot import (LABEL_TO_ID, METRIC_NAMES, PROJECT_ROOT, _environment,
                        classification_metrics, load_splits, make_support_plan,
                        sha256_file, state_fingerprint, summarize_runs, write_csv)

VERSION = "resnet50_fixed_support_finetune_v1"
SOURCE_FILES = ("src/fewshot.py", "src/fewshot_finetune.py", "src/data.py", "src/models.py",
                "scripts/run_resnet_fewshot_finetune.py")
ROW_KEYS = ("source_index", "patient_id", "label", "label_id", "image_name", "image_sha256")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _portable_rows(rows):
    return [{key: str(row[key]) for key in ROW_KEYS} for row in rows]


def verify_reference(reference, frames, metadata):
    """Require precisely the prior 81 supports, ignoring only machine paths."""
    reference = Path(reference).resolve()
    protocol = _json(reference / "protocol.json")
    if protocol.get("status") != "completed" or protocol.get("protocol_version") != "resnet50_frozen_fewshot_v1":
        raise ValueError("Reference must be a completed frozen ResNet few-shot experiment")
    if protocol.get("shots") != [1, 5, 10, 20] or protocol.get("repeats") != 20 or protocol.get("sampling_seed") != 20261022:
        raise ValueError("Reference is not the prespecified 1/5/10/20 x 20 support plan")
    for split in ("train", "val", "test"):
        old = protocol["splits"][split]
        if old["membership_sha256"] != metadata[split]["membership_sha256"]:
            raise ValueError(f"Reference {split} membership mismatch")
        if split != "test":
            keys = ("source_index", "patient_id", "label", "image_sha256")
            expected = [{key: str(row[key]) for key in keys} for row in old["images"]]
            current = [{key: str(row[key]) for key in keys} for row in frames[split]]
            if expected != current:
                raise ValueError(f"Reference {split} image/order/hash mismatch")
    runs = make_support_plan(frames["train"], [1, 5, 10, 20], 20, 20261022)
    plan = _read_csv(reference / "support_plan.csv")
    if [row["run_id"] for row in plan] != [run["run_id"] for run in runs]:
        raise ValueError("Reference support-plan run/order mismatch")
    hashes = {"protocol.json": sha256_file(reference / "protocol.json"),
              "support_plan.csv": sha256_file(reference / "support_plan.csv")}
    for run, row in zip(runs, plan):
        name = f"supports/{run['run_id']}.csv"
        if row["support_csv"] != name or sha256_file(reference / name) != row["support_csv_sha256"]:
            raise ValueError(f"Reference support CSV checksum mismatch: {run['run_id']}")
        if _portable_rows(_read_csv(reference / name)) != _portable_rows(run["rows"]):
            raise ValueError(f"Reference support membership/order mismatch: {run['run_id']}")
        hashes[name] = sha256_file(reference / name)
    # Aggregate frozen results are optional and used only for descriptive comparison.
    for name in ("per_run_metrics.csv", "summary_metrics.csv"):
        if (reference / name).is_file():
            expected = protocol.get("artifact_sha256", {}).get(name)
            if expected is None or sha256_file(reference / name) != expected:
                raise ValueError(f"Frozen comparison artifact checksum mismatch: {name}")
            hashes[name] = sha256_file(reference / name)
    return runs, hashes


def make_config(args, n_train):
    return {"mode": "smoke" if args.smoke_test else "formal", "training_seed": 42,
            "sampling_seed": 20261022, "weights": "ResNet50_Weights.IMAGENET1K_V2",
            "dropout": 0.3, "batch_size": args.batch_size, "num_workers": args.num_workers,
            "device_requested": args.device, "head_rounds": 1 if args.smoke_test else 5,
            "finetune_rounds": 1 if args.smoke_test else 20,
            "steps_per_round": 2 if args.smoke_test else math.ceil(n_train / args.batch_size),
            "step_policy": "full-size batches from repeated shuffled selected support; equal updates for all budgets",
            "optimizer": "AdamW", "head_lr": 1e-3, "finetune_lr": 1e-4, "weight_decay": 1e-4,
            "finetune_scheduler": "CosineAnnealingLR per round, eta_min=0",
            "batch_norm": "running buffers fixed in both stages; affine trainable in stage 2",
            "train_transform": "src.data.get_train_transforms(crop_border=False)",
            "val_transform": "src.data.get_eval_transforms(crop_border=False)",
            "selection": "final stage-2 checkpoint only; no validation tuning or checkpoint selection",
            "precision": "float32; no AMP; TF32 disabled", "deterministic_algorithms": True}


def prepare(args):
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Refusing to overwrite existing output; use --resume: {output}")
    if args.resume and not (output / "protocol.json").is_file():
        raise ValueError("--resume requires an existing prepared protocol.json")
    frames, metadata = load_splits(args.splits_dir, args.image_root)
    runs, reference_hashes = verify_reference(args.reference_root, frames, metadata)
    if args.smoke_test:
        runs = [runs[0]]
    if args.run_ids and (len(args.run_ids) != len(set(args.run_ids)) or set(args.run_ids) - {r["run_id"] for r in runs}):
        raise ValueError("--run-ids contains duplicate or unknown run IDs")
    config = make_config(args, len(frames["train"]))
    data = {split: {"membership_sha256": metadata[split]["membership_sha256"],
                    "ordered_image_identities": _portable_rows(frames[split]) if split != "test" else None}
            for split in ("train", "val", "test")}
    identity = {"protocol_version": VERSION, "config": config, "data": data,
                "reference_sha256": reference_hashes,
                "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES},
                "run_ids": [r["run_id"] for r in runs]}
    if args.resume:
        protocol = _json(output / "protocol.json")
        if protocol.get("identity") != identity or protocol.get("identity_sha256") != _digest(identity):
            raise ValueError("Resume rejected: source, configuration, reference, or data changed. Use a new output directory.")
        for name, expected in protocol["support_artifact_sha256"].items():
            if not (output / name).is_file() or sha256_file(output / name) != expected:
                raise ValueError(f"Resume rejected: saved support artifact changed: {name}")
        for run in runs:
            validate_completed(output / "runs" / run["run_id"], identity_hash=protocol["identity_sha256"], run_id=run["run_id"])
        return frames, runs, protocol
    output.mkdir(parents=True, exist_ok=True)
    (output / "supports").mkdir()
    (output / "runs").mkdir()
    summaries = []
    for run in runs:
        name = f"supports/{run['run_id']}.csv"
        write_csv(output / name, run["rows"])
        summaries.append({"run_id": run["run_id"], "budget": run["budget"], "repeat": run["repeat"],
                          "n_train_images": len(run["rows"]), "n_train_patients": len({r["patient_id"] for r in run["rows"]}),
                          "support_csv": name})
    write_csv(output / "support_plan.csv", summaries)
    protocol = {"protocol_version": VERSION, "mode": config["mode"], "status": "prepared_not_run",
                "created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity, "identity_sha256": _digest(identity),
                "config": config, "splits": metadata, "reference_root": str(Path(args.reference_root).resolve()),
                "environment_prepared": _environment(), "test_images_accessed": False,
                "support_artifact_sha256": {str(p.relative_to(output)): sha256_file(p)
                                             for p in sorted((output / "supports").glob("*.csv"))}}
    protocol["support_artifact_sha256"]["support_plan.csv"] = sha256_file(output / "support_plan.csv")
    _atomic_json(output / "protocol.json", protocol)
    (output / "REPORT.md").write_text("# ResNet 固定支持集微调\n\n状态：仅完成准备，未运行训练，无性能结论。\n", encoding="utf-8")
    return frames, runs, protocol


def validate_completed(directory, identity_hash, run_id, verify_model=True):
    """An atomic completion marker is written only after reload and evaluation."""
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.is_file():
        return None
    complete = _json(marker)
    required = {"model.pt", "config.json", "history.csv", "predictions.csv", "result.json"}
    if complete.get("identity_sha256") != identity_hash or complete.get("run_id") != run_id:
        raise ValueError(f"Completed run identity mismatch: {run_id}")
    if set(complete.get("artifact_sha256", {})) != required:
        raise ValueError(f"Completed run artifact manifest mismatch: {run_id}")
    for name, expected in complete["artifact_sha256"].items():
        # Aggregation runs after each fit; large checkpoints were already checked
        # at resume/start or written and verified just now. Avoid quadratic I/O.
        if name == "model.pt" and not verify_model:
            continue
        if not (directory / name).is_file() or sha256_file(directory / name) != expected:
            raise ValueError(f"Completed run corrupted: {run_id}/{name}")
    result = _json(directory / "result.json")
    if result.get("run_id") != run_id or not result.get("checkpoint_reload_verified") or not result.get("bn_buffers_unchanged"):
        raise ValueError(f"Completed result verification missing: {run_id}")
    return result


class RepeatedBatchSampler:
    """Cycle independent shuffled support permutations to fill every batch."""
    def __init__(self, n, batch_size, steps, seed):
        if min(n, batch_size, steps) < 1:
            raise ValueError("Positive sampler sizes required")
        self.n, self.batch_size, self.steps, self.seed = n, batch_size, steps, seed

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed)
        pool, cursor = [], 0
        for _ in range(self.steps):
            batch = []
            while len(batch) < self.batch_size:
                if cursor == len(pool):
                    pool = list(range(self.n))
                    rng.shuffle(pool)
                    cursor = 0
                take = min(self.batch_size - len(batch), len(pool) - cursor)
                batch.extend(pool[cursor:cursor + take])
                cursor += take
            yield batch


class ImageRows:
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["label_id"]


def _worker_seed(_worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def seed_training(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_training_stage(model, stage):
    import torch
    if stage not in ("head", "finetune"):
        raise ValueError("Unknown stage")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = stage == "finetune" or name.startswith("fc.")
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_fingerprint(model):
    import torch
    values = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, value in module.named_buffers(recurse=False):
                values[f"{name}.{key}"] = value.detach().cpu().contiguous().numpy().tobytes().hex()
    return _digest(values)


def _backbone_fingerprint(model):
    values = {name: value.detach().cpu().contiguous().numpy().tobytes().hex()
              for name, value in model.state_dict().items() if not name.startswith("fc.")}
    return _digest(values)


def train_fixed_updates(model, dataset, config, device):
    """No validation access. Return history and invariance evidence."""
    import torch
    from torch.utils.data import DataLoader
    bn_before = bn_fingerprint(model)
    backbone_before = _backbone_fingerprint(model)
    history, global_round = [], 0
    for stage, rounds, lr in (("head", config["head_rounds"], config["head_lr"]),
                              ("finetune", config["finetune_rounds"], config["finetune_lr"])):
        set_training_stage(model, stage)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                     weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rounds) if stage == "finetune" else None
        for round_index in range(rounds):
            started = time.perf_counter()
            set_training_stage(model, stage)
            generator = torch.Generator().manual_seed(config["training_seed"] + global_round)
            sampler = RepeatedBatchSampler(len(dataset), config["batch_size"], config["steps_per_round"],
                                           config["training_seed"] + global_round)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["num_workers"],
                                worker_init_fn=_worker_seed, generator=generator, pin_memory=device.type == "cuda")
            total_loss, steps, examples = 0.0, 0, 0
            current_lr = optimizer.param_groups[0]["lr"]
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                finite_gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not torch.stack(finite_gradients).all():
                    raise RuntimeError("Nonfinite training gradient")
                optimizer.step()
                steps += 1
                examples += len(labels)
                total_loss += float(loss.detach()) * len(labels)
            if steps != config["steps_per_round"] or examples != steps * config["batch_size"]:
                raise RuntimeError("Fixed-update/full-batch policy violated")
            history.append({"stage": stage, "round": round_index + 1, "steps": steps, "image_occurrences": examples,
                            "lr": current_lr, "mean_train_loss": total_loss / examples,
                            "seconds": time.perf_counter() - started})
            print(f"  {stage} round {round_index + 1}/{rounds}: steps={steps}, train_loss={total_loss / examples:.5f}", flush=True)
            global_round += 1
            if scheduler is not None:
                scheduler.step()
        if stage == "head" and _backbone_fingerprint(model) != backbone_before:
            raise RuntimeError("Backbone changed during head-only stage")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("Nonfinite final checkpoint tensor")
    bn_after = bn_fingerprint(model)
    if bn_before != bn_after:
        raise RuntimeError("BatchNorm running buffers changed")
    backbone_after = _backbone_fingerprint(model)
    if backbone_after == backbone_before:
        raise RuntimeError("Fine-tuning did not change the backbone")
    return history, {"bn_before_sha256": bn_before, "bn_after_sha256": bn_after,
                     "bn_buffers_unchanged": True, "head_stage_backbone_unchanged": True,
                     "initial_backbone_sha256": backbone_before, "final_backbone_sha256": backbone_after,
                     "finetune_backbone_changed": True,
                     "actual_optimizer_steps": sum(row["steps"] for row in history),
                     "actual_image_occurrences": sum(row["image_occurrences"] for row in history)}


def save_reload_final(model, path, factory, device):
    import torch
    before = state_fingerprint(model)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, path)
    reloaded = factory().to(device)
    reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    after = state_fingerprint(reloaded)
    if before != after:
        raise RuntimeError("Saved final checkpoint failed state reload verification")
    reloaded.eval()
    return reloaded, {"final_state_sha256": before, "reloaded_state_sha256": after, "checkpoint_reload_verified": True}


def evaluate(model, dataset, config, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=False)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for images, _ in loader:
            chunks.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(chunks)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise RuntimeError("Invalid final probabilities")
    return probabilities


def _train_run(args, frames, run, protocol, directory, device):
    from src.data import get_eval_transforms, get_train_transforms
    from src.models import build_resnet50
    config = protocol["config"]
    seed_training(config["training_seed"])
    model = build_resnet50(dropout=config["dropout"], pretrained=True).to(device)
    initial_hash = state_fingerprint(model)
    _atomic_json(directory / "config.json", {"identity_sha256": protocol["identity_sha256"], "run_id": run["run_id"],
                                            "config": config, "support": _portable_rows(run["rows"]), "initial_state_sha256": initial_hash,
                                            "execution_environment": protocol["environment_execution"]})
    history, training = train_fixed_updates(model, ImageRows(run["rows"], get_train_transforms(False)), config, device)
    write_csv(directory / "history.csv", history)
    reloaded, checkpoint = save_reload_final(model, directory / "model.pt",
                                             lambda: build_resnet50(dropout=config["dropout"], pretrained=False), device)
    del model
    probabilities = evaluate(reloaded, ImageRows(frames["val"], get_eval_transforms(False)), config, device)
    labels = [row["label_id"] for row in frames["val"]]
    metrics = classification_metrics(labels, probabilities)
    predictions = []
    for row, probability in zip(frames["val"], probabilities):
        predictions.append({"patient_id": row["patient_id"], "image_name": row["image_name"], "label_id": row["label_id"],
                            "predicted_id": int(probability.argmax()),
                            **{f"prob_{label}": float(probability[label_id]) for label, label_id in LABEL_TO_ID.items()}})
    write_csv(directory / "predictions.csv", predictions)
    result = {"run_id": run["run_id"], "budget": run["budget"], "repeat": run["repeat"],
              "n_train_images": len(run["rows"]), "n_train_patients": len({r["patient_id"] for r in run["rows"]}),
              "initial_state_sha256": initial_hash, **metrics, **training, **checkpoint}
    _atomic_json(directory / "result.json", result)
    artifacts = {name: sha256_file(directory / name) for name in ("model.pt", "config.json", "history.csv", "predictions.csv", "result.json")}
    _atomic_json(directory / "complete.json", {"identity_sha256": protocol["identity_sha256"], "run_id": run["run_id"], "artifact_sha256": artifacts})
    return result


def write_aggregate(args, runs, protocol):
    output = Path(args.output_root).resolve()
    results = []
    for run in runs:
        result = validate_completed(output / "runs" / run["run_id"], protocol["identity_sha256"], run["run_id"], verify_model=False)
        if result is not None:
            results.append(result)
    all_done = len(results) == len(runs)
    if results and len({r["initial_state_sha256"] for r in results}) != 1:
        raise RuntimeError("Independent supports did not start from identical model initialization")
    status = "smoke_completed" if all_done and protocol["mode"] == "smoke" else "completed" if all_done else "partial"
    protocol.update(status=status, completed_runs=len(results), expected_runs=len(runs))
    summaries = summarize_runs(results) if results else []
    if results:
        write_csv(output / "per_run_metrics.csv", results)
        write_csv(output / "summary_metrics.csv", summaries)
    lines = ["# ResNet 固定支持集微调", "", f"状态：{status}；完成 {len(results)}/{len(runs)} 次。", "",
             "仅 formal 且 completed 的 81 次结果构成完整学习曲线；smoke 和 partial 仅供运行检查或进度查看。", "",
             "所有预算使用相同初始 ImageNetV2 权重、分类头种子42、完整批量大小和更新次数。每个训练轮重复打乱已选支持集填满批次；"
             "轮不等于小支持集的一次遍历。两个阶段固定 BatchNorm 运行统计，第二阶段训练仿射参数。最后一轮模型保存并重载验证后评估，未按验证集挑选模型。", "",
             "| 每类图片 | 已完成支持集 | 验证准确率均值 | 抽样标准差 | 最低—最高 |", "|---|---:|---:|---:|---:|"]
    for row in summaries:
        if row["metric"] != "accuracy":
            continue
        sd = "不适用" if row["sample_sd"] is None else f"{row['sample_sd'] * 100:.2f} 个百分点"
        lines.append(f"| {row['budget']} | {row['n_support_draws']} | {row['mean'] * 100:.2f}% | {sd} | {row['min'] * 100:.2f}%—{row['max'] * 100:.2f}% |")
    frozen_path = Path(args.reference_root) / "per_run_metrics.csv"
    if results and frozen_path.is_file():
        frozen = {row["run_id"]: row for row in _read_csv(frozen_path)}
        paired = [{"run_id": row["run_id"], "budget": row["budget"], "frozen_accuracy": float(frozen[row["run_id"]]["accuracy"]),
                   "finetuned_accuracy": row["accuracy"], "accuracy_difference": row["accuracy"] - float(frozen[row["run_id"]]["accuracy"])} for row in results]
        write_csv(output / "paired_frozen_comparison.csv", paired)
        lines += ["", "paired_frozen_comparison.csv 按同一支持集逐次配对。差异反映整个适配流程的变化（包括增强、分类头训练和主干微调），不能解释为单一组件的因果作用。"]
    lines += ["", "全量只运行一次。标准差描述相同验证患者下支持集抽样的波动，不是患者泛化置信区间。训练随机种子固定，未估计优化随机性。",
              "测试集仅检查 CSV 身份隔离，未访问测试图像。当前验证集已有探索历史。历史约98%使用不同训练细节，应先与本轮全量对照比较。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if status == "completed":
        _plot(output, results, summaries, args.reference_root)
    _atomic_json(output / "protocol.json", protocol)
    return results


def _plot(output, results, summaries, reference_root):
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_fewshot_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    few = [row for row in summaries if row["metric"] == "accuracy" and row["budget"] != "full"]
    full = next(row for row in results if row["budget"] == "full")
    ax.scatter([int(r["budget"]) for r in results if r["budget"] != "full"],
               [r["accuracy"] * 100 for r in results if r["budget"] != "full"], alpha=.25, s=15, color="#286e9c")
    ax.errorbar([int(r["budget"]) for r in few], [r["mean"] * 100 for r in few],
                yerr=[r["sample_sd"] * 100 for r in few], fmt="o-", capsize=4, label="Fine-tuning, mean +/- support SD")
    ax.axhline(full["accuracy"] * 100, linestyle="--", label="Fine-tuning full data, one run")
    frozen_path = Path(reference_root) / "summary_metrics.csv"
    if frozen_path.is_file():
        frozen = [r for r in _read_csv(frozen_path) if r["metric"] == "accuracy" and r["budget"] != "full"]
        ax.plot([int(r["budget"]) for r in frozen], [float(r["mean"]) * 100 for r in frozen], "s--", color="#ba683d", label="Frozen linear probe (different adaptation)")
    ax.set(xlabel="Labeled training images per class", ylabel="Validation accuracy (%)", xticks=[1, 5, 10, 20], ylim=(0, 102))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.text(.5, .01, "Variation across support draws on fixed validation patients, not a patient confidence interval.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "learning_curve.png", dpi=200)
    fig.savefig(output / "learning_curve.pdf")
    plt.close(fig)


def execute(args, frames, runs, protocol):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check the server GPU/PyTorch installation; no silent CPU fallback.")
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(args.output_root).resolve()
    protocol["environment_execution"] = _environment()
    protocol["environment_execution"].update(device=str(device), torch_version=torch.__version__,
                                              gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    protocol.setdefault("execution_attempts", []).append({
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_run_ids": args.run_ids,
        "environment": protocol["environment_execution"],
    })
    protocol["status"] = "running"
    _atomic_json(output / "protocol.json", protocol)
    selected = set(args.run_ids) if args.run_ids else {r["run_id"] for r in runs}
    current = None
    try:
        for index, run in enumerate(runs, 1):
            current = run["run_id"]
            directory = output / "runs" / current
            if current not in selected:
                continue
            if validate_completed(directory, protocol["identity_sha256"], current) is not None:
                print(f"[{index}/{len(runs)}] {current}: verified, already completed", flush=True)
                continue
            if directory.exists():
                archive = output / "incomplete_attempts"
                archive.mkdir(exist_ok=True)
                shutil.move(str(directory), str(archive / f"{current}_{time.time_ns()}"))
            directory.mkdir()
            started = time.perf_counter()
            print(f"[{index}/{len(runs)}] {current}: {len(run['rows'])} unique labeled images", flush=True)
            result = _train_run(args, frames, run, protocol, directory, device)
            print(f"  final accuracy={result['accuracy']:.4f}; elapsed={time.perf_counter() - started:.1f}s", flush=True)
            write_aggregate(args, runs, protocol)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_aggregate(args, runs, protocol)
    except Exception as exc:
        protocol["status"] = "failed"
        protocol["failure"] = {"run_id": current, "type": type(exc).__name__, "message": str(exc)}
        _atomic_json(output / "protocol.json", protocol)
        raise
