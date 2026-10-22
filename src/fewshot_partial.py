"""Paired 5/10-shot ResNet layer4 adaptation with immutable full-tuning controls.

Only train-support and validation images are loaded. The existing trainers and
completed experiments remain untouched. Packaging works without images or CUDA.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import statistics
import tarfile
import time
from types import SimpleNamespace

from src.baseline_reproduction import (check_environment, output_lock, read_csv, read_json,
                                      require, runtime_identity, verify_prediction_metrics)
from src.fewshot import (PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, classification_metrics,
                        load_splits, sha256_file, state_fingerprint, summarize_runs, write_csv)
from src.fewshot_finetune import (SOURCE_FILES as OLD_SOURCES, VERSION as OLD_VERSION,
                                 ImageRows, RepeatedBatchSampler, _atomic_json, _digest,
                                 _portable_rows, _backbone_fingerprint, _worker_seed,
                                 bn_fingerprint, evaluate, make_config, save_reload_final,
                                 seed_training, validate_completed, verify_reference)

VERSION = "resnet50_layer4_paired_5_10_shot_v1"
RUN_IDS = tuple(f"shot_{shot:03d}_repeat_{repeat:03d}" for repeat in range(20) for shot in (5, 10))
OWN_SOURCES = ("src/fewshot_partial.py", "src/baseline_reproduction.py",
               "scripts/run_resnet_layer4_suite.py", "scripts/run_layer4_suite_server.sh")
RUN_FILES = ("config.json", "history.csv", "predictions.csv", "result.json", "complete.json")
AGGREGATE_FILES = ("per_run_metrics.csv", "summary_metrics.csv", "paired_comparison.csv", "paired_summary.csv", "REPORT.md")


def fixed_config(n_train):
    return make_config(SimpleNamespace(smoke_test=False, batch_size=16, num_workers=4, device="cuda"), n_train)


def partial_config(n_train):
    return {**fixed_config(n_train), "stage2_trainable": "layer4.* and fc.* only",
            "stage2_batch_norm_affine": "layer4 only; all running buffers fixed"}


def safe_path(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f"Unsafe artifact path: {name}")
    return path


def verify_history(path, config):
    history = read_csv(path)
    expected = [(stage, index + 1) for stage, count in (("head", config["head_rounds"]),
                                                     ("finetune", config["finetune_rounds"])) for index in range(count)]
    require([(r["stage"], int(r["round"])) for r in history] == expected, "Training history stage/round mismatch")
    for row in history:
        stage, index = row["stage"], int(row["round"]) - 1
        lr = config["head_lr"] if stage == "head" else config["finetune_lr"] * (1 + math.cos(math.pi * index / config["finetune_rounds"])) / 2
        require(int(row["steps"]) == config["steps_per_round"] and
                int(row["image_occurrences"]) == config["steps_per_round"] * config["batch_size"], "Update budget mismatch")
        require(math.isclose(float(row["lr"]), lr, rel_tol=1e-10, abs_tol=1e-15), "Training learning-rate schedule mismatch")
        require(math.isfinite(float(row["mean_train_loss"])) and float(row["mean_train_loss"]) >= 0, "Invalid training loss")
    return history


def compare_head_history(new, old, count):
    keys = ("stage", "round", "steps", "image_occurrences", "lr", "mean_train_loss")
    require(len(new) >= count and len(old) >= count, "Missing head-stage history")
    for left, right in zip(new[:count], old[:count]):
        for key in keys:
            same = str(left[key]) == str(right[key]) if key == "stage" else math.isclose(float(left[key]), float(right[key]), rel_tol=1e-9, abs_tol=1e-9)
            require(same, f"Head-stage history differs from paired full-tuning reference: {key}")
    return True


def verify_reference_bundle(root, frames, metadata, check_sources=True):
    runs, frozen_hashes = verify_reference(root / "frozen", frames, metadata)
    runs = [run for run in runs if run["run_id"] in RUN_IDS]
    require(tuple(run["run_id"] for run in runs) == RUN_IDS, "Expected exactly 40 paired support draws")
    protocol = read_json(root / "finetune/protocol.json")
    identity = protocol.get("identity", {})
    require(protocol.get("status") == "completed" and protocol.get("mode") == "formal" and
            protocol.get("protocol_version") == OLD_VERSION and protocol.get("completed_runs") == 81 and
            protocol.get("expected_runs") == 81 and protocol.get("test_images_accessed") is False,
            "Reference fine-tuning protocol is not a completed formal 81-run suite")
    require(protocol.get("identity_sha256") == _digest(identity), "Reference identity digest mismatch")
    require(identity.get("protocol_version") == OLD_VERSION and identity.get("config") == protocol.get("config") == fixed_config(len(frames["train"])),
            "Reference fixed-update configuration mismatch")
    require(identity.get("reference_sha256") == frozen_hashes, "Reference frozen provenance mismatch")
    expected_ids = [r["run_id"] for r in read_csv(root / "frozen/support_plan.csv")]
    require(identity.get("run_ids") == expected_ids, "Reference run plan mismatch")
    require(set(identity.get("source_sha256", {})) == set(OLD_SOURCES), "Reference source manifest mismatch")
    if check_sources:
        for name, expected in identity["source_sha256"].items():
            require(sha256_file(safe_path(PROJECT_ROOT, name)) == expected, f"Reference source changed: {name}")
    for split in ("train", "val", "test"):
        require(identity["data"][split]["membership_sha256"] == metadata[split]["membership_sha256"], f"Reference {split} membership mismatch")
        if split != "test":
            require(identity["data"][split]["ordered_image_identities"] == _portable_rows(frames[split]), f"Reference {split} image/order/hash mismatch")
    support_hashes = protocol.get("support_artifact_sha256", {})
    require(set(support_hashes) == {"support_plan.csv"} | {f"supports/{run_id}.csv" for run_id in expected_ids}, "Reference support artifact manifest mismatch")
    for name, expected in support_hashes.items():
        require(sha256_file(safe_path(root / "finetune", name)) == expected, f"Reference support artifact changed: {name}")
    hashes = {f"frozen/{name}": value for name, value in frozen_hashes.items()}
    hashes.update({f"finetune/{name}": value for name, value in support_hashes.items()})
    hashes["finetune/protocol.json"] = sha256_file(root / "finetune/protocol.json")
    environment = protocol["environment_execution"]
    initials = set()
    for run in runs:
        run_id = run["run_id"]
        directory = root / "finetune/runs" / run_id
        result = validate_completed(directory, protocol["identity_sha256"], run_id, verify_model=(directory / "model.pt").exists())
        require(result is not None, f"Reference run is incomplete: {run_id}")
        config = read_json(directory / "config.json")
        require(config.get("identity_sha256") == protocol["identity_sha256"] and config.get("run_id") == run_id and config.get("config") == protocol["config"], "Reference run configuration mismatch")
        require(config.get("support") == _portable_rows(run["rows"]) == _portable_rows(read_csv(root / "finetune/supports" / f"{run_id}.csv")), "Reference paired support mismatch")
        require(config.get("execution_environment") == environment, "Reference run environment differs")
        require(config.get("initial_state_sha256") == result.get("initial_state_sha256"), "Reference initialization record mismatch")
        require(result.get("budget") == run["budget"] and result.get("repeat") == run["repeat"] and result.get("n_train_images") == len(run["rows"]) and
                result.get("n_train_patients") == len({r["patient_id"] for r in run["rows"]}), "Reference run budget mismatch")
        steps = (protocol["config"]["head_rounds"] + protocol["config"]["finetune_rounds"]) * protocol["config"]["steps_per_round"]
        require(result.get("actual_optimizer_steps") == steps and result.get("actual_image_occurrences") == steps * 16 and
                result.get("head_stage_backbone_unchanged") is True and result.get("finetune_backbone_changed") is True,
                "Reference training evidence mismatch")
        for left, right in (("bn_before_sha256", "bn_after_sha256"), ("final_state_sha256", "reloaded_state_sha256")):
            require(isinstance(result.get(left), str) and result[left] == result.get(right), f"Reference fingerprint mismatch: {left}")
        verify_prediction_metrics(directory / "predictions.csv", result, frames["val"])
        verify_history(directory / "history.csv", protocol["config"])
        initials.add(result["initial_state_sha256"])
        hashes.update({f"finetune/runs/{run_id}/{name}": sha256_file(directory / name) for name in RUN_FILES})
    require(len(initials) == 1, "Reference runs used different initializations")
    return runs, hashes, environment


def verify_runtime(old, current):
    require(current["python"] == old["python"] and current["packages"] == old["packages"],
            "Environment differs from paired controls; activate their original Python/package environment")
    require(current["gpu"] == old["gpu"], "GPU model differs from paired controls")


def check_inputs(args):
    require(not args.output_root.is_relative_to(args.reference_root) and not args.reference_root.is_relative_to(args.output_root),
            "Reference and output directories must be disjoint")
    frames, metadata = load_splits(args.splits_dir, args.image_root)
    runs, hashes, environment = verify_reference_bundle(args.reference_root, frames, metadata)
    require(len(frames["train"]) == 446 and len(frames["val"]) == 96 and len(frames["test"]) == 99,
            "This paired experiment requires the audited 446/96/99 split")
    identity = {"version": VERSION, "config": partial_config(len(frames["train"])), "run_ids": list(RUN_IDS),
                "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in sorted(set(OWN_SOURCES) | set(OLD_SOURCES))},
                "reference_sha256": hashes, "reference_environment": environment, "splits": metadata,
                "data": {split: _portable_rows(frames[split]) if split != "test" else None for split in frames},
                "splits_dir": str(args.splits_dir), "image_root": str(args.image_root)}
    return frames, runs, identity


def prepare_suite(args, identity, environment):
    root = args.output_root
    path = root / "suite.json"
    if path.exists():
        suite = read_json(path)
        require(suite.get("identity") == identity and suite.get("identity_sha256") == _digest(identity), "Resume rejected: source/data/config/reference changed")
        require(suite.get("runtime_identity") == runtime_identity(environment), "Resume rejected: runtime changed")
        verify_copied_reference(root, suite)
        return suite
    require(not [p for p in root.iterdir() if p.name != ".reproduction.lock"], "Output is nonempty without a valid suite identity")
    for name in identity["reference_sha256"]:
        target = safe_path(root / "references", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(args.reference_root, name), target)
    suite = {"version": VERSION, "identity": identity, "identity_sha256": _digest(identity), "environment": environment,
             "runtime_identity": runtime_identity(environment), "status": "prepared_not_run", "completed_runs": 0,
             "created_utc": datetime.now(timezone.utc).isoformat(), "test_images_accessed": False}
    verify_copied_reference(root, suite)
    _atomic_json(path, suite)
    return suite


def verify_copied_reference(root, suite):
    identity = suite["identity"]
    require(suite.get("version") == VERSION and identity.get("version") == VERSION and
            suite.get("identity_sha256") == _digest(identity) and identity.get("run_ids") == list(RUN_IDS), "Suite identity mismatch")
    for name, expected in identity["reference_sha256"].items():
        path = safe_path(root / "references", name)
        require(path.is_file() and sha256_file(path) == expected, f"Copied reference corrupted: {name}")
    frames = {split: identity["data"][split] for split in ("train", "val")}
    frames["test"] = []  # Identity checks use metadata only; test pixels were never loaded.
    _, hashes, environment = verify_reference_bundle(root / "references", frames, identity["splits"], check_sources=False)
    require(hashes == identity["reference_sha256"] and environment == identity["reference_environment"], "Copied reference identity mismatch")
    require(identity["config"] == partial_config(len(frames["train"])), "Partial configuration mismatch")


def set_training_stage(model, stage, scope="layer4"):
    import torch
    require(stage in ("head", "finetune") and scope in ("full", "layer4"), "Unknown training stage/scope")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.") or (stage == "finetune" and (scope == "full" or name.startswith("layer4.")))
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_fingerprint(model, part):
    values = {name: value.detach().cpu().contiguous().numpy().tobytes().hex() for name, value in model.named_parameters()
              if (part == "early" and not name.startswith(("layer4.", "fc."))) or
                 (part in ("layer4", "fc") and name.startswith(part + "."))}
    require(bool(values), f"Model is missing parameter group: {part}")
    return _digest(values)


def train_fixed_updates(model, dataset, config, device, scope="layer4"):
    """The old loop's RNG/optimizer order is retained; only trainable scope changes."""
    import torch
    from torch.utils.data import DataLoader
    bn_before, backbone_before = bn_fingerprint(model), _backbone_fingerprint(model)
    early_before = parameter_fingerprint(model, "early")
    stage2_before = None
    history, global_round = [], 0
    for stage, rounds, lr in (("head", config["head_rounds"], config["head_lr"]),
                              ("finetune", config["finetune_rounds"], config["finetune_lr"])):
        set_training_stage(model, stage, scope)
        if stage == "finetune":
            stage2_before = {part: parameter_fingerprint(model, part) for part in ("layer4", "fc")}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rounds) if stage == "finetune" else None
        for round_index in range(rounds):
            started = time.perf_counter()
            set_training_stage(model, stage, scope)
            generator = torch.Generator().manual_seed(config["training_seed"] + global_round)
            sampler = RepeatedBatchSampler(len(dataset), config["batch_size"], config["steps_per_round"], config["training_seed"] + global_round)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["num_workers"], worker_init_fn=_worker_seed,
                                generator=generator, pin_memory=device.type == "cuda")
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
            require(steps == config["steps_per_round"] and examples == steps * config["batch_size"], "Fixed-update/full-batch policy violated")
            history.append({"stage": stage, "round": round_index + 1, "steps": steps, "image_occurrences": examples,
                            "lr": current_lr, "mean_train_loss": total_loss / examples, "seconds": time.perf_counter() - started})
            print(f"  {stage} round {round_index + 1}/{rounds}: steps={steps}, train_loss={total_loss / examples:.5f}", flush=True)
            global_round += 1
            if scheduler is not None:
                scheduler.step()
        if stage == "head":
            require(_backbone_fingerprint(model) == backbone_before, "Backbone changed during head-only stage")
    require(all(torch.isfinite(value).all() for value in model.state_dict().values()), "Nonfinite final checkpoint tensor")
    bn_after, backbone_after = bn_fingerprint(model), _backbone_fingerprint(model)
    early_after = parameter_fingerprint(model, "early")
    require(bn_before == bn_after and backbone_before != backbone_after, "BatchNorm changed or backbone did not adapt")
    if scope == "layer4":
        require(early_before == early_after, "Frozen stem/layer1-layer3 parameters changed")
    stage2_after = {part: parameter_fingerprint(model, part) for part in ("layer4", "fc")}
    require(all(stage2_before[part] != stage2_after[part] for part in stage2_before), "layer4/fc failed to update in stage 2")
    return history, {"bn_before_sha256": bn_before, "bn_after_sha256": bn_after, "bn_buffers_unchanged": True,
                     "head_stage_backbone_unchanged": True, "initial_backbone_sha256": backbone_before,
                     "final_backbone_sha256": backbone_after, "finetune_backbone_changed": True,
                     "early_before_sha256": early_before, "early_after_sha256": early_after,
                     "early_parameters_unchanged": early_before == early_after,
                     "stage2_before_sha256": stage2_before, "stage2_after_sha256": stage2_after,
                     "layer4_updated": True, "fc_updated_in_stage2": True,
                     "actual_optimizer_steps": sum(row["steps"] for row in history),
                     "actual_image_occurrences": sum(row["image_occurrences"] for row in history)}


def train_run(root, suite, frames, run, directory, device):
    from src.data import get_eval_transforms, get_train_transforms
    from src.models import build_resnet50
    config = suite["identity"]["config"]
    reference = root / "references/finetune/runs" / run["run_id"]
    old_result = read_json(reference / "result.json")
    seed_training(config["training_seed"])
    model = build_resnet50(dropout=config["dropout"], pretrained=True).to(device)
    initial = state_fingerprint(model)
    require(initial == old_result["initial_state_sha256"], "Initialization differs from paired full-tuning control")
    _atomic_json(directory / "config.json", {"identity_sha256": suite["identity_sha256"], "run_id": run["run_id"],
                 "config": config, "support": _portable_rows(run["rows"]), "initial_state_sha256": initial,
                 "execution_environment": suite["environment"]})
    history, evidence = train_fixed_updates(model, ImageRows(run["rows"], get_train_transforms(False)), config, device)
    write_csv(directory / "history.csv", history)
    compare_head_history(history, read_csv(reference / "history.csv"), config["head_rounds"])
    reloaded, checkpoint = save_reload_final(model, directory / "model.pt", lambda: build_resnet50(dropout=config["dropout"], pretrained=False), device)
    del model
    probabilities = evaluate(reloaded, ImageRows(frames["val"], get_eval_transforms(False)), config, device)
    metrics = classification_metrics([row["label_id"] for row in frames["val"]], probabilities)
    write_csv(directory / "predictions.csv", [{"patient_id": row["patient_id"], "image_name": row["image_name"], "label_id": row["label_id"],
              "predicted_id": int(prob.argmax()), **{f"prob_{label}": float(prob[index]) for label, index in LABEL_TO_ID.items()}}
              for row, prob in zip(frames["val"], probabilities)])
    result = {"run_id": run["run_id"], "budget": run["budget"], "repeat": run["repeat"], "n_train_images": len(run["rows"]),
              "n_train_patients": len({row["patient_id"] for row in run["rows"]}), "initial_state_sha256": initial,
              "initialization_matches_reference": True, "head_history_matches_reference": True, "test_images_accessed": False,
              **metrics, **evidence, **checkpoint}
    _atomic_json(directory / "result.json", result)
    artifacts = {name: sha256_file(directory / name) for name in (*RUN_FILES[:-1], "model.pt")}
    _atomic_json(directory / "complete.json", {"identity_sha256": suite["identity_sha256"], "run_id": run["run_id"], "artifact_sha256": artifacts})
    return result


def validate_run(root, suite, run_id, require_weights=True, skip_weights=False):
    require(run_id in RUN_IDS, "Unknown run")
    directory = root / "runs" / run_id
    # Only online aggregation skips weights already checked at start/after the fit.
    # Resume and offline packaging still verify every checkpoint that is present.
    require(not (skip_weights and require_weights), "Conflicting weight validation policy")
    result = validate_completed(directory, suite["identity_sha256"], run_id,
                                verify_model=not skip_weights and (require_weights or (directory / "model.pt").exists()))
    if result is None:
        return None
    config = read_json(directory / "config.json")
    expected = suite["identity"]["config"]
    old = root / "references/finetune/runs" / run_id
    old_config, old_result = read_json(old / "config.json"), read_json(old / "result.json")
    require(config.get("config") == expected and config.get("run_id") == run_id and config.get("identity_sha256") == suite["identity_sha256"], "Run configuration mismatch")
    require(config.get("support") == old_config["support"] and config.get("execution_environment") == suite["environment"], "Run support/environment mismatch")
    require(result.get("initial_state_sha256") == config.get("initial_state_sha256") == old_result["initial_state_sha256"], "Run initialization mismatch")
    for key in ("budget", "repeat", "n_train_images", "n_train_patients"):
        require(result.get(key) == old_result.get(key), f"Run metadata mismatch: {key}")
    for key in ("early_parameters_unchanged", "layer4_updated", "fc_updated_in_stage2", "head_stage_backbone_unchanged",
                "initialization_matches_reference", "head_history_matches_reference", "finetune_backbone_changed"):
        require(result.get(key) is True, f"Missing training invariant: {key}")
    require(result.get("test_images_accessed") is False, "Unexpected test image access")
    for left, right in (("early_before_sha256", "early_after_sha256"), ("bn_before_sha256", "bn_after_sha256"),
                        ("final_state_sha256", "reloaded_state_sha256")):
        require(isinstance(result.get(left), str) and result[left] == result.get(right), f"Training fingerprint mismatch: {left}")
    for part in ("layer4", "fc"):
        require(result.get("stage2_before_sha256", {}).get(part) and result["stage2_before_sha256"][part] != result.get("stage2_after_sha256", {}).get(part), "Stage-2 update fingerprint mismatch")
    history = verify_history(directory / "history.csv", expected)
    compare_head_history(history, read_csv(old / "history.csv"), expected["head_rounds"])
    steps = sum(int(row["steps"]) for row in history)
    require(result.get("actual_optimizer_steps") == steps == 700 and result.get("actual_image_occurrences") == steps * 16, "Run step evidence mismatch")
    verify_prediction_metrics(directory / "predictions.csv", result, suite["identity"]["data"]["val"])
    return result


def aggregate_rows(root, suite, require_weights=False, skip_weights=False):
    results, paired = [], []
    for run_id in RUN_IDS:
        result = validate_run(root, suite, run_id, require_weights, skip_weights)
        if result is None:
            continue
        old = read_json(root / "references/finetune/runs" / run_id / "result.json")
        for method, row in (("full_finetune", old), ("layer4_fc", result)):
            results.append({"method": method, **{key: row[key] for key in ("run_id", "budget", "repeat", "n_train_images", "n_train_patients", *METRIC_NAMES)}})
        paired.append({"run_id": run_id, "budget": result["budget"], "repeat": result["repeat"],
                       **{f"{metric}_{kind}": value for metric in METRIC_NAMES for kind, value in
                          (("full", old[metric]), ("layer4", result[metric]), ("difference", result[metric] - old[metric]))}})
    summaries = [{"method": method, **row} for method in ("full_finetune", "layer4_fc") for row in summarize_runs([r for r in results if r["method"] == method])]
    paired_summary = []
    for budget in ("5", "10"):
        for metric in METRIC_NAMES:
            differences = [r[f"{metric}_difference"] for r in paired if r["budget"] == budget]
            if differences:
                signed = [-v if metric == "loss" else v for v in differences]
                paired_summary.append({"budget": budget, "metric": metric, "n_pairs": len(differences), "mean_difference": statistics.mean(differences),
                                       "sample_sd": statistics.stdev(differences) if len(differences) > 1 else None,
                                       "wins": sum(v > 1e-12 for v in signed), "ties": sum(abs(v) <= 1e-12 for v in signed), "losses": sum(v < -1e-12 for v in signed)})
    return results, summaries, paired, paired_summary


def write_aggregate(root, suite):
    results, summaries, paired, paired_summary = aggregate_rows(root, suite, skip_weights=True)
    if paired:
        for name, rows in zip(AGGREGATE_FILES[:4], (results, summaries, paired, paired_summary)):
            write_csv(root / name, rows)
    suite.update(completed_runs=len(paired), status="completed" if len(paired) == 40 else "partial")
    lines = ["# ResNet-50 layer4＋分类头：5-shot / 10-shot 配对比较", "", f"状态：{suite['status']}；新增完成 {len(paired)}/40 次。", "",
             "主指标为固定验证集准确率，Macro-F1 为次指标。整体微调对照复用原40次结果；新方法仅改变第二阶段可训练参数范围。", "",
             "| 每类标注图像 | 配对数 | 整体微调准确率 | layer4＋分类头准确率 | 平均差（百分点） | 胜/平/负 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in paired_summary:
        if row["metric"] == "accuracy":
            values = {r["method"]: r for r in summaries if r["budget"] == row["budget"] and r["metric"] == "accuracy"}
            formatted = lambda method: f"{values[method]['mean']*100:.2f}% ± {(values[method]['sample_sd'] or 0)*100:.2f}"
            lines.append(f"| {row['budget']} | {row['n_pairs']} | {formatted('full_finetune')} | {formatted('layer4_fc')} | {row['mean_difference']*100:+.2f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "每组固定 seed42、同一支持集和初始化、140次分类头更新＋560次第二阶段更新，所有 BatchNorm 运行统计固定。",
              "逐组核验初始化与头阶段训练损失；旧实验未保存逐批输入指纹，不能声称新旧完整输入流已逐位核验。",
              "标准差反映同一验证患者集下20次支持集抽样波动，不是独立患者置信区间；未估计训练种子波动。",
              "测试集仅检查 CSV 身份隔离，未读取测试图像。当前验证集已有探索历史，不能视为未接触的最终测试。",
              "此实验只检验部分微调是否构成更强对照；无论结果高低，都不预先证明方法创新或专利创造性。", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if len(paired) == 40:
        suite["aggregate_sha256"] = {name: sha256_file(root / name) for name in AGGREGATE_FILES}
    _atomic_json(root / "suite.json", suite)


def run_suite(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    environment = check_environment()
    frames, runs, identity = check_inputs(args)
    verify_runtime(identity["reference_environment"], environment)
    with output_lock(args.output_root):
        suite = prepare_suite(args, identity, environment)
        # Audit every completed fit before starting any additional GPU work.
        for run_id in RUN_IDS:
            validate_run(args.output_root, suite, run_id)
        current = None
        try:
            for index, run in enumerate(runs, 1):
                current = run["run_id"]
                if validate_run(args.output_root, suite, current) is not None:
                    print(f"[{index}/40] {current}: verified, already completed", flush=True)
                    continue
                directory = args.output_root / "runs" / current
                if directory.exists():
                    archive = args.output_root / "incomplete_attempts"
                    archive.mkdir(exist_ok=True)
                    shutil.move(str(directory), str(archive / f"{current}_{time.time_ns()}"))
                directory.mkdir(parents=True)
                suite.update(status="running", active_run=current)
                _atomic_json(args.output_root / "suite.json", suite)
                print(f"[{index}/40] {current}: layer4 + fc, {len(run['rows'])} labeled images", flush=True)
                train_run(args.output_root, suite, frames, run, directory, torch.device("cuda"))
                validate_run(args.output_root, suite, current)
                write_aggregate(args.output_root, suite)
                torch.cuda.empty_cache()
            suite.pop("active_run", None)
            suite.pop("failure", None)
            write_aggregate(args.output_root, suite)
        except Exception as exc:
            suite.update(status="failed", failure={"run_id": current, "type": type(exc).__name__, "message": str(exc)})
            _atomic_json(args.output_root / "suite.json", suite)
            raise
    print("All 40 layer4 fits completed. Pack with the same entry point's pack mode.", flush=True)


def pack_suite(args):
    root = args.output_root
    with output_lock(root):
        suite = read_json(root / "suite.json")
        verify_copied_reference(root, suite)
        require(suite.get("status") == "completed" and suite.get("completed_runs") == 40, "Only a completed 40-run suite can be packed")
        expected_rows = aggregate_rows(root, suite)
        require(len(expected_rows[0]) == 80 and len(expected_rows[2]) == 40, "Incomplete paired results")
        require(set(suite.get("aggregate_sha256", {})) == set(AGGREGATE_FILES), "Aggregate manifest mismatch")
        for index, name in enumerate(AGGREGATE_FILES):
            require(sha256_file(root / name) == suite["aggregate_sha256"][name], f"Aggregate corrupted: {name}")
            if index < 4:
                expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in expected_rows[index]]
                require(read_csv(root / name) == expected, f"Aggregate differs from run evidence: {name}")
        archive = root.parent / f"{root.name}_results.tar.gz"
        checksum = Path(str(archive) + ".sha256")
        require(not archive.exists() and not checksum.exists(), "Refusing to overwrite an existing result archive/checksum")
        names = ["suite.json", *AGGREGATE_FILES]
        names += [f"references/{name}" for name in suite["identity"]["reference_sha256"]]
        names += [f"runs/{run_id}/{name}" for run_id in RUN_IDS for name in RUN_FILES]
        temporary = archive.with_name(archive.name + ".tmp")
        require(not temporary.exists(), "Unfinished archive exists; preserve/remove it before packaging")
        with tarfile.open(temporary, "w:gz") as handle:
            for name in sorted(names):
                path = safe_path(root, name)
                require(path.is_file(), f"Missing archive file: {name}")
                handle.add(path, arcname=name, recursive=False)
        temporary.replace(archive)
        checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"Packed: {archive}\nChecksum: {checksum}", flush=True)
        return archive


def main(args):
    if args.mode == "pack":
        return pack_suite(args)
    if args.mode == "run":
        return run_suite(args)
    frames, runs, identity = check_inputs(args)
    environment = check_environment()
    verify_runtime(identity["reference_environment"], environment)
    print(f"Checks passed: {len(runs)} paired controls; train/val/test={len(frames['train'])}/{len(frames['val'])}/{len(frames['test'])}; test images not loaded.")
    print("Ready for 40 new CUDA fits: 5/10-shot × 20 draws, seed42, layer4 + fc, 700 updates per fit.")
