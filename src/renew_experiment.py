"""Isolated, patient-grouped renew experiments; old splits and weights are unused."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import tarfile
import time

from src.baseline_reproduction import (output_lock, read_csv, read_json, require,
                                      verify_prediction_metrics)
from src.fewshot import (PROJECT_ROOT, LABEL_TO_ID, METRIC_NAMES, _environment,
                        classification_metrics, sha256_file, state_fingerprint, write_csv)
from src.fewshot_finetune import (ImageRows, _atomic_json, _digest, evaluate,
                                save_reload_final, seed_training, train_fixed_updates,
                                validate_completed)
from src.renew_data import (EXPECTED_FINGERPRINT, audit_dataset, make_patient_folds,
                            get_transforms, get_geometry_transforms)

VERSION = "renew_resnet50_patient5fold_fullframe_v1"
METHODS = ("original", "full_frame")
SOURCE_FILES = ("src/renew_experiment.py", "src/renew_data.py", "src/data.py", "src/models.py",
                "src/fewshot.py", "src/fewshot_finetune.py", "src/baseline_reproduction.py",
                "scripts/run_renew_experiment.py", "scripts/run_renew_server.sh")
ROW_KEYS = ("source_index", "patient_id", "patient_key", "label", "label_id", "image_name", "image_sha256")
RUN_FILES = ("config.json", "history.csv", "predictions.csv", "result.json", "complete.json")
AGGREGATES = ("per_run_metrics.csv", "paired_comparison.csv", "oof_metrics.csv",
              "oof_predictions.csv", "oof_paired_comparison.csv", "summary_metrics.csv", "REPORT.md")


def now():
    return datetime.now(timezone.utc).isoformat()


def portable(rows):
    return [{key: row[key] for key in ROW_KEYS} for row in rows]


def fixed_config():
    return {"mode": "formal", "training_seed": 42, "fold_seed": 42, "sampling_seed": 20261104,
            "weights": "ResNet50_Weights.IMAGENET1K_V2", "dropout": .3, "batch_size": 16,
            "num_workers": 4, "head_rounds": 5, "finetune_rounds": 20, "steps_per_round": 28,
            "head_lr": 1e-3, "finetune_lr": 1e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
            "finetune_scheduler": "CosineAnnealingLR per round, eta_min=0",
            "batch_norm": "running statistics frozen; affine trainable in second stage",
            "selection": "final checkpoint; no holdout-driven early stopping or threshold selection",
            "precision": "float32; AMP and TF32 disabled; deterministic algorithms",
            "update_budget": "140 head + 560 whole-network steps, independent of fold/support size",
            "pairing": "same support and initialization; augmentation RNG sequences may differ",
            "interpretation": "development CV; full-frame pipeline also changes scale, padding and rotation edges"}


def source_hashes():
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def make_plan(rows, config):
    folds = make_patient_folds(rows, n_folds=5, seed=config["fold_seed"])
    fold_records, supports, full, few = [], {}, [], []
    for fold in folds:
        index = fold["fold"]
        train, heldout = fold["train"], fold["eval"]
        keys = sorted(fold["train_patient_keys"])
        require(len(keys) >= 10, "Each fold needs at least ten training patients")
        require(not set(keys) & set(fold["eval_patient_keys"]), "Patient leakage in fold")
        fold_records.append({"fold": index, "train": [r["image_name"] for r in train],
                             "eval": [r["image_name"] for r in heldout],
                             "train_patient_keys": keys, "eval_patient_keys": fold["eval_patient_keys"]})
        selections = [("full", 0, keys, None)]
        for repeat in range(5):
            seed = config["sampling_seed"] + index * 100 + repeat
            shuffled = list(keys)
            random.Random(seed).shuffle(shuffled)
            selections.extend((str(shot), repeat, shuffled[:shot], seed) for shot in (5, 10))
        for budget, repeat, patients, seed in selections:
            chosen = set(patients)
            selected = [row for row in train if row["patient_key"] in chosen]
            require(len(selected) == 3 * len(patients), "Support must contain three images per patient")
            stem = f"fold_{index:02d}_full" if budget == "full" else f"fold_{index:02d}_shot_{int(budget):03d}_repeat_{repeat:02d}"
            support_path = f"supports/{stem}.csv"
            supports[support_path] = portable(selected)
            for method in METHODS:
                job = {"run_id": stem + "_" + method, "fold": index, "method": method,
                       "budget": budget, "repeat": repeat, "sampling_seed": seed,
                       "patient_order_prefix": patients, "support_csv": support_path,
                       "support": [r["image_name"] for r in selected],
                       "eval": [r["image_name"] for r in heldout]}
                (full if budget == "full" else few).append(job)
    require(len(full) == 10 and len(few) == 100, "Expected 10 full and 100 few-shot runs")
    evaluated = [name for f in fold_records for name in f["eval"]]
    require(sorted(evaluated) == sorted(r["image_name"] for r in rows), "Every image must be held out exactly once")
    return fold_records, supports, full + few


def plan_artifacts(identity):
    rows = {row["image_name"]: row for row in identity["data"]}
    files = {"dataset.csv": identity["data"]}
    for fold in identity["folds"]:
        for split in ("train", "eval"):
            files[f"folds/fold_{fold['fold']:02d}_{split}.csv"] = [rows[n] for n in fold[split]]
    for job in identity["jobs"]:
        files[job["support_csv"]] = [rows[n] for n in job["support"]]
    return files


def validate_formal_identity(identity):
    require(identity.get("version") == VERSION and identity.get("config") == fixed_config(),
            "Formal packaging requires the fixed training protocol")
    rows = identity["data"]
    require(len(rows) == 93 and len({r["patient_key"] for r in rows}) == 31, "Formal cohort must contain 93 images and 31 patients")
    require([r["source_index"] for r in rows] == list(range(93)), "Invalid source row indices")
    names = [r["image_name"] for r in rows]
    require(names == sorted(names) and len(set(names)) == 93, "Invalid source filename order")
    for row in rows:
        match = re.fullmatch(r"([0-9]+)\s*(lt|yz|zz)\.png", row["image_name"], flags=re.IGNORECASE)
        require(match is not None and match.group(1) == row["patient_id"] and match.group(2).lower() == row["label"], "Source filename identity mismatch")
        require(str(int(row["patient_id"])) == row["patient_key"] and LABEL_TO_ID.get(row["label"]) == row["label_id"], "Source patient/label mapping mismatch")
        require(re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]) is not None, "Invalid image fingerprint")
    import hashlib
    digest = hashlib.sha256("".join(r["image_name"] + "\t" + r["image_sha256"] + "\n" for r in rows).encode("utf-8")).hexdigest()
    require(digest == identity.get("dataset_fingerprint") == EXPECTED_FINGERPRINT, "Formal source fingerprint mismatch")
    require(set(identity.get("source_sha256", {})) == set(SOURCE_FILES), "Source-code manifest incomplete")


def verify_plan_artifacts(root, suite):
    identity = suite["identity"]
    require(suite.get("version") == VERSION and suite.get("identity_sha256") == _digest(identity), "Suite identity corrupt")
    expected = plan_artifacts(identity)
    require(set(suite.get("plan_artifact_sha256", {})) == set(expected), "Plan manifest incomplete")
    for name, rows in expected.items():
        path = root / name
        require(path.is_file() and sha256_file(path) == suite["plan_artifact_sha256"][name], f"Plan artifact changed: {name}")
        require(read_csv(path) == [{k: str(v) for k, v in row.items()} for row in rows], f"Plan content mismatch: {name}")
    reconstructed = make_plan(identity["data"], identity["config"])
    require(reconstructed[0] == identity["folds"] and reconstructed[2] == identity["jobs"], "Fold/support plan is not deterministic protocol")


def prepare(root, image_root, *, loader=None, config=None):
    """The CLI uses strict audit and fixed_config; injection is for synthetic tests."""
    root, image_root = Path(root).resolve(), Path(image_root).resolve()
    require(not root.is_relative_to(image_root) and not image_root.is_relative_to(root), "Data and output directories must be disjoint")
    rows, metadata = (loader or audit_dataset)(image_root, strict=True)
    config = fixed_config() if config is None else config
    folds, _, jobs = make_plan(rows, config)
    identity = {"version": VERSION, "config": config, "data": portable(rows),
                "dataset_fingerprint": metadata["dataset_fingerprint"], "folds": folds,
                "jobs": jobs, "source_sha256": source_hashes()}
    path = root / "suite.json"
    if path.exists():
        suite = read_json(path)
        require(suite["identity"] == identity and suite["identity_sha256"] == _digest(identity),
                "Resume rejected: source/data/config/fold/support identity changed")
        verify_plan_artifacts(root, suite)
        return rows, suite
    root.mkdir(parents=True, exist_ok=True)
    require(not [p for p in root.iterdir() if p.name != ".reproduction.lock"], "Output is not empty and has no suite identity")
    artifacts = {}
    for name, content in plan_artifacts(identity).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_csv(target, content)
        artifacts[name] = sha256_file(target)
    suite = {"version": VERSION, "identity": identity, "identity_sha256": _digest(identity),
             "plan_artifact_sha256": artifacts, "created_utc": now(), "status": "prepared_not_run",
             "execution_paths": {"image_root": str(image_root), "output_root": str(root)},
             "runtime_identity": None, "completed_runs": 0, "expected_full_runs": 10, "expected_all_runs": 110,
             "clinical_labels_verified": False, "old_dataset_used": False, "old_splits_modified": False}
    _atomic_json(path, suite)
    verify_plan_artifacts(root, suite)
    return rows, suite


def environment(require_cuda=False):
    import torch
    present = torch.cuda.is_available()
    require(not require_cuda or present, "Formal training requires CUDA; no silent CPU fallback")
    value = _environment()
    value.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                 gpu=torch.cuda.get_device_name(0) if present else None)
    free, total = torch.cuda.mem_get_info() if present else (None, None)
    value.update(gpu_free_mib=free / 1024**2 if free is not None else None,
                 gpu_used_mib=(total - free) / 1024**2 if total is not None else None,
                 gpu_total_mib=total / 1024**2 if total is not None else None)
    return value


def runtime_key(value):
    return {key: value.get(key) for key in ("python", "packages", "cuda", "cudnn", "gpu")}


def configure_torch():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _model_factory(pretrained, dropout):
    from src.models import build_resnet50
    return build_resnet50(pretrained=pretrained, dropout=dropout)


def validate_run(root, suite, job, require_model=True):
    directory = root / "runs" / job["run_id"]
    result = validate_completed(directory, suite["identity_sha256"], job["run_id"], verify_model=require_model)
    if result is None:
        return None
    config = suite["identity"]["config"]
    saved = read_json(directory / "config.json")
    require(saved.get("config") == config and saved.get("job") == job and
            saved.get("identity_sha256") == suite["identity_sha256"], "Run configuration mismatch")
    for key in ("run_id", "fold", "method", "budget", "repeat"):
        require(result.get(key) == job[key], f"Run metadata mismatch: {key}")
    require(saved.get("execution_runtime") == suite["runtime_identity"], "Run runtime mismatch")
    require(saved.get("initial_state_sha256") == result.get("initial_state_sha256"), "Initialization record mismatch")
    require(result.get("n_train_images") == len(job["support"]) and
            result.get("n_train_patients") == len(job["patient_order_prefix"]), "Support counts differ")
    require(result.get("head_stage_backbone_unchanged") is True and result.get("finetune_backbone_changed") is True,
            "Missing two-stage training evidence")
    for before, after in (("bn_before_sha256", "bn_after_sha256"), ("final_state_sha256", "reloaded_state_sha256")):
        require(isinstance(result.get(before), str) and result[before] == result.get(after), "Training fingerprint mismatch")
    require(result.get("initial_backbone_sha256") != result.get("final_backbone_sha256"), "Backbone did not change")
    history = read_csv(directory / "history.csv")
    schedule = [(stage, i + 1) for stage, count in (("head", config["head_rounds"]), ("finetune", config["finetune_rounds"])) for i in range(count)]
    require([(r["stage"], int(r["round"])) for r in history] == schedule, "Training schedule mismatch")
    for row in history:
        lr = config["head_lr"] if row["stage"] == "head" else config["finetune_lr"] * (1 + math.cos(math.pi * (int(row["round"]) - 1) / config["finetune_rounds"])) / 2
        require(int(row["steps"]) == config["steps_per_round"] and int(row["image_occurrences"]) == config["steps_per_round"] * config["batch_size"], "Training update budget mismatch")
        require(math.isclose(float(row["lr"]), lr, rel_tol=1e-9, abs_tol=1e-15) and math.isfinite(float(row["mean_train_loss"])), "Training history invalid")
    steps = len(schedule) * config["steps_per_round"]
    require(result.get("actual_optimizer_steps") == steps and result.get("actual_image_occurrences") == steps * config["batch_size"], "Training evidence budget mismatch")
    by_name = {row["image_name"]: row for row in suite["identity"]["data"]}
    verify_prediction_metrics(directory / "predictions.csv", result, [by_name[n] for n in job["eval"]])
    return result


def train_one(root, suite, job, rows_by_name, device, *, model_factory=None):
    config = suite["identity"]["config"]
    directory = root / "runs" / job["run_id"]
    require(validate_run(root, suite, job) is None, "Completed run cannot be overwritten")
    if directory.exists():
        archive = root / "incomplete_attempts"
        archive.mkdir(exist_ok=True)
        shutil.move(str(directory), str(archive / f"{job['run_id']}_{time.time_ns()}"))
    directory.mkdir(parents=True)
    factory = model_factory or (lambda pretrained: _model_factory(pretrained, config["dropout"]))
    seed_training(config["training_seed"])
    model = factory(True).to(device)
    initial = state_fingerprint(model)
    _atomic_json(directory / "config.json", {"identity_sha256": suite["identity_sha256"], "job": job,
                                            "config": config, "initial_state_sha256": initial,
                                            "execution_runtime": suite["runtime_identity"]})
    history, evidence = train_fixed_updates(model, ImageRows([rows_by_name[n] for n in job["support"]],
                                           get_transforms(job["method"], train=True)), config, device)
    write_csv(directory / "history.csv", history)
    reloaded, checkpoint = save_reload_final(model, directory / "model.pt", lambda: factory(False), device)
    del model
    eval_rows = [rows_by_name[n] for n in job["eval"]]
    probabilities = evaluate(reloaded, ImageRows(eval_rows, get_transforms(job["method"], train=False)), config, device)
    predictions = [{"patient_id": row["patient_id"], "image_name": row["image_name"], "label_id": row["label_id"],
                    "predicted_id": int(prob.argmax()), **{f"prob_{label}": float(prob[i]) for label, i in LABEL_TO_ID.items()}}
                   for row, prob in zip(eval_rows, probabilities)]
    write_csv(directory / "predictions.csv", predictions)
    result = {**{key: job[key] for key in ("run_id", "fold", "method", "budget", "repeat")},
              "n_train_images": len(job["support"]), "n_train_patients": len(job["patient_order_prefix"]),
              "initial_state_sha256": initial, **classification_metrics([r["label_id"] for r in eval_rows], probabilities),
              **evidence, **checkpoint}
    _atomic_json(directory / "result.json", result)
    artifacts = {name: sha256_file(directory / name) for name in (*RUN_FILES[:-1], "model.pt")}
    _atomic_json(directory / "complete.json", {"identity_sha256": suite["identity_sha256"],
                                               "run_id": job["run_id"], "artifact_sha256": artifacts})
    validate_run(root, suite, job)
    return result


def aggregate(root, suite, *, require_models=False):
    """Recompute OOF metrics from probabilities, never pool repeated cohorts."""
    import numpy as np
    results, by_group, predictions, paired = [], {}, [], []
    for job in suite["identity"]["jobs"]:
        result = validate_run(root, suite, job, require_model=require_models)
        if result is None:
            continue
        results.append(result)
        key = (job["fold"], job["budget"], job["repeat"])
        by_group.setdefault(key, {})[job["method"]] = result
        for row in read_csv(root / "runs" / job["run_id"] / "predictions.csv"):
            predictions.append({"fold": job["fold"], "method": job["method"], "budget": job["budget"], "repeat": job["repeat"], **row})
    require(len({r["initial_state_sha256"] for r in results}) <= 1, "Initializations differ between paired runs")
    for (fold, budget, repeat), pair in by_group.items():
        if set(pair) == set(METHODS):
            paired.append({"fold": fold, "budget": budget, "repeat": repeat,
                           **{metric + "_delta": pair["full_frame"][metric] - pair["original"][metric] for metric in METRIC_NAMES}})
    oof = []
    expected_names = sorted(r["image_name"] for r in suite["identity"]["data"])
    for method in METHODS:
        for budget in ("full", "5", "10"):
            for repeat in range(1 if budget == "full" else 5):
                group = [p for p in predictions if (p["method"], p["budget"], p["repeat"]) == (method, budget, repeat)]
                if len({p["fold"] for p in group}) != 5:
                    continue
                require(sorted(p["image_name"] for p in group) == expected_names, "OOF image coverage mismatch")
                probs = np.asarray([[float(p[f"prob_{label}"]) for label in LABEL_TO_ID] for p in group], dtype=np.float32)
                metrics = classification_metrics([int(p["label_id"]) for p in group], probs)
                oof.append({"method": method, "budget": budget, "repeat": repeat, "n_images": len(group),
                            "n_patients": len({p["patient_id"] for p in group}), **metrics})
    oof_pairs, summaries = [], []
    for budget in ("full", "5", "10"):
        for repeat in range(1 if budget == "full" else 5):
            pair = {r["method"]: r for r in oof if (r["budget"], r["repeat"]) == (budget, repeat)}
            if set(pair) == set(METHODS):
                oof_pairs.append({"budget": budget, "repeat": repeat,
                                  **{m + "_delta": pair["full_frame"][m] - pair["original"][m] for m in METRIC_NAMES}})
        for method in METHODS:
            group = [r for r in oof if (r["method"], r["budget"]) == (method, budget)]
            for metric in METRIC_NAMES:
                values = [r[metric] for r in group]
                if values:
                    summaries.append({"method": method, "budget": budget, "metric": metric, "n_oof_repeats": len(values),
                                      "mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
    return {"results": results, "paired": paired, "oof": oof, "predictions": predictions,
            "oof_pairs": oof_pairs, "summaries": summaries}


def write_aggregates(root, suite):
    data = aggregate(root, suite)
    fields = ("run_id", "fold", "method", "budget", "repeat", "n_train_images", "n_train_patients", *METRIC_NAMES)
    write_csv(root / "per_run_metrics.csv", [{k: r[k] for k in fields} for r in data["results"]], fields)
    delta_fields = tuple(m + "_delta" for m in METRIC_NAMES)
    write_csv(root / "paired_comparison.csv", data["paired"], ("fold", "budget", "repeat", *delta_fields))
    write_csv(root / "oof_metrics.csv", data["oof"], ("method", "budget", "repeat", "n_images", "n_patients", *METRIC_NAMES))
    write_csv(root / "oof_predictions.csv", data["predictions"], ("fold", "method", "budget", "repeat", "patient_id", "image_name", "label_id", "predicted_id", "prob_lt", "prob_yz", "prob_zz"))
    write_csv(root / "oof_paired_comparison.csv", data["oof_pairs"], ("budget", "repeat", *delta_fields))
    write_csv(root / "summary_metrics.csv", data["summaries"], ("method", "budget", "metric", "n_oof_repeats", "mean", "sample_sd"))
    (root / "REPORT.md").write_text(report_text(data), encoding="utf-8")
    suite["completed_runs"] = len(data["results"])
    full_count = sum(r["budget"] == "full" for r in data["results"])
    suite["completed_full_runs"] = full_count
    suite["status"] = "completed" if len(data["results"]) == 110 else "full_completed" if full_count == 10 and len(data["results"]) == 10 else "partial"
    suite["aggregate_sha256"] = {name: sha256_file(root / name) for name in AGGREGATES}
    _atomic_json(root / "suite.json", suite)
    return data


def report_text(data):
    lines = ["# renew ResNet-50 患者级开发性交叉验证", "", f"已完成训练：{len(data['results'])}/110。", "",
             "每位患者只在一个折中留出；每份完整OOF结果覆盖同一批31名患者的93张图像。",
             "支持集重复的SD描述抽样波动，不是独立患者的不确定性；不同折的训练患者也有重叠。",
             "full仅有一份OOF结果，不伪造全量SD。合并OOF的AUC使用不同折模型分数，概率尺度可能不同。",
             "固定末轮评价，不根据留出折选择训练轮数或阈值；依据这些结果选方法仍属于开发，不能称为外部验证。",
             "full_frame同时改变视野、有效分辨率、补边和旋转边缘处理；不能单独归因于解剖结构保留。",
             "原截图可能含患者或检查文字；数据和报告仅供本地授权研究使用。", "",
             "| 方法 | 每类预算 | OOF重复数 | 准确率 | 支持抽样SD |", "|---|---|---:|---:|---:|"]
    for row in data["summaries"]:
        if row["metric"] == "accuracy":
            sd = "n/a" if row["sample_sd"] is None else f"{row['sample_sd']:.6f}"
            lines.append(f"| {row['method']} | {row['budget']} | {row['n_oof_repeats']} | {row['mean']:.6f} | {sd} |")
    return "\n".join(lines) + "\n"


def run_suite(args):
    configure_torch()
    import torch
    current_environment = environment(require_cuda=True)
    with output_lock(args.output_root):
        rows, suite = prepare(args.output_root, args.image_root)
        current_runtime = runtime_key(current_environment)
        require(suite["runtime_identity"] in (None, current_runtime), "Runtime changed; use a new output directory")
        suite["runtime_identity"] = current_runtime
        suite["environment"] = current_environment
        suite.setdefault("execution_attempts", []).append({"at_utc": now(), "phase": args.phase, "image_root": str(args.image_root)})
        _atomic_json(args.output_root / "suite.json", suite)
        jobs = suite["identity"]["jobs"]
        if args.phase == "fewshot":
            require(all(validate_run(args.output_root, suite, j) is not None for j in jobs if j["budget"] == "full"), "Complete the ten full-data fits before fewshot")
        selected = [j for j in jobs if args.phase == "all" or (j["budget"] == "full") == (args.phase == "full")]
        by_name = {r["image_name"]: r for r in rows}
        try:
            for i, job in enumerate(selected, 1):
                if validate_run(args.output_root, suite, job) is not None:
                    print(f"[{i}/{len(selected)}] {job['run_id']}: verified completed", flush=True)
                    continue
                print(f"[{i}/{len(selected)}] {job['run_id']}: training {len(job['support'])} images", flush=True)
                train_one(args.output_root, suite, job, by_name, torch.device("cuda"))
                write_aggregates(args.output_root, suite)
                torch.cuda.empty_cache()
            write_aggregates(args.output_root, suite)
        except Exception as exc:
            suite["status"] = "failed"
            suite["last_failure"] = {"at_utc": now(), "type": type(exc).__name__, "message": str(exc)}
            _atomic_json(args.output_root / "suite.json", suite)
            raise


def preview(root, rows):
    from PIL import Image, ImageDraw, ImageOps
    target = root / "preview"
    target.mkdir(exist_ok=True)
    chosen = sorted(rows, key=lambda r: r["image_name"])[:6]
    width, height = 360, 280
    sheets = [("inputs.png", ("original frame", "original eval", "full_frame eval")),
              ("rotation_qa.png", ("full_frame -15 degrees", "full_frame 0 degrees", "full_frame +15 degrees"))]
    for filename, captions in sheets:
        sheet = Image.new("RGB", (width * 3, height * len(chosen) + 42), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "Local QA only: original image text is NOT deidentified.", fill="black")
        for row_index, row in enumerate(chosen):
            with Image.open(row["path"]) as source:
                im = source.convert("RGB")
                values = ([im.copy(), get_geometry_transforms("original")(im.copy()), get_geometry_transforms("full_frame")(im.copy())]
                          if filename == "inputs.png" else [get_geometry_transforms("full_frame", train=True, angle=angle)(im.copy()) for angle in (-15, 0, 15)])
            for column, value in enumerate(values):
                tile = ImageOps.contain(value, (width - 16, height - 42))
                x, y = column * width, row_index * height + 42
                sheet.paste(tile, (x + (width - tile.width) // 2, y + 30 + (height - 42 - tile.height) // 2))
                draw.text((x + 8, y + 8), f"N{row_index + 1:03d} {captions[column]}", fill="black")
        sheet.save(target / filename)
    _atomic_json(target / "preview.json", {"count": len(chosen), "ordering": "first six by filename; no case-based selection",
                                           "images_deidentified": False, "class_labels_shown": False})


def pack_suite(root):
    """Offline packaging validates stored data identities without opening image paths."""
    root = Path(root).resolve()
    with output_lock(root):
        suite = read_json(root / "suite.json")
        validate_formal_identity(suite["identity"])
        verify_plan_artifacts(root, suite)
        data = aggregate(root, suite, require_models=False)
        count = len(data["results"])
        require(count in (10, 110) and sum(r["budget"] == "full" for r in data["results"]) == 10,
                "Pack requires exactly the complete full phase (10) or all 110 fits")
        require(suite.get("completed_runs") == count and suite.get("status") == ("full_completed" if count == 10 else "completed"), "Suite completion state mismatch")
        require(set(suite.get("aggregate_sha256", {})) == set(AGGREGATES), "Aggregate manifest incomplete")
        for name in AGGREGATES:
            require((root / name).is_file() and sha256_file(root / name) == suite["aggregate_sha256"][name], f"Aggregate changed: {name}")
        # Metrics are reconstructed from per-run predictions, not trusted from summaries.
        fields = ("run_id", "fold", "method", "budget", "repeat", "n_train_images", "n_train_patients", *METRIC_NAMES)
        checks = {"oof_metrics.csv": data["oof"], "paired_comparison.csv": data["paired"],
                  "oof_paired_comparison.csv": data["oof_pairs"], "summary_metrics.csv": data["summaries"],
                  "oof_predictions.csv": data["predictions"],
                  "per_run_metrics.csv": [{k: r[k] for k in fields} for r in data["results"]]}
        for name, rows in checks.items():
            require(read_csv(root / name) == [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows], f"Aggregate values do not match predictions: {name}")
        require((root / "REPORT.md").read_text(encoding="utf-8") == report_text(data), "Report values do not match predictions")
        phase = "full" if count == 10 else "all"
        archive = root.parent / f"{root.name}_{phase}_results.tar.gz"
        require(not archive.exists(), f"Refusing to overwrite result package: {archive}")
        names = ["suite.json", *suite["plan_artifact_sha256"], *AGGREGATES]
        for result in data["results"]:
            names.extend(f"runs/{result['run_id']}/{name}" for name in RUN_FILES)
        with tarfile.open(archive, "x:gz") as handle:
            for name in sorted(names):
                path = root / name
                require(path.is_file() and not path.is_symlink(), f"Missing/unsafe packaging input: {name}")
                handle.add(path, arcname=name, recursive=False)
        Path(str(archive) + ".sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n", encoding="utf-8")
        return archive


def main(args):
    if args.mode == "pack":
        print(pack_suite(args.output_root))
    elif args.mode == "run":
        run_suite(args)
    else:
        with output_lock(args.output_root):
            rows, suite = prepare(args.output_root, args.image_root)
            if args.mode == "preview":
                preview(args.output_root, rows)
                print(f"Preview ready: {args.output_root / 'preview'}")
            else:
                info = environment(require_cuda=False)
                _atomic_json(args.output_root / "check_environment.json", info)
                print(f"Data/plan/source verified: {len(rows)} images, 31 patients, 5 folds, 110 planned fits.")
                print(f"GPU: {info['gpu'] or 'not available; check only'}; first run phase: full (10 fits).")
                if info["gpu"]:
                    print(f"Visible GPU memory MiB: used={info['gpu_used_mib']:.0f}, free={info['gpu_free_mib']:.0f}, total={info['gpu_total_mib']:.0f}.")
