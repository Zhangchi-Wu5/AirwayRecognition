"""An isolated, resumable full-data comparison using unchanged training code.

The historical and fixed-update trainers run in separate processes so their
global PyTorch precision and determinism settings cannot contaminate each other.
Only the validation split is evaluated. No few-shot method is selected here.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

from src.fewshot import (PROJECT_ROOT, LABEL_TO_ID, _environment, classification_metrics,
                        load_splits, sha256_file, write_csv)
from src.fewshot_finetune import (_atomic_json, _digest, _portable_rows, make_config,
                                 validate_completed, verify_reference)

VERSION = "resnet_full_baseline_reproduction_v1"
JOBS = ("historical_42", "historical_2026", "historical_3407", "current_2026", "current_3407")
SEEDS = (42, 2026, 3407)
OWN_SOURCES = ("src/baseline_reproduction.py", "scripts/run_resnet_baseline_reproduction.py",
               "scripts/run_baseline_reproduction_server.sh")
CURRENT_SOURCES = ("src/fewshot.py", "src/fewshot_finetune.py", "src/data.py", "src/models.py",
                   "scripts/run_resnet_fewshot.py", "scripts/run_resnet_fewshot_finetune.py")
REFERENCE_FILES = ("config.json", "result.json", "predictions.csv", "history.csv", "complete.json")
AGGREGATE_FILES = ("per_run_metrics.csv", "summary_metrics.csv", "paired_seed_comparison.csv", "REPORT.md")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_environment():
    import torch
    import torchvision
    import numpy
    import pandas
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on the server; there is no CPU training fallback.")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    environment = _environment()
    environment.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(0), pandas=pandas.__version__,
                       thread_environment={name: os.environ.get(name) for name in
                                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                       torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads())
    return environment


def runtime_identity(environment):
    return {key: environment.get(key) for key in ("python", "packages", "cuda", "cudnn", "gpu", "pandas",
                                                 "thread_environment", "torch_num_threads", "torch_num_interop_threads")}


def verify_reference_runtime(identity, environment):
    """Retained seed 42 must remain comparable to the two newly seeded fits."""
    old = identity["current_reference"]["environment"]
    require(environment["python"] == old["python"] and environment["packages"] == old["packages"],
            "Environment differs from retained seed 42; activate its original Python/package environment")
    require(environment["gpu"] == old["gpu"], "GPU model differs from retained seed 42; use its original GPU model")


def _sources(reference):
    for name, expected in reference["source_sha256"].items():
        path = (PROJECT_ROOT / name).resolve()
        require(path.is_relative_to(PROJECT_ROOT) and path.is_file() and sha256_file(path) == expected,
                f"Historical training source differs: {name}. Restore the audited source before reproduction.")
    names = set(reference["source_sha256"]) | set(OWN_SOURCES) | set(CURRENT_SOURCES)
    return {name: sha256_file(PROJECT_ROOT / name) for name in sorted(names)}


def make_current_config(args, n_train, seed):
    config = make_config(SimpleNamespace(smoke_test=False, batch_size=16, num_workers=args.num_workers,
                                        device="cuda"), n_train)
    # This is a NEW protocol, not a mutation of a prepared/completed experiment.
    config["training_seed"] = seed
    return config


def verify_current_reference(args, frames, metadata):
    root = args.reference_root
    runs, _ = verify_reference(root / "frozen", frames, metadata)
    protocol = read_json(root / "finetune/protocol.json")
    require(protocol.get("status") == "completed" and protocol.get("mode") == "formal",
            "Reference fine-tuning suite is not formally completed")
    require(protocol.get("config") == make_current_config(args, len(frames["train"]), 42),
            "Current seed-42 reference configuration differs; preserve its worker count and full-data protocol")
    require(protocol.get("identity_sha256") == _digest(protocol.get("identity")), "Reference protocol identity hash mismatch")
    for name, expected in protocol["identity"]["source_sha256"].items():
        require(sha256_file(PROJECT_ROOT / name) == expected, f"Current reference source differs: {name}")
    for split in ("train", "val", "test"):
        require(protocol["identity"]["data"][split]["membership_sha256"] == metadata[split]["membership_sha256"],
                f"Current reference {split} membership mismatch")
        if split != "test":
            require(protocol["identity"]["data"][split]["ordered_image_identities"] == _portable_rows(frames[split]),
                    f"Current reference {split} image/order/hash mismatch")
    directory = root / "finetune/runs/full"
    result = validate_completed(directory, protocol["identity_sha256"], "full", verify_model=(directory / "model.pt").exists())
    require(result is not None, "Current reference full run is incomplete")
    config = read_json(directory / "config.json")
    require(config["config"] == protocol["config"] and config["identity_sha256"] == protocol["identity_sha256"],
            "Current reference full configuration/identity mismatch")
    require(config["support"] == _portable_rows(frames["train"]), "Current reference full support differs")
    verify_prediction_metrics(directory / "predictions.csv", result, frames["val"])
    hashes = {name: sha256_file(directory / name) for name in REFERENCE_FILES}
    hashes["protocol.json"] = sha256_file(root / "finetune/protocol.json")
    return {"artifact_sha256": hashes, "identity_sha256": protocol["identity_sha256"],
            "config": protocol["config"], "environment": protocol["environment_execution"],
            "result": result}, next(run for run in runs if run["run_id"] == "full")


def check_inputs(args):
    require(args.output_root != args.reference_root and not args.output_root.is_relative_to(args.reference_root)
            and not args.reference_root.is_relative_to(args.output_root), "New output and existing reference must be disjoint")
    reference = read_json(args.historical_reference)
    sources = _sources(reference)
    frames, metadata = load_splits(args.splits_dir, args.image_root)
    old_frames, old_metadata = load_splits(args.historical_splits_dir, args.image_root)
    csv_exact = {}
    for split in ("train", "val", "test"):
        old = reference["splits"][split]
        require(old_metadata[split]["membership_sha256"] == old["membership_sha256"],
                f"Historical {split} membership mismatch")
        require(metadata[split]["membership_sha256"] == old["membership_sha256"],
                f"Current {split} membership mismatch")
        if split != "test":
            identities = lambda rows: sorted((r["patient_id"], r["label"], r["image_name"], r["image_sha256"]) for r in rows)
            require(identities(frames[split]) == identities(old_frames[split]), f"Historical/current {split} images differ")
        csv_exact[split] = sha256_file(args.historical_splits_dir / f"{split}.csv") == old["csv_sha256"]
    current, full = verify_current_reference(args, frames, metadata)
    identity = {"version": VERSION, "seeds": list(SEEDS), "jobs": list(JOBS), "batch_size": 16,
                "num_workers": args.num_workers, "sources": sources,
                "historical_reference_sha256": sha256_file(args.historical_reference),
                "current_reference": current, "data": metadata, "historical_data": old_metadata,
                "historical_csv_bytes_match": csv_exact,
                "row_order_unverified": not all(csv_exact.values()),
                "historical_splits_dir": str(args.historical_splits_dir),
                "splits_dir": str(args.splits_dir), "image_root": str(args.image_root)}
    return frames, full, identity, reference


def verify_prediction_metrics(path, result, val_rows=None):
    import numpy as np
    rows = read_csv(path)
    require(bool(rows), "Empty validation predictions")
    if val_rows is not None:
        expected = [(r["patient_id"], r["image_name"], int(r["label_id"])) for r in val_rows]
        actual = [(r["patient_id"], r["image_name"], int(r["label_id"])) for r in rows]
        require(actual == expected, "Prediction identity/order mismatch")
    # Both trainers evaluate float32 softmax. Preserve its clipping epsilon for
    # log_loss when restoring serialized probabilities, including exact 0/1.
    probabilities = np.array([[float(row[f"prob_{label}"]) for label in LABEL_TO_ID] for row in rows], dtype=np.float32)
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and (probabilities <= 1).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5), "Invalid probabilities")
    require([int(row["predicted_id"]) for row in rows] == probabilities.argmax(1).tolist(), "Predicted labels differ from probabilities")
    metrics = classification_metrics([int(row["label_id"]) for row in rows], probabilities)
    for name, value in metrics.items():
        # sklearn releases differ in clipping/normalizing float32 log-loss.
        # Eight epsilons (<1e-6) permit portable offline numeric verification.
        tolerance = 8 * float(np.finfo(np.float32).eps) if name == "loss" else 1e-9
        require(name in result and math.isclose(float(result[name]), value, rel_tol=1e-9, abs_tol=tolerance),
                f"Prediction metric mismatch: {name}")
    return metrics


@contextmanager
def output_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".reproduction.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another run/pack process holds this output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_suite(args, identity, environment, reference):
    root = args.output_root
    path = root / "suite.json"
    if path.exists():
        suite = read_json(path)
        require(suite["identity"] == identity and suite["identity_sha256"] == _digest(identity),
                "Resume rejected: source/data/reference/config changed; use a new output")
        require(suite["runtime_identity"] == runtime_identity(environment), "Resume rejected: runtime environment changed")
        verify_copied_reference(root, suite)
        return suite
    require(not [p for p in root.iterdir() if p.name != ".reproduction.lock"], "Output is not empty and has no suite identity")
    target = root / "references/current_seed_42"
    target.mkdir(parents=True)
    for name in REFERENCE_FILES:
        shutil.copyfile(args.reference_root / "finetune/runs/full" / name, target / name)
    _atomic_json(target / "provenance.json", identity["current_reference"])
    _atomic_json(root / "references/historical.json", reference)
    reference_hashes = {str(path.relative_to(root)): sha256_file(path) for path in sorted((root / "references").rglob("*")) if path.is_file()}
    suite = {"version": VERSION, "identity": identity, "identity_sha256": _digest(identity),
             "environment": environment, "runtime_identity": runtime_identity(environment),
             "reference_artifact_sha256": reference_hashes, "status": "prepared_not_run",
             "created_utc": datetime.now(timezone.utc).isoformat(), "test_images_accessed": False}
    _atomic_json(path, suite)
    return suite


def verify_copied_reference(root, suite):
    for name, expected in suite["reference_artifact_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and sha256_file(path) == expected,
                f"Copied reference corrupted: {name}")
    source = root / "references/current_seed_42"
    verify_prediction_metrics(source / "predictions.csv", read_json(source / "result.json"))


def job_directory(root, job):
    require(job in JOBS, "Unexpected job")
    return root / "runs" / job


def complete_job(directory, suite, job):
    files = {str(path.relative_to(directory)): sha256_file(path)
             for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "job_complete.json"}
    _atomic_json(directory / "job_complete.json", {"job": job, "identity_sha256": suite["identity_sha256"], "artifact_sha256": files})


def validate_job(root, suite, job, require_weights=True):
    directory = job_directory(root, job)
    path = directory / "job_complete.json"
    if not path.exists():
        return None
    complete = read_json(path)
    require(complete.get("job") == job and complete.get("identity_sha256") == suite["identity_sha256"], "Job identity mismatch")
    artifacts = complete.get("artifact_sha256", {})
    expected = {"result.json", "predictions.csv"}
    if job.startswith("historical_"):
        seed = job.rsplit("_", 1)[1]
        expected.update({"training/protocol.json", f"training/seed_{seed}/pair_metadata.json"})
        expected.update(f"training/seed_{seed}/baseline/{name}" for name in
                        ("run_config.json", "history.csv", "training_input_trace.csv", "best_val.pt", "last_model.pt"))
        expected.add("runtime_flags.json")
    else:
        expected.add("protocol.json")
        expected.update(f"full/{name}" for name in ("config.json", "history.csv", "model.pt", "predictions.csv", "result.json", "complete.json"))
    require(set(artifacts) == expected, "Incomplete or unexpected job manifest")
    for name, expected in artifacts.items():
        artifact = (directory / name).resolve()
        require(artifact.is_relative_to(directory.resolve()), "Unsafe artifact path")
        if artifact.suffix in (".pt", ".pth") and not artifact.exists() and not require_weights:
            continue
        require(artifact.is_file() and sha256_file(artifact) == expected, f"Completed job corrupted/missing: {job}/{name}")
    result = read_json(directory / "result.json")
    require(result.get("job") == job and result.get("identity_sha256") == suite["identity_sha256"], "Result identity mismatch")
    require(result.get("test_images_accessed") is False, "Unexpected test evaluation")
    bn_field = "batchnorm_buffers_unchanged" if job.startswith("historical_") else "bn_buffers_unchanged"
    require(result.get("checkpoint_reload_verified") is True and result.get(bn_field) is True,
            "Missing checkpoint reload or BatchNorm invariance evidence")
    verify_prediction_metrics(directory / "predictions.csv", result)
    return result


def worker_command(args, job):
    command = [sys.executable, "-u", "-m", "scripts.run_resnet_baseline_reproduction", "_worker", "--job", job]
    for name in ("splits_dir", "historical_splits_dir", "image_root", "reference_root", "historical_reference", "output_root", "num_workers"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return command


def run_subprocess(args, job):
    log_dir = args.output_root / "logs"
    log_dir.mkdir(exist_ok=True)
    command = worker_command(args, job)
    with (log_dir / f"{job}.log").open("a", encoding="utf-8") as log:
        log.write(f"\nStarted {datetime.now(timezone.utc).isoformat()}\n{json.dumps(command)}\n")
        log.flush()
        with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def historical_arguments(args, seed, directory):
    return ["--splits-dir", str(args.historical_splits_dir), "--image-root", str(args.image_root),
            "--output-root", str(directory / "training"), "--seeds", str(seed), "--arms", "baseline",
            "--skip-evaluation", "--device", "cuda", "--batch-size", "16", "--num-workers", str(args.num_workers),
            "--stage1-epochs", "5", "--stage2-epochs", "20", "--stage1-lr", "0.001", "--stage2-lr", "0.0001",
            "--weight-decay", "0.0001", "--dropout", "0.3"]


def train_historical(args, suite, job, directory):
    import torch
    import numpy as np
    from scripts import run_region_relation_ablation as runner
    from src.region_candidates import CandidateConfig
    from src.region_relations import build_region_relation_resnet50
    seed = int(job.rsplit("_", 1)[1])
    training_args = runner.parse_args(historical_arguments(args, seed, directory))
    reference = read_json(args.historical_reference)
    for key, value in reference["common_training_arguments"].items():
        require(getattr(training_args, key) == value, f"Historical training argument differs: {key}")
    _atomic_json(directory / "runtime_flags.json", {
        "when": "before_original_runner", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "note": "Original historical flags were not recorded; preserve this process's defaults, then invoke the original set_seed."})
    # Original runner and global torch defaults are deliberately preserved.
    runner.run(training_args)
    arm = directory / "training" / f"seed_{seed}" / "baseline"
    config = read_json(arm / "run_config.json")
    train, val, _ = runner.load_existing_splits(args.historical_splits_dir, args.image_root)
    device = torch.device("cuda")
    _, loader = runner.make_loaders(train, val, training_args, seed, device, CandidateConfig())
    model = build_region_relation_resnet50(**{**config["model_config"], "pretrained": False})
    checkpoint = torch.load(arm / "last_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            images, _, candidates = runner.move_batch(batch, device)
            probabilities.append(model(images, candidates).softmax(1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predictions = [{"patient_id": str(row.patient_id), "image_name": Path(row.path).name,
                    "label_id": int(row.label_id), "predicted_id": int(prob.argmax()),
                    **{f"prob_{label}": float(prob[index]) for label, index in LABEL_TO_ID.items()}}
                   for row, prob in zip(val.itertuples(index=False), probabilities)]
    write_csv(directory / "predictions.csv", predictions)
    metrics = classification_metrics(val.label_id.to_numpy(), probabilities)
    require(math.isclose(metrics["accuracy"], config["last_val_accuracy"], abs_tol=1e-12), "Historical reload accuracy changed")
    original = reference["runs"][str(seed)]
    trace_match = config["training_input_traces"] == original["training_input_traces"]
    initial_match = config["initial_state_sha256"] == original["initial_state_sha256"]
    result = {"job": job, "method": "historical_epoch", "seed": seed, "identity_sha256": suite["identity_sha256"],
              **metrics, "initial_state_matches_history": initial_match,
              "final_state_matches_history": config["final_state_sha256"] == original["final_state_sha256"],
              "training_input_stream_matches_history": trace_match,
              "historical_csv_bytes_match": suite["identity"]["historical_csv_bytes_match"],
              "row_order_unverified": not trace_match,
              "reproduction_scope": "initialization_and_input_stream_verified" if trace_match and initial_match else "same_membership_reproduction",
              "historical_accuracy": original["last_val_accuracy"],
              "historical_accuracy_difference": metrics["accuracy"] - original["last_val_accuracy"],
              "completed_epochs": config["completed_epochs"], "checkpoint_reload_verified": config["checkpoint_reload_verified"],
              "batchnorm_buffers_unchanged": config["batchnorm_buffers_unchanged"], "test_images_accessed": False}
    _atomic_json(directory / "result.json", result)


def train_current(args, suite, job, directory):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from src.fewshot_finetune import _train_run
    seed = int(job.rsplit("_", 1)[1])
    frames, metadata = load_splits(args.splits_dir, args.image_root)
    runs, reference_hashes = verify_reference(args.reference_root / "frozen", frames, metadata)
    full = next(run for run in runs if run["run_id"] == "full")
    config = make_current_config(args, len(frames["train"]), seed)
    identity = {"protocol_version": VERSION + "_current_full", "suite_identity_sha256": suite["identity_sha256"],
                "config": config, "data": metadata, "reference_sha256": reference_hashes,
                "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in CURRENT_SOURCES}, "run_ids": ["full"]}
    protocol = {"protocol_version": identity["protocol_version"], "identity": identity, "identity_sha256": _digest(identity),
                "config": config, "environment_execution": _environment(), "test_images_accessed": False, "status": "running"}
    _atomic_json(directory / "protocol.json", protocol)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = directory / "full"
    run_dir.mkdir()
    result = _train_run(SimpleNamespace(), frames, full, protocol, run_dir, torch.device("cuda"))
    shutil.copyfile(run_dir / "predictions.csv", directory / "predictions.csv")
    _atomic_json(directory / "result.json", {**result, "job": job, "method": "current_fixed_updates", "seed": seed,
                                             "identity_sha256": suite["identity_sha256"], "test_images_accessed": False})
    protocol["status"] = "completed"
    _atomic_json(directory / "protocol.json", protocol)


def run_worker(args):
    if args.job.startswith("current_"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    suite = read_json(args.output_root / "suite.json")
    require(suite["identity_sha256"] == _digest(suite["identity"]), "Suite identity corruption")
    # Recheck actual inputs per process, including between jobs in one long run.
    _, _, identity, _ = check_inputs(args)
    require(identity == suite["identity"], "Worker arguments/data/reference/source changed")
    require(runtime_identity(check_environment()) == suite["runtime_identity"], "Worker runtime drift")
    directory = job_directory(args.output_root, args.job)
    require(not directory.exists(), "Worker output already exists")
    directory.mkdir(parents=True)
    if args.job.startswith("historical_"):
        train_historical(args, suite, args.job, directory)
    else:
        train_current(args, suite, args.job, directory)
    complete_job(directory, suite, args.job)
    validate_job(args.output_root, suite, args.job)


def aggregate(root, suite):
    verify_copied_reference(root, suite)
    rows = []
    for job in JOBS:
        result = validate_job(root, suite, job)
        require(result is not None, f"Cannot summarize incomplete job: {job}")
        rows.append(result)
    old = read_json(root / "references/current_seed_42/result.json")
    rows.append({**old, "method": "current_fixed_updates", "seed": 42, "job": "current_42_referenced",
                 "test_images_accessed": False})
    columns = ("method", "seed", "job", "accuracy", "macro_f1", "balanced_accuracy", "historical_accuracy",
               "historical_accuracy_difference", "reproduction_scope", "initial_state_matches_history", "final_state_matches_history", "training_input_stream_matches_history")
    write_csv(root / "per_run_metrics.csv", [{key: row.get(key, "") for key in columns} for row in rows])
    summary = []
    for method in ("historical_epoch", "current_fixed_updates"):
        matched = [row for row in rows if row["method"] == method]
        summary.append({"method": method, "n_seeds": len(matched),
                        "accuracy_mean": statistics.mean(row["accuracy"] for row in matched),
                        "accuracy_sd": statistics.stdev(row["accuracy"] for row in matched),
                        "macro_f1_mean": statistics.mean(row["macro_f1"] for row in matched),
                        "macro_f1_sd": statistics.stdev(row["macro_f1"] for row in matched)})
    write_csv(root / "summary_metrics.csv", summary)
    paired = []
    for seed in SEEDS:
        historical = next(row for row in rows if row["method"] == "historical_epoch" and row["seed"] == seed)
        current = next(row for row in rows if row["method"] == "current_fixed_updates" and row["seed"] == seed)
        paired.append({"seed": seed, "historical_accuracy": historical["accuracy"], "current_accuracy": current["accuracy"],
                       "accuracy_current_minus_historical": current["accuracy"] - historical["accuracy"],
                       "historical_macro_f1": historical["macro_f1"], "current_macro_f1": current["macro_f1"],
                       "macro_f1_current_minus_historical": current["macro_f1"] - historical["macro_f1"]})
    write_csv(root / "paired_seed_comparison.csv", paired)
    text = ["# 全量强基线复现", "", "5 次新训练完成，当前流程 seed 42 引用已完成实验。全部指标来自固定验证集；测试图像未读取。",
            "", "| 流程 | 验证准确率均值 ± 跨种子标准差 |", "|---|---:|"]
    for item in summary:
        text.append(f"| {item['method']} | {item['accuracy_mean']:.2%} ± {100*item['accuracy_sd']:.2f} 个百分点 |")
    text += ["", "## 复现边界", "", f"历史 CSV 字节一致性：{suite['identity']['historical_csv_bytes_match']}。",
             "CSV 路径变化也会改变字节哈希；只有实际训练输入流哈希一致才能确认增强图像及顺序一致。"]
    for row in rows:
        if row["method"] == "historical_epoch":
            text.append(f"- seed {row['seed']}：初始化一致={row['initial_state_matches_history']}；训练输入流一致={row['training_input_stream_matches_history']}；最终权重一致={row['final_state_matches_history']}；准确率相对历史差异={100*row['historical_accuracy_difference']:+.2f} 个百分点。")
    text += ["", "## 相同种子的流程对照", "", "| seed | 当前固定更新 − 原 epoch 流程准确率 |", "|---|---:|"]
    for row in paired:
        text.append(f"| {row['seed']} | {100*row['accuracy_current_minus_historical']:+.2f} 个百分点 |")
    text += ["", "标准差表示三个训练种子的波动，不是独立患者的置信区间。历史与当前流程仍存在采样/增强随机过程及批次组成差异；结果不是单因素消融。",
             "", "此阶段停止于全量复现；应先解释结果，再预先确定 5-shot / 10-shot 比较方法，不能按验证集效果反复重跑直到达到 98%。", ""]
    (root / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    suite["aggregate_sha256"] = {name: sha256_file(root / name) for name in AGGREGATE_FILES}
    return rows


def run_suite(args):
    frames, full, identity, reference = check_inputs(args)
    environment = check_environment()
    verify_reference_runtime(identity, environment)
    with output_lock(args.output_root):
        suite = prepare_suite(args, identity, environment, reference)
        suite["status"] = "running"
        _atomic_json(args.output_root / "suite.json", suite)
        try:
            for job in JOBS:
                if validate_job(args.output_root, suite, job) is not None:
                    print(f"{job}: verified, already completed", flush=True)
                    continue
                directory = job_directory(args.output_root, job)
                if directory.exists():
                    archive = args.output_root / "incomplete_attempts"
                    archive.mkdir(exist_ok=True)
                    shutil.move(str(directory), str(archive / f"{job}_{time.time_ns()}"))
                print(f"Starting {job}", flush=True)
                run_subprocess(args, job)
                require(validate_job(args.output_root, suite, job) is not None, "Worker did not complete")
            aggregate(args.output_root, suite)
            suite["status"] = "completed"
            suite.pop("failure", None)
            suite["completed_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(args.output_root / "suite.json", suite)
        except Exception as exc:
            suite["status"] = "failed"
            suite["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            _atomic_json(args.output_root / "suite.json", suite)
            raise
    print(f"All full-data comparisons completed. Pack with: bash scripts/run_baseline_reproduction_server.sh pack --output-root {args.output_root}", flush=True)


def pack_results(args):
    root = args.output_root
    require((root / "suite.json").is_file(), "No suite found to pack")
    with output_lock(root):
        suite = read_json(root / "suite.json")
        require(suite.get("status") == "completed" and suite["identity_sha256"] == _digest(suite["identity"]), "Suite incomplete or identity corrupted")
        verify_copied_reference(root, suite)
        for job in JOBS:
            require(validate_job(root, suite, job, require_weights=False) is not None, f"Incomplete job: {job}")
        require(set(suite.get("aggregate_sha256", {})) == set(AGGREGATE_FILES), "Aggregate manifest incomplete")
        for name, expected in suite["aggregate_sha256"].items():
            require(sha256_file(root / name) == expected, f"Aggregate changed: {name}")
        archive = root.parent / f"{root.name}_results.tar.gz"
        checksum = Path(str(archive) + ".sha256")
        require(not archive.exists() and not checksum.exists(), f"Archive already exists: {archive}")
        paths = [path for path in sorted(root.rglob("*")) if path.is_file()
                 and "incomplete_attempts" not in path.relative_to(root).parts
                 and path.suffix not in (".pt", ".pth") and not path.name.startswith(".")]
        require(not any(path.is_symlink() for path in paths), "Refusing symlink in result package")
        temporary = archive.with_suffix(".gz.tmp")
        try:
            with tarfile.open(temporary, "w:gz") as handle:
                for path in paths:
                    handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
            temporary.replace(archive)
            checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Results: {archive}\nChecksum: {checksum}", flush=True)
    return archive


def main(args):
    if args.mode == "check":
        frames, _, identity, _ = check_inputs(args)
        environment = check_environment()
        verify_reference_runtime(identity, environment)
        print(f"Ready: train={len(frames['train'])}, val={len(frames['val'])}; five NEW fits; existing seed 42 retained; GPU={environment['gpu']}")
        print(f"Historical CSV byte matches: {identity['historical_csv_bytes_match']}; training input streams will be compared after fitting.")
    elif args.mode == "run":
        run_suite(args)
    elif args.mode == "pack":
        pack_results(args)
    else:
        run_worker(args)
