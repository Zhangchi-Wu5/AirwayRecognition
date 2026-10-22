"""Run both prespecified ResNet comparisons in one server checkout."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fewshot import PROJECT_ROOT, load_splits, make_support_plan, sha256_file, write_json
from src.fewshot_finetune import prepare as prepare_finetune, validate_completed, verify_reference

SOURCE_FILES = ("src/fewshot.py", "src/fewshot_finetune.py", "src/data.py", "src/models.py",
                "scripts/run_resnet_fewshot.py", "scripts/run_resnet_fewshot_finetune.py",
                "scripts/run_resnet_fewshot_suite.py", "scripts/run_fewshot_suite_server.sh")
AGGREGATES = ("REPORT.md", "per_run_metrics.csv", "summary_metrics.csv", "paired_frozen_comparison.csv",
              "learning_curve.png", "learning_curve.pdf")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "pack"))
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/resnet_fewshot_suite_20261022")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size >= 1 and num-workers >= 0 required")
    for name in ("splits_dir", "image_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def check_environment():
    """Exercise the numerical dependencies and selected GPU without training."""
    packages = {name: importlib.import_module(name) for name in
                ("torch", "torchvision", "numpy", "scipy", "sklearn", "PIL", "matplotlib", "pandas", "threadpoolctl")}
    torch = packages["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; activate the server environment and check CUDA_VISIBLE_DEVICES. No CPU fallback.")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    from sklearn.linear_model import LogisticRegression
    with packages["threadpoolctl"].threadpool_limits(limits=1):
        LogisticRegression(max_iter=100).fit([[0, 0], [1, 0], [0, 1], [2, 2], [1, 2], [2, 1]], [0, 1, 2, 0, 1, 2])
    print(f"CUDA ready: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}", flush=True)
    return {"packages": {name: getattr(module, "__version__", None) for name, module in packages.items()},
            "python": sys.version, "executable": sys.executable, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}


def check_data(args):
    frames, metadata = load_splits(args.splits_dir, args.image_root)
    runs = make_support_plan(frames["train"], [1, 5, 10, 20], 20, 20261022)
    print(f"Data verified: train={len(frames['train'])}, val={len(frames['val'])}; {len(runs)} identical supports per method; test images unopened.", flush=True)
    return frames, metadata


@contextmanager
def suite_lock(output):
    """An OS lock is released automatically even if the process is killed."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".suite.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another suite run or pack process already owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def suite_identity(args, metadata):
    return {"version": "resnet_fewshot_server_suite_v1", "device": "cuda",
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "splits_dir": str(args.splits_dir), "image_root": str(args.image_root),
            "sources": {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES},
            "data": metadata}


def ensure_identity(args, metadata):
    path = args.output_root / "suite.json"
    identity = suite_identity(args, metadata)
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError("Suite source, data or configuration changed; use a new --output-root.")
        return saved
    unexpected = [p for p in args.output_root.iterdir() if p.name != ".suite.lock"]
    if unexpected:
        raise ValueError("Output already contains files without a suite identity; use a new --output-root.")
    saved = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_not_run"}
    write_json(path, saved)
    return saved


def verify_frozen(args, frames, metadata):
    """Only skip a completed frozen phase if every original artifact is intact."""
    root = args.output_root / "frozen"
    protocol = read_json(root / "protocol.json")
    runs, _ = verify_reference(root, frames, metadata)
    for key, value in (("device_requested", "cuda"), ("batch_size", args.batch_size), ("num_workers", args.num_workers)):
        if protocol.get(key) != value:
            raise ValueError(f"Frozen configuration changed: {key}")
    expected_sources = {name: sha256_file(PROJECT_ROOT / name) for name in
                        ("src/fewshot.py", "scripts/run_resnet_fewshot.py")}
    if protocol["environment"].get("source_sha256") != expected_sources:
        raise ValueError("Frozen source changed; use a new output directory")
    expected = {"REPORT.md", "learning_curve.pdf", "learning_curve.png", "per_run_metrics.csv", "predictions.csv",
                "summary_metrics.csv", "support_plan.csv", "features/train.npz", "features/val.npz"}
    expected.update(f"supports/{run['run_id']}.csv" for run in runs)
    expected.update(f"heads/{run['run_id']}.npz" for run in runs)
    artifacts = protocol.get("artifact_sha256", {})
    if set(artifacts) != expected:
        raise ValueError("Frozen artifact manifest is incomplete or unexpected")
    for name, checksum in artifacts.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Frozen artifact corrupted: {name}")
    return protocol


def phase_command(args, phase):
    common = ["--splits-dir", str(args.splits_dir), "--image-root", str(args.image_root),
              "--device", "cuda", "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)]
    if phase == "frozen":
        return [sys.executable, "-u", "-m", "scripts.run_resnet_fewshot", *common,
                "--output-root", str(args.output_root / "frozen"),
                "--shots", "1", "5", "10", "20", "--repeats", "20", "--sampling-seed", "20261022"]
    command = [sys.executable, "-u", "-m", "scripts.run_resnet_fewshot_finetune", *common,
               "--reference-root", str(args.output_root / "frozen"), "--output-root", str(args.output_root / phase)]
    if phase == "smoke":
        command.append("--smoke-test")
    if (args.output_root / phase / "protocol.json").exists():
        command.append("--resume")
    return command


def run_phase(args, phase):
    """Keep a phase log even when the launching terminal disconnects."""
    with (args.output_root / f"{phase}.log").open("a", encoding="utf-8") as log:
        command = phase_command(args, phase)
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


def finetune_args(args, phase="finetune"):
    return SimpleNamespace(reference_root=args.output_root / "frozen", splits_dir=args.splits_dir,
                           image_root=args.image_root, output_root=args.output_root / phase,
                           device="cuda", batch_size=args.batch_size, num_workers=args.num_workers,
                           resume=True, run_ids=None, smoke_test=phase == "smoke")


def verify_finetune(args, phase="finetune"):
    _, runs, protocol = prepare_finetune(finetune_args(args, phase))
    required_status = "smoke_completed" if phase == "smoke" else "completed"
    if protocol.get("status") != required_status or protocol.get("completed_runs") != len(runs):
        raise ValueError(f"{phase} is not complete")
    for run in runs:
        if validate_completed(args.output_root / phase / "runs" / run["run_id"], protocol["identity_sha256"], run["run_id"], verify_model=False) is None:
            raise ValueError(f"Missing completed {phase} run: {run['run_id']}")
    return protocol


def aggregate_hashes(args):
    return {name: sha256_file(args.output_root / "finetune" / name) for name in AGGREGATES}


def run_suite(args, frames, metadata):
    with suite_lock(args.output_root):
        saved = ensure_identity(args, metadata)
        runtime = getattr(args, "runtime", {"packages": {}, "python": sys.version})
        if "runtime" in saved and any(saved["runtime"].get(key) != runtime.get(key) for key in ("packages", "python", "cuda")):
            raise ValueError("Runtime package/Python/CUDA versions changed; restore the original environment or use a new output root.")
        saved.setdefault("runtime", runtime)
        saved.setdefault("execution_attempts", []).append({"started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime})
        saved["status"] = "running"
        write_json(args.output_root / "suite.json", saved)
        frozen = args.output_root / "frozen"
        protocol_path = frozen / "protocol.json"
        try:
            if protocol_path.exists() and read_json(protocol_path).get("status") == "completed":
                verify_frozen(args, frames, metadata)
                print("Frozen phase verified; reuse completed server result.", flush=True)
            else:
                if frozen.exists():
                    archive = args.output_root / "incomplete_attempts"
                    archive.mkdir(exist_ok=True)
                    shutil.move(str(frozen), str(archive / f"frozen_{time.time_ns()}"))
                run_phase(args, "frozen")
                verify_frozen(args, frames, metadata)
            for phase in ("smoke", "finetune"):
                run_phase(args, phase)
                verify_finetune(args, phase)
            saved.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(), formal_fits=162,
                         finetune_aggregate_sha256=aggregate_hashes(args))
            write_json(args.output_root / "suite.json", saved)
            print(f"Both formal phases completed. Pack with: bash scripts/run_fewshot_suite_server.sh pack --output-root {args.output_root}", flush=True)
        except Exception as exc:
            saved.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            write_json(args.output_root / "suite.json", saved)
            raise


def pack_results(args, frames, metadata):
    with suite_lock(args.output_root):
        saved = ensure_identity(args, metadata)
        if saved.get("status") != "completed":
            raise ValueError("Cannot pack: suite has not completed both formal phases")
        verify_frozen(args, frames, metadata)
        verify_finetune(args)
        if saved.get("finetune_aggregate_sha256") != aggregate_hashes(args):
            raise ValueError("Fine-tuning aggregate report/table/plot hashes changed or are missing")
        archive = args.output_root.with_name(args.output_root.name + "_results.tar.gz")
        if archive.exists():
            raise ValueError(f"Result archive already exists; move it before packing again: {archive}")
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        try:
            with tarfile.open(temporary, "w:gz") as bundle:
                for path in sorted(args.output_root.rglob("*")):
                    relative = path.relative_to(args.output_root)
                    if not path.is_file() or path.is_symlink() or path.suffix.lower() in (".pt", ".pth"):
                        continue
                    if "incomplete_attempts" in relative.parts or relative.parts[0] == "smoke" or path.name == ".suite.lock":
                        continue
                    bundle.add(path, arcname=str(Path(args.output_root.name) / relative), recursive=False)
            temporary.replace(archive)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        checksum = sha256_file(archive)
        archive.with_suffix(archive.suffix + ".sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
        print(f"Results packed (model weights retained on server): {archive}\nSHA256: {checksum}", flush=True)
        return archive


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.mode != "pack":
        args.runtime = check_environment()
    frames, metadata = check_data(args)
    if args.mode == "check":
        print("Preflight passed. No formal training or experiment outputs created.", flush=True)
    elif args.mode == "run":
        run_suite(args, frames, metadata)
    else:
        pack_results(args, frames, metadata)


if __name__ == "__main__":
    main()
