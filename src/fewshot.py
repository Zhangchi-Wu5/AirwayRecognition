"""Auditable fixed-class, patient-disjoint few-shot linear probing.

Planning uses only the standard library. Numeric/image dependencies are loaded
only when an experiment is actually run. Test images are never opened.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import warnings

LABEL_TO_ID = {"lt": 0, "yz": 1, "zz": 2}
PROTOCOL_VERSION = "resnet50_frozen_fewshot_v1"
LINEAR_CONFIG = {"solver": "lbfgs", "penalty": "l2", "C": 1.0, "max_iter": 5000, "tol": 1e-8}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fields = fieldnames or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _identity_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "label", "label_id", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty split")
    seen_paths, seen_ids = set(), set()
    result = []
    for index, raw in enumerate(rows):
        if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in required):
            raise ValueError(f"{path}: missing identity or label at row {index + 2}")
        row = {key: raw[key].strip() for key in ("patient_id", "label", "label_id", "path")}
        if row["label"] not in LABEL_TO_ID or row["label_id"] != str(LABEL_TO_ID[row["label"]]):
            raise ValueError(f"{path}: inconsistent label mapping at row {index + 2}")
        row["label_id"] = LABEL_TO_ID[row["label"]]
        row["image_name"] = row["path"].replace("\\", "/").rsplit("/", 1)[-1]
        identity = (row["patient_id"], row["label"], row["image_name"])
        canonical = row["path"].replace("\\", "/")
        if canonical in seen_paths or identity in seen_ids:
            raise ValueError(f"{path}: duplicate image identity/path")
        seen_paths.add(canonical)
        seen_ids.add(identity)
        row["source_index"] = index
        result.append(row)
    return result


def load_splits(splits_dir, image_root=None):
    """Check CSV identities in all splits; read files only in train and val.

    An explicit image_root rebases each CSV path to image_root/basename. Relative
    paths without rebasing are interpreted relative to the split CSV directory.
    """
    splits_dir = Path(splits_dir).expanduser().resolve()
    root = None if image_root is None else Path(image_root).expanduser().resolve()
    frames = {name: _identity_rows(splits_dir / f"{name}.csv") for name in ("train", "val", "test")}
    for name, rows in frames.items():
        for row in rows:
            original = Path(row["path"]).expanduser()
            candidate = root / row["image_name"] if root is not None else original
            if not candidate.is_absolute():
                candidate = splits_dir / candidate
            row["original_path"] = row["path"]
            row["path"] = str(candidate.resolve())
        if len({row["path"] for row in rows}) != len(rows):
            raise ValueError(f"Resolved duplicate path in {name}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key, description in (("patient_id", "Patient"), ("path", "Resolved image path"), ("original_path", "Image path")):
            if {r[key] for r in frames[left]} & {r[key] for r in frames[right]}:
                raise ValueError(f"{description} overlap between {left} and {right}")
    metadata = {}
    for name, rows in frames.items():
        if name in ("train", "val") and {r["label"] for r in rows} != set(LABEL_TO_ID):
            raise ValueError(f"{name}: all three classes required")
        members = [{key: row[key] for key in ("patient_id", "label", "label_id", "image_name")} for row in rows]
        members.sort(key=lambda row: (row["patient_id"], row["label"], row["image_name"]))
        metadata[name] = {
            "csv_sha256": sha256_file(splits_dir / f"{name}.csv"),
            "membership_sha256": hashlib.sha256(json.dumps(members, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "n_images": len(rows), "n_patients": len({r["patient_id"] for r in rows}),
            "class_counts": {label: sum(r["label"] == label for r in rows) for label in LABEL_TO_ID},
            "members": members,
            "image_access": "identity_only_no_image_access" if name == "test" else "hashed_for_provenance",
        }
        if name != "test":
            for row in rows:
                if not Path(row["path"]).is_file():
                    raise FileNotFoundError(f"Missing {name} image: {row['path']}")
                row["image_sha256"] = sha256_file(row["path"])
            metadata[name]["images"] = [{key: r[key] for key in ("source_index", "patient_id", "label", "path", "image_sha256")} for r in rows]
    train_hashes = {r["image_sha256"] for r in frames["train"]}
    if train_hashes & {r["image_sha256"] for r in frames["val"]}:
        raise ValueError("Identical image content appears in train and val")
    return frames, metadata


def make_support_plan(train, shots=(1, 5, 10, 20), repeats=20, sampling_seed=20261022):
    """Use a per-repeat/class patient permutation and nested K prefixes.

    Missing-class patients remain eligible in classes they have. When a patient
    has multiple images in one class, choose exactly one before taking prefixes.
    """
    shots = sorted(set(shots))
    if not shots or any(type(k) is not int or k < 1 for k in shots):
        raise ValueError("shots must contain positive integers")
    if type(repeats) is not int or repeats < 2 or type(sampling_seed) is not int or sampling_seed < 0:
        raise ValueError("repeats >= 2 and nonnegative integer sampling_seed required")
    groups = {}
    for label in LABEL_TO_ID:
        patients = {}
        for row in train:
            if row["label"] == label:
                patients.setdefault(row["patient_id"], []).append(row)
        if len(patients) < max(shots):
            raise ValueError(f"{label}: {len(patients)} patients cannot supply {max(shots)}-shot")
        groups[label] = patients
    runs = []
    for repeat in range(repeats):
        ordered = {}
        for label, patients in groups.items():
            seed_text = f"{PROTOCOL_VERSION}:{sampling_seed}:{repeat}:{label}"
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest(), "big"))
            patient_ids = sorted(patients)
            rng.shuffle(patient_ids)
            ordered[label] = [rng.choice(sorted(patients[p], key=lambda r: r["image_name"])) for p in patient_ids]
        for shot in shots:
            support = [row for label in LABEL_TO_ID for row in ordered[label][:shot]]
            runs.append({"run_id": f"shot_{shot:03d}_repeat_{repeat:03d}", "budget": str(shot),
                         "shot": shot, "repeat": repeat, "rows": support})
    runs.append({"run_id": "full", "budget": "full", "shot": None, "repeat": None, "rows": list(train)})
    return runs


def _environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow", "matplotlib", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "git_commit": commit, "git_dirty": dirty,
            "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in
                              ("src/fewshot.py", "scripts/run_resnet_fewshot.py")}}


def prepare_experiment(args):
    """Validate everything before creating an empty, new output directory."""
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty output: {output}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size >= 1 and num_workers >= 0 required")
    frames, metadata = load_splits(args.splits_dir, args.image_root)
    runs = make_support_plan(frames["train"], args.shots, args.repeats, args.sampling_seed)
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "status": "prepared_not_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "fixed_three_class_patient_disjoint_low_label_transfer_screening",
        "evaluation_split": "val", "test_images_accessed": False,
        "label_mapping": LABEL_TO_ID, "shots": sorted(set(args.shots)), "repeats": args.repeats,
        "sampling_seed": args.sampling_seed, "seed_role": "support_selection_only_no_optimizer_randomness",
        "sampling": "per_repeat_per_class_patient_permutation_one_image_per_patient_nested_K_prefixes",
        "full_reference_runs": 1,
        "pretrained": {"architecture": "resnet50", "weights": "ResNet50_Weights.IMAGENET1K_V2",
                       "source": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
                       "local_finetuned_checkpoint_allowed": False, "backbone": "fully_frozen_eval",
                       "feature": "2048D_global_average_pool_before_fc"},
        "preprocessing": {"resize_short_edge": 256, "center_crop": 224, "interpolation": "bilinear", "antialias": True,
                          "RGB": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                          "black_border_removal": False, "augmentation": None},
        "feature_normalization": "each_image_L2_only_no_dataset_statistics",
        "linear_head": dict(LINEAR_CONFIG), "class_weight": None,
        "label_budget_scope": "only_selected_training_labels_fit_head; all_train_labels_used_for_stratified_sampling; all_val_labels_used_for_exploratory_evaluation",
        "selection_policy": "fixed_hyperparameters_no_validation_selection_no_checkpoint_selection",
        "uncertainty": "support_sampling_mean_sample_SD_min_max_on_same_fixed_val_patients_not_independent_patient_CI",
        "historical_98_percent": "different_full_finetuning_protocol_not_this_frozen_linear_probe",
        "device_requested": args.device, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "splits": metadata, "environment": _environment(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "supports").mkdir()
    summaries = []
    for run in runs:
        name = f"supports/{run['run_id']}.csv"
        write_csv(output / name, run["rows"])
        row = {"run_id": run["run_id"], "budget": run["budget"], "shot": run["shot"], "repeat": run["repeat"],
               "n_train_images": len(run["rows"]), "n_train_patients": len({r["patient_id"] for r in run["rows"]}),
               "support_csv": name, "support_csv_sha256": sha256_file(output / name)}
        for label in LABEL_TO_ID:
            row[f"n_{label}"] = sum(r["label"] == label for r in run["rows"])
            row[f"patients_{label}"] = len({r["patient_id"] for r in run["rows"] if r["label"] == label})
        summaries.append(row)
    write_csv(output / "support_plan.csv", summaries)
    write_json(output / "protocol.json", protocol)
    (output / "REPORT.md").write_text(
        "# ResNet 小样本可行性筛查\n\n状态：支持集计划已生成，实验尚未运行，暂无性能结论。\n\n"
        "固定 ImageNet 预训练 ResNet-50，冻结主干，仅拟合线性分类头。"
        "这与历史约 98% 的全量微调方案不同。测试集仅核验身份，不读取测试图像。\n", encoding="utf-8")
    return frames, runs, protocol


class _ImageDataset:
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.rows[index]["path"]) as image:
            return self.transform(image.convert("RGB"))


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def l2_normalize(features):
    import numpy as np
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Features must be a finite matrix")
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero feature vector cannot be normalized")
    return features / norm


def extract_features(frames, args):
    """Construct only official external weights; never accept local checkpoints."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights, resnet50
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        selected = args.device
    device = torch.device(selected)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    fingerprint = state_fingerprint(model)
    model.fc = nn.Identity()
    model.requires_grad_(False).eval().to(device)
    backbone_before = state_fingerprint(model)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    extracted = {}
    started = time.perf_counter()
    for split in ("train", "val"):
        print(f"Extracting {split}: {len(frames[split])} images on {device}", flush=True)
        loader = DataLoader(_ImageDataset(frames[split], transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        chunks = []
        with torch.inference_mode():
            for batch in loader:
                chunks.append(model(batch.to(device)).cpu().numpy())
        extracted[split] = l2_normalize(np.concatenate(chunks))
        if extracted[split].shape != (len(frames[split]), 2048):
            raise ValueError(f"Unexpected ResNet feature shape for {split}")
    backbone_after = state_fingerprint(model)
    if backbone_before != backbone_after:
        raise RuntimeError("Frozen backbone state changed during extraction")
    return extracted, {"device_used": str(device), "pretrained_state_sha256": fingerprint,
                       "backbone_state_before_sha256": backbone_before,
                       "backbone_state_after_sha256": backbone_after,
                       "backbone_state_unchanged": True,
                       "extraction_seconds": time.perf_counter() - started,
                       "torch_deterministic_algorithms": True}


def fit_head(features, labels):
    """Fit only the passed support labels, using fixed deterministic settings."""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    if set(np.asarray(labels).tolist()) != set(LABEL_TO_ID.values()):
        raise ValueError("Support must contain all three classes")
    model = LogisticRegression(**LINEAR_CONFIG)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(features, labels)
    messages = [str(w.message) for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or max(model.n_iter_) >= LINEAR_CONFIG["max_iter"]:
        raise RuntimeError("Linear head did not converge; run failed without retuning: " + " | ".join(messages))
    return model, {"converged": True, "n_iter": int(max(model.n_iter_)), "fit_seconds": time.perf_counter() - started,
                   "warnings": " | ".join(messages)}


def classification_metrics(labels, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score, roc_auc_score
    predicted = probabilities.argmax(axis=1)
    recalls = recall_score(labels, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(labels, predicted)),
               "macro_f1": float(f1_score(labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
               "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
               "auc_ovr_macro": float(roc_auc_score(labels, probabilities, labels=[0, 1, 2], average="macro", multi_class="ovr")),
               "loss": float(log_loss(labels, probabilities, labels=[0, 1, 2]))}
    metrics.update({f"recall_{label}": float(recalls[label_id]) for label, label_id in LABEL_TO_ID.items()})
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite evaluation metric")
    return metrics


METRIC_NAMES = ("accuracy", "macro_f1", "balanced_accuracy", "auc_ovr_macro", "loss", "recall_lt", "recall_yz", "recall_zz")


def summarize_runs(results):
    import statistics
    summaries = []
    for budget in sorted({r["budget"] for r in results if r["budget"] != "full"}, key=int) + ["full"]:
        group = [r for r in results if r["budget"] == budget]
        if not group:
            continue
        if budget == "full" and len(group) != 1:
            raise ValueError("Full reference must run once")
        for metric in METRIC_NAMES:
            values = [r[metric] for r in group]
            summaries.append({"budget": budget, "metric": metric, "n_support_draws": len(group),
                              "mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                              "min": min(values), "max": max(values)})
    return summaries


def write_report(output, results, summaries, protocol):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_fewshot_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summaries if r["metric"] == "accuracy"]
    few = [r for r in rows if r["budget"] != "full"]
    full = next(r for r in rows if r["budget"] == "full")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sampled = [r for r in results if r["budget"] != "full"]
    ax.scatter([int(r["budget"]) for r in sampled], [100 * r["accuracy"] for r in sampled],
               s=17, alpha=.3, color="#386cb0", label="Individual support draws")
    ax.errorbar([int(r["budget"]) for r in few], [100 * r["mean"] for r in few],
                yerr=[100 * r["sample_sd"] for r in few], fmt="o-", capsize=5,
                label="Frozen ResNet-50 + linear head (mean +/- sample SD)")
    ax.axhline(100 * full["mean"], linestyle="--", color="#a64b2a", label="Same protocol, full training set (one fit)")
    ax.set(xlabel="Labeled training images per class", ylabel="Validation accuracy (%)", xticks=[int(r["budget"]) for r in few])
    ax.set_ylim(0, 102)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(.5, .01, "Support-draw variation on fixed validation patients; not a patient confidence interval.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "learning_curve.png", dpi=200)
    fig.savefig(output / "learning_curve.pdf")
    plt.close(fig)
    lines = ["# ResNet 小样本可行性筛查", "", "状态：实验完成（探索性验证集评估）。", "",
             "ImageNet1K_V2 ResNet-50 全部冻结并使用 eval 模式；2048 维特征逐图 L2 归一化。"
             "固定 L2 逻辑回归线性头（C=1，未调优），不使用验证集选择参数或训练轮次。"
             "这与历史约 98% 的全量微调结果不是同一实验协议。", "",
             "| 每类标注数 | 支持集抽样次数 | 准确率均值 | 抽样标准差 | 最低—最高 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        sd = "不适用" if row["sample_sd"] is None else f"{100 * row['sample_sd']:.2f} 个百分点"
        label = "全量（同协议参考）" if row["budget"] == "full" else row["budget"]
        lines.append(f"| {label} | {row['n_support_draws']} | {100 * row['mean']:.2f}% | {sd} | {100 * row['min']:.2f}%—{100 * row['max']:.2f}% |")
    lines += ["", "每一重复、每一类别先随机排列训练患者，每位患者最多选一张该类图片，1/5/10/20 等预算采用同一排列的前缀。"
              "缺少某一类别的患者仍可贡献其拥有类别的图片，不要求每位患者三类齐全。不同类可以使用同一患者。"
              "具体患者数和支持集成员见 support_plan.csv 与 supports/。", "",
              "标准差和最小—最大值仅表示支持集抽样引起的波动，所有重复共享同一批验证患者，不能当作独立病例的置信区间。"
              "全量参考只拟合一次，不伪造重复。仅支持集标签用于各次分类头拟合；训练标签用于分层抽样，验证标签用于本轮探索性评估。", "",
              "测试集 CSV 仅用于身份和患者隔离校验，未读取测试图像。当前验证集曾用于既往探索，最终结论仍需未参与设计的患者。"
              "如果低预算表现下降，可据此评估标注效率研究；如果各预算已较高，应考虑更具临床意义的泛化条件。", "",
              "若低样本和本轮冻结主干的全量参考均明显低于历史全量微调成绩，差异首先提示适配方式或固定正则化可能限制性能，"
              "不能直接归因于标注不足。小样本下降应首先相对于本轮同协议的全量参考判断。", "",
              "其他指标及波动见 summary_metrics.csv；逐图预测见 predictions.csv；分类头和特征缓存分别位于 heads/ 与 features/。"
              "protocol.json 记录预训练权重指纹、图像/代码哈希、依赖环境及运行状态。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute_experiment(args, frames, runs, protocol, extractor=None):
    """Run prepared in-memory plan; extractor injection is for offline tests."""
    output = Path(args.output_root).expanduser().resolve()
    started = time.perf_counter()
    protocol["status"] = "running"
    write_json(output / "protocol.json", protocol)
    current_run = None
    try:
        import numpy as np
        features, extraction_info = (extractor or extract_features)(frames, args)
        protocol["runtime"] = extraction_info
        (output / "features").mkdir()
        (output / "heads").mkdir()
        for split in ("train", "val"):
            features[split] = l2_normalize(features[split])
            if len(features[split]) != len(frames[split]):
                raise ValueError(f"Feature count does not match {split}")
            rows = frames[split]
            np.savez_compressed(output / "features" / f"{split}.npz", features=features[split],
                                patient_id=np.array([r["patient_id"] for r in rows]),
                                image_name=np.array([r["image_name"] for r in rows]),
                                source_index=np.array([r["source_index"] for r in rows]),
                                label_id=np.array([r["label_id"] for r in rows]),
                                image_sha256=np.array([r["image_sha256"] for r in rows]))
        val_labels = np.array([r["label_id"] for r in frames["val"]])
        results, predictions = [], []
        for number, run in enumerate(runs, 1):
            current_run = run["run_id"]
            indices = [row["source_index"] for row in run["rows"]]
            support_labels = np.array([row["label_id"] for row in run["rows"]])
            head, convergence = fit_head(features["train"][indices], support_labels)
            if head.classes_.tolist() != [0, 1, 2]:
                raise ValueError("Unexpected classifier probability column order")
            probabilities = head.predict_proba(features["val"])
            metrics = classification_metrics(val_labels, probabilities)
            record = {"run_id": current_run, "budget": run["budget"], "shot": run["shot"], "repeat": run["repeat"],
                      "n_train_images": len(indices), "n_train_patients": len({r["patient_id"] for r in run["rows"]}),
                      **metrics, **convergence}
            for label in LABEL_TO_ID:
                record[f"n_{label}"] = sum(r["label"] == label for r in run["rows"])
            results.append(record)
            np.savez_compressed(output / "heads" / f"{current_run}.npz", coef=head.coef_, intercept=head.intercept_,
                                classes=head.classes_, n_iter=head.n_iter_, support_indices=np.asarray(indices))
            for row, probability in zip(frames["val"], probabilities):
                predictions.append({"run_id": current_run, "budget": run["budget"], "repeat": run["repeat"],
                                    "patient_id": row["patient_id"], "image_name": row["image_name"],
                                    "label_id": row["label_id"], "predicted_id": int(probability.argmax()),
                                    **{f"prob_{label}": float(probability[label_id]) for label, label_id in LABEL_TO_ID.items()}})
            write_csv(output / "per_run_metrics.csv", results)
            print(f"[{number}/{len(runs)}] {current_run}: accuracy={metrics['accuracy']:.4f}, iterations={convergence['n_iter']}", flush=True)
        write_csv(output / "predictions.csv", predictions)
        summaries = summarize_runs(results)
        write_csv(output / "summary_metrics.csv", summaries)
        write_report(output, results, summaries, protocol)
        protocol["runtime"]["total_seconds"] = time.perf_counter() - started
        protocol["runtime"]["completed_runs"] = len(results)
        protocol["status"] = "completed"
        protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
        protocol["artifact_sha256"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*"))
                                       if p.is_file() and p.name != "protocol.json"}
        write_json(output / "protocol.json", protocol)
        return results, summaries
    except Exception as exc:
        protocol["status"] = "failed"
        protocol["failure"] = {"run_id": current_run, "exception": type(exc).__name__, "message": str(exc),
                               "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "protocol.json", protocol)
        (output / "REPORT.md").write_text(
            "# ResNet 小样本可行性筛查\n\n状态：失败，实验未完成，不应据此作完整性能结论。\n\n"
            f"失败位置：{current_run or 'feature extraction'}；错误：{exc}\n\n详见 protocol.json。\n", encoding="utf-8")
        raise
