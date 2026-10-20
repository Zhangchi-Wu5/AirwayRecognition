"""Probe sensitivity to BatchNorm running statistics without training weights.

Recompute BN buffers on training images only, then compare original/recalibrated
models on the validation set with the same paired synthetic glare perturbations.
This intervention cannot identify or rule out harm from BN during training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from PIL import Image, __version__ as pillow_version
import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision

from scripts.evaluate_glare_pairs import (
    CLASS_ORDER, MODEL_ORDER, PROJECT_ROOT, SCRIPT_VERSION as GLARE_VERSION,
    collect_predictions, compare_models, load_models, sha256_file,
    summarize_predictions, validate_manifest, write_summary,
)
from src.data import get_eval_transforms

SCRIPT_VERSION = "1.0.0"


def validate_split_separation(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Reject both patient overlap and resolved-image overlap before calibration."""
    for name, frame in (("train", train), ("val", val)):
        if frame.empty or not {"patient_id", "path"}.issubset(frame.columns):
            raise ValueError(f"Non-empty {name} manifest requires patient_id and path")
    if set(train.patient_id.astype(str)) & set(val.patient_id.astype(str)):
        raise ValueError("Train/validation patient_id overlap: calibration leakage")
    train_paths = {str(Path(path).expanduser().resolve()) for path in train.path}
    val_paths = {str(Path(path).expanduser().resolve()) for path in val.path}
    if train_paths & val_paths:
        raise ValueError("Train/validation image path overlap: calibration leakage")


def _tensor_hash(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def recalibrate_bn(model, train_manifest: pd.DataFrame, batch_size: int,
                   device, crop_border: bool = False) -> tuple[pd.DataFrame, dict]:
    """Update only BN buffers in a deterministic, label-free, single forward pass.

    Each BN uses momentum=batch_images/cumulative_images, including a short last
    batch. Its running variance is a sample-weighted average of PyTorch's unbiased
    within-batch variances, NOT the exact pooled variance of the entire dataset.
    All modules start in eval mode; only BN runs in train mode (dropout stays off).
    Original training flags and BN momenta are restored, including on failure.
    """
    if batch_size < 1 or train_manifest.empty:
        raise ValueError("Calibration needs training images and positive batch_size")
    if not {"case_id", "path"}.issubset(train_manifest.columns):
        raise ValueError("Calibration manifest requires case_id and path")
    bn_modules = {name: module for name, module in model.named_modules()
                  if isinstance(module, _BatchNorm)}
    if not bn_modules or any(not m.track_running_stats for m in bn_modules.values()):
        raise ValueError("Calibration requires BatchNorm with track_running_stats=True")
    model.to(device)
    modes = {module: module.training for module in model.modules()}
    momenta = {module: module.momentum for module in bn_modules.values()}
    before = {name: (m.running_mean.detach().cpu().clone(),
                     m.running_var.detach().cpu().clone(), int(m.num_batches_tracked))
              for name, m in bn_modules.items()}
    bn_buffer_names = {f"{name}.{buffer}" if name else buffer
                       for name in bn_modules
                       for buffer in ("running_mean", "running_var", "num_batches_tracked")}

    def hashes():
        return (_tensor_hash(model.named_parameters()),
                _tensor_hash((name, tensor) for name, tensor in model.named_buffers()
                             if name not in bn_buffer_names))

    parameter_before, other_before = hashes()
    transform = get_eval_transforms(crop_border=crop_border)
    manifest = train_manifest.sort_values("case_id").reset_index(drop=True)
    seen, n_batches = 0, 0
    try:
        model.eval()
        for module in bn_modules.values():
            module.reset_running_stats()
            module.train()
        with torch.no_grad():
            for start in range(0, len(manifest), batch_size):
                tensors = []
                for path in manifest.iloc[start:start + batch_size].path:
                    with Image.open(path) as opened:
                        tensors.append(transform(opened.convert("RGB")))
                size = len(tensors)
                for module in bn_modules.values():
                    module.momentum = size / (seen + size)
                model(torch.stack(tensors).to(device))
                seen += size
                n_batches += 1
                print(f"BN calibration: {seen}/{len(manifest)} training images", flush=True)
    finally:
        for module, mode in modes.items():
            module.training = mode
        for module, momentum in momenta.items():
            module.momentum = momentum
    parameter_after, other_after = hashes()
    if parameter_before != parameter_after or other_before != other_after:
        raise RuntimeError("BN calibration changed parameters or non-BN buffers")
    rows = []
    for name, module in bn_modules.items():
        mean_before, var_before, batches_before = before[name]
        mean_after, var_after = module.running_mean.detach().cpu(), module.running_var.detach().cpu()
        if not torch.isfinite(mean_after).all() or not torch.isfinite(var_after).all():
            raise ValueError(f"Non-finite calibrated BN statistics: {name}")
        rows.append(dict(module=name, batches_before=batches_before,
                         batches_after=int(module.num_batches_tracked),
                         running_mean_l2_change=float((mean_after - mean_before).norm()),
                         running_var_l2_change=float((var_after - var_before).norm()),
                         running_mean_abs_mean_before=float(mean_before.abs().mean()),
                         running_mean_abs_mean_after=float(mean_after.abs().mean()),
                         running_var_mean_before=float(var_before.mean()),
                         running_var_mean_after=float(var_after.mean())))
    return pd.DataFrame(rows), dict(parameter_sha256_before=parameter_before,
        parameter_sha256_after=parameter_after, non_bn_buffer_sha256_before=other_before,
        non_bn_buffer_sha256_after=other_after, parameters_unchanged=True,
        non_bn_buffers_unchanged=True, n_images=seen, n_batches=n_batches)


def compare_bn_effects(original: pd.DataFrame, recalibrated: pd.DataFrame,
                       n_bootstrap: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per-model after-minus-before contrasts with paired patient bootstrap.

    Common-clean-correct means the SAME model was correct both before and after
    recalibration. This is distinct from the full/baseline common-correct cohort.
    """
    if any(set(frame.model) != set(MODEL_ORDER) for frame in (original, recalibrated)):
        raise ValueError("Both phases must contain baseline and full models")
    comparisons = []
    for name in MODEL_ORDER:
        before = original.loc[original.model == name].assign(model="baseline")
        after = recalibrated.loc[recalibrated.model == name].assign(model="full")
        result = compare_models(pd.concat([before, after], ignore_index=True), n_bootstrap, seed)
        result = result.rename(columns={column: column.replace("baseline_", "before_", 1)
                                        if column.startswith("baseline_") else column.replace("full_", "after_", 1)
                                        for column in result if column.startswith(("baseline_", "full_"))})
        result["comparison"] = "recalibrated-minus-original"
        result.insert(0, "model", name)
        comparisons.append(result)
    return pd.concat(comparisons, ignore_index=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits_patent")
    parser.add_argument("--split", choices=["val"], default="val", help="Only validation is allowed for diagnosis")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coverages", type=float, nargs="+", default=[0, .05, .10, .20])
    parser.add_argument("--placements", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--full-attn-hires", action="store_true")
    parser.add_argument("--no-full-crop-border", dest="full_crop_border", action="store_false", default=True)
    parser.add_argument("--baseline-crop-border", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.placements < 1 or args.bootstrap < 0 or args.seed < 0:
        parser.error("batch-size/placements must be positive; bootstrap/seed must be nonnegative")
    if any(not np.isfinite(c) or not 0 <= c <= 1 for c in args.coverages):
        parser.error("coverages must be finite numbers in [0, 1]")
    if len(set(args.coverages)) != len(args.coverages):
        parser.error("coverages cannot contain duplicates")
    args.coverages = sorted(set([0.0, *args.coverages]))
    return args


def _fingerprint(manifest):
    members = manifest[["case_id", "patient_id", "label", "label_id"]].sort_values("case_id").to_dict("records")
    return hashlib.sha256(json.dumps(members, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_report(path: Path, effects: pd.DataFrame):
    lines = ["# BatchNorm 统计量敏感性诊断", "",
        "仅用训练集图像重算 BN 运行统计量；分类权重、BN 可学习参数及其他缓冲区保持不变。校准不使用标签或验证集图像。",
        "验证集在校准前后使用相同亮斑、位置与预处理；本诊断没有训练或保存新权重。", "",
        "差值均为校准后 − 校准前；准确率正值、共同正确变错率与概率下降负值表示改善。", "",
        "|模型|名义反光档位|准确率差（百分点）|共同正确变错率差（百分点）|变错率差 95% 患者区间|真实类别概率下降差|",
        "|---|---:|---:|---:|---|---:|"]
    metric = "common_clean_correct_to_wrong_rate"
    for _, row in effects.iterrows():
        low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
        interval = "NA" if pd.isna(low) else f"[{100 * low:+.3f}, {100 * high:+.3f}]"
        lines.append(f"|{row.model}|{100 * row.coverage:.1f}%|{100 * row.perturbed_accuracy_difference:+.3f}|"
                     f"{100 * row[f'{metric}_difference']:+.3f}|{interval}|{row.true_probability_drop_mean_difference:+.5f}|")
    lines += ["", "共同正确队列在每个模型内定义：该模型校准前后对干净图都正确。不能与另一个模型的队列混用。",
        "original/ 与 recalibrated/ 的 paired_comparison.csv 各自是完整方案减基线；顶层 bn_effect.csv 是同一模型校准后减校准前。",
        "不要把两个不同共同正确队列的差值相减，当成 BN 与模型方案的因果交互效应。", "",
        "若校准后恢复，只能说明当前权重对 BN 统计量敏感；若没有恢复，也不能排除训练期 BN 影响已进入权重。",
        "校准采用固定图像排序和批大小；逐批无偏方差的样本加权均值不是全数据精确合并方差，结果可能随批划分改变。",
        "真正检验训练期原因仍需固定训练种子与其他设置，重训只改变 BN 策略的对照。勿根据本次验证反复挑选测试集表现。", "",
        "合成近白方块是压力测试；它会遮挡解剖结构，不代表真实反光成像。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("output-dir must be absent or empty; existing results will not be overwritten")
    split_paths = {name: args.splits_dir / f"{name}.csv" for name in ("train", "val")}
    manifests = {name: validate_manifest(pd.read_csv(path, dtype=str), args.image_root)
                 for name, path in split_paths.items()}
    validate_split_separation(manifests["train"], manifests["val"])
    checkpoint_hashes = {"baseline": sha256_file(args.baseline_checkpoint), "full": sha256_file(args.full_checkpoint)}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = load_models(args, device)
    crop = {"baseline": args.baseline_crop_border, "full": args.full_crop_border}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(phase):
        print(f"Validation phase: {phase}", flush=True)
        predictions = collect_predictions(models, manifests["val"], args.coverages,
            args.placements, args.seed, args.batch_size, device, crop)
        summary = summarize_predictions(predictions)
        comparison = compare_models(predictions, args.bootstrap, args.seed)
        destination = args.output_dir / phase
        destination.mkdir()
        predictions.to_csv(destination / "per_image_predictions.csv", index=False)
        summary.to_csv(destination / "summary.csv", index=False)
        comparison.to_csv(destination / "paired_comparison.csv", index=False)
        write_summary(destination / "summary.md", summary, comparison)
        return predictions

    original = evaluate("original")
    changes, integrity = [], {}
    for name in MODEL_ORDER:
        print(f"Recalibrating {name} BN from TRAIN images only", flush=True)
        frame, integrity[name] = recalibrate_bn(models[name], manifests["train"], args.batch_size, device, crop[name])
        changes.append(frame.assign(model=name))
    recalibrated = evaluate("recalibrated")
    effects = compare_bn_effects(original, recalibrated, args.bootstrap, args.seed)
    pd.concat(changes, ignore_index=True).to_csv(args.output_dir / "bn_changes.csv", index=False)
    effects.to_csv(args.output_dir / "bn_effect.csv", index=False)
    _write_report(args.output_dir / "summary.md", effects)
    metadata = dict(completed=True, script_version=SCRIPT_VERSION, glare_evaluator_version=GLARE_VERSION,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        device=str(device), class_order=CLASS_ORDER, checkpoint_sha256=checkpoint_hashes,
        splits={name: dict(csv_sha256=sha256_file(split_paths[name]), membership_sha256=_fingerprint(frame),
                          n_images=len(frame), n_patients=frame.patient_id.nunique()) for name, frame in manifests.items()},
        patient_and_resolved_path_overlap_checked=True, integrity=integrity,
        calibration=dict(source="train.csv only", labels_used=False, augmentation="none",
            preprocessing="model-specific crop flag; Resize(256), CenterCrop(224), ImageNet normalization",
            order="case_id sorted", dropout="eval", batchnorm="train, running stats reset, no gradients",
            momentum="batch_images / cumulative_images (short last batch sample-weighted)",
            variance="sample-weighted mean of within-batch unbiased variances, not exact pooled population variance",
            mutable_state=["BN.running_mean", "BN.running_var", "BN.num_batches_tracked"],
            checkpoint_saved=False),
        comparison=dict(evaluation_split="val", perturbations="Same evaluate_glare_pairs synthetic near-white squares, seed and placements before/after",
            phase_paired_comparison="full-minus-baseline within each phase",
            bn_effect="recalibrated-minus-original separately for each model",
            common_clean_correct="Within bn_effect only: SAME model correct on clean images before AND after calibration",
            direction="Positive accuracy difference favors recalibration; negative error/drop difference favors recalibration",
            rates="image-placement weighted; zero-coverage one observation/image; zero denominator NaN",
            bootstrap="paired patient-cluster percentile 95%, all images and placements retained; no CI for <2 patients or bootstrap=0",
            bootstrap_replicates=args.bootstrap),
        versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
            numpy=np.__version__, pandas=pd.__version__, pillow=pillow_version),
        limitations=["A BN-buffer intervention probes sensitivity; neither recovery nor non-recovery proves or excludes a training-time cause.",
            "Calibration depends on batch grouping and model-specific preprocessing; changing it is a diagnostic intervention, not a neutral correction.",
            "No retraining, checkpoint replacement, test-set selection or claim of robustness improvement.",
            "Different common-clean-correct cohorts must not be subtracted to claim a causal interaction.",
            "Synthetic glare may obscure diagnostic anatomy; patient identifiers remain in local prediction files."])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {args.output_dir.resolve()} (validation-only BN diagnostic)", flush=True)


if __name__ == "__main__":
    main()
