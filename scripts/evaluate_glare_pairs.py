"""Paired synthetic white-patch stress test of baseline and full checkpoints.

Run with ``python -m scripts.evaluate_glare_pairs --help``. Every pair shares the
same perturbation in ORIGINAL RGB coordinates. No training or image export occurs.
The square patches are an artificial stress test, not an optical glare simulator.
"""
import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from PIL import Image, __version__ as pillow_version
import torch
import torchvision

from src.attention import build_attention_resnet50
from src.data import CropBlackBorder, ID_TO_LABEL, LABEL_TO_ID, get_eval_geometry_transforms, get_eval_transforms
from src.models import build_resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = [ID_TO_LABEL[i] for i in range(3)]
MODEL_ORDER = ("baseline", "full")
SCRIPT_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: pd.DataFrame, image_root: Path | None = None) -> pd.DataFrame:
    """Validate labels/patients and resolve unique images; preserve leading-zero IDs."""
    required = {"patient_id", "label", "label_id", "path"}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError(f"Non-empty split CSV requires columns: {sorted(required)}")
    frame = manifest.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("Split CSV contains missing identifiers, labels or paths")
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Empty {column} in split CSV")
    label_ids = pd.to_numeric(frame["label_id"], errors="coerce")
    expected = frame["label"].map(LABEL_TO_ID)
    if expected.isna().any() or not label_ids.eq(expected).all():
        raise ValueError("label and label_id must agree with lt=0, yz=1, zz=2")
    frame["label_id"] = label_ids.astype(int)
    if image_root is not None and not Path(image_root).is_dir():
        raise ValueError(f"Image root is not a directory: {image_root}")
    paths, cases = [], []
    for row in frame.itertuples(index=False):
        basename = str(row.path).replace("\\", "/").rsplit("/", 1)[-1]
        path = (Path(image_root) / basename if image_root is not None else Path(row.path)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Image does not exist: {path}; use --image-root to relocate a flat dataset")
        paths.append(str(path))
        identity = json.dumps([row.patient_id, row.label, basename], ensure_ascii=False, separators=(",", ":"))
        cases.append(hashlib.sha256(identity.encode()).hexdigest()[:24])
    if len(set(paths)) != len(paths) or len(set(cases)) != len(cases):
        raise ValueError("Duplicate image path or case identity after path resolution")
    frame["path"], frame["case_id"] = paths, cases
    return frame.reset_index(drop=True)


def stable_seed(seed: int, case_id: str, placement: int) -> int:
    """Seed independent of CSV order, batch size, model, and coverage level."""
    payload = json.dumps([int(seed), str(case_id), int(placement)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def clean_crop_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    return CropBlackBorder().get_bbox(image)


def valid_field_mask(image: Image.Image) -> np.ndarray:
    """Original-resolution field excluding dark pixels connected to an image edge."""
    from src.viz import detect_dark_border_mask
    return ~detect_dark_border_mask(image)


def make_glare_pair(image: Image.Image, coverage: float, seed: int,
                    valid_mask: np.ndarray | None = None) -> tuple[Image.Image, np.ndarray]:
    """Stamp 252/255 squares; overlapping/clipped patches reduce actual coverage.

    Requested area is relative to the ORIGINAL image area. Square side scales as
    sqrt(width*height)*20/224. Centers lie in the valid field, and pixels outside
    it are never modified. For a given seed, increasing coverage adds patches to
    the same sequence, so masks are nested (rounding can produce equal masks).
    """
    if not np.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("coverage must be finite and in [0, 1]")
    rgb = np.asarray(image.convert("RGB")).copy()
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if coverage == 0:
        return Image.fromarray(rgb), mask
    valid = valid_field_mask(image) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != mask.shape or not valid.any():
        raise ValueError("A non-empty valid field matching the original image is required")
    side = max(1, int(round(np.sqrt(h * w) * 20 / 224)))
    count = max(1, int(round(coverage * h * w / (side * side))))
    centers = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    for index in rng.choice(centers, size=count, replace=True):
        y, x = divmod(int(index), w)
        y0, x0 = y - side // 2, x - side // 2
        mask[max(0, y0):min(h, y0 + side), max(0, x0):min(w, x0 + side)] = True
    mask &= valid
    rgb[mask] = 252
    return Image.fromarray(rgb), mask


def fixed_preprocess(image: Image.Image, bbox=None) -> torch.Tensor:
    """Use the clean-image bbox, then the repository's exact evaluation transform."""
    if bbox is not None:
        image = image.crop(bbox)
    return get_eval_transforms(crop_border=False)(image)


def input_mask_coverage(mask: np.ndarray, bbox=None) -> float:
    """Fractional area after identical crop/resize; bilinear edges count fractionally."""
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    if bbox is not None:
        image = image.crop(bbox)
    transformed = get_eval_geometry_transforms(crop_border=False)(image)
    return float(np.asarray(transformed, dtype=np.float64).mean() / 255)


def _infer(model, tensors, device) -> np.ndarray:
    with torch.inference_mode():
        probs = torch.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu().numpy()
    if probs.shape != (len(tensors), 3) or not np.isfinite(probs).all():
        raise ValueError("Model must produce finite three-class logits")
    return probs


def _prediction_row(base, model_name, clean, probs) -> dict:
    row = dict(base, model=model_name)
    truth = int(row["true_label_id"])
    clean_pred, perturbed_pred = int(clean.argmax()), int(probs.argmax())
    clean_ok, perturbed_ok = clean_pred == truth, perturbed_pred == truth
    row.update(clean_pred=CLASS_ORDER[clean_pred], perturbed_pred=CLASS_ORDER[perturbed_pred],
               clean_pred_id=clean_pred, perturbed_pred_id=perturbed_pred,
               clean_correct=int(clean_ok), perturbed_correct=int(perturbed_ok),
               flipped=int(clean_pred != perturbed_pred),
               correct_to_wrong=int(clean_ok and not perturbed_ok),
               wrong_to_correct=int(not clean_ok and perturbed_ok),
               true_probability_delta=float(probs[truth] - clean[truth]),
               true_probability_drop=float(clean[truth] - probs[truth]),
               probability_l1_change=float(np.abs(probs - clean).sum()))
    for i, label in enumerate(CLASS_ORDER):
        row[f"clean_prob_{label}"] = float(clean[i])
        row[f"prob_{label}"] = float(probs[i])
        row[f"delta_prob_{label}"] = float(probs[i] - clean[i])
    return row


def collect_predictions(models: Mapping, manifest: pd.DataFrame, coverages, placements: int,
                        seed: int, batch_size: int, device, crop_border: Mapping) -> pd.DataFrame:
    """Evaluate clean once per image/model, then paired perturbations in bounded batches."""
    if set(models) != set(MODEL_ORDER) or set(crop_border) != set(MODEL_ORDER):
        raise ValueError("models and crop_border must contain baseline and full")
    if batch_size < 1 or placements < 1:
        raise ValueError("batch_size and placements must be positive")
    manifest = manifest.sort_values("case_id").reset_index(drop=True)
    for model in models.values():
        model.to(device).eval()
    clean, boxes = {name: {} for name in MODEL_ORDER}, {}
    pending = {name: [] for name in MODEL_ORDER}

    def flush_clean(name):
        if pending[name]:
            ids, tensors = zip(*pending[name])
            for case_id, probability in zip(ids, _infer(models[name], tensors, device)):
                clean[name][case_id] = probability
            pending[name].clear()

    for case in manifest.itertuples(index=False):
        with Image.open(case.path) as opened:
            original = opened.convert("RGB")
        boxes[case.case_id] = clean_crop_bbox(original) if any(crop_border.values()) else None
        for name in MODEL_ORDER:
            bbox = boxes[case.case_id] if crop_border[name] else None
            pending[name].append((case.case_id, fixed_preprocess(original, bbox)))
            if len(pending[name]) >= batch_size:
                flush_clean(name)
    for name in MODEL_ORDER:
        flush_clean(name)
    print(f"Clean predictions cached: {len(manifest)} images × 2 models", flush=True)
    records = []

    def flush_perturbed(name):
        if pending[name]:
            bases, tensors = zip(*pending[name])
            for base, probability in zip(bases, _infer(models[name], tensors, device)):
                records.append(_prediction_row(base, name, clean[name][base["case_id"]], probability))
            pending[name].clear()

    for image_number, case in enumerate(manifest.itertuples(index=False), 1):
        with Image.open(case.path) as opened:
            original = opened.convert("RGB")
        valid = valid_field_mask(original)
        if not valid.any():
            raise ValueError(f"No valid field found for case {case.case_id}")
        for coverage in sorted(set(coverages)):
            for placement in range(1 if coverage == 0 else placements):
                pair_seed = stable_seed(seed, case.case_id, placement)
                perturbed, mask = make_glare_pair(original, coverage, pair_seed, valid)
                mask_hash = hashlib.sha256(np.asarray(mask.shape, dtype="<i8").tobytes() + mask.tobytes()).hexdigest()
                shared = dict(case_id=case.case_id, image_name=Path(case.path).name, patient_id=case.patient_id,
                              true_label=case.label, true_label_id=int(case.label_id),
                              coverage=float(coverage), placement=placement,
                              perturbation_seed=pair_seed, mask_sha256=mask_hash,
                              actual_coverage=float(mask.mean()),
                              actual_fov_coverage=float(mask.sum() / valid.sum()),
                              original_width=original.width, original_height=original.height)
                for name in MODEL_ORDER:
                    bbox = boxes[case.case_id] if crop_border[name] else None
                    base = dict(shared, input_coverage=input_mask_coverage(mask, bbox),
                                crop_bbox=json.dumps(bbox or (0, 0, original.width, original.height)))
                    if coverage == 0:
                        probability = clean[name][case.case_id]
                        records.append(_prediction_row(base, name, probability, probability))
                    else:
                        pending[name].append((base, fixed_preprocess(perturbed, bbox)))
                        if len(pending[name]) >= batch_size:
                            flush_perturbed(name)
        if image_number % 10 == 0 or image_number == len(manifest):
            print(f"Paired glare evaluation: {image_number}/{len(manifest)} images", flush=True)
    for name in MODEL_ORDER:
        flush_perturbed(name)
    return pd.DataFrame(records).sort_values(["coverage", "case_id", "placement", "model"]).reset_index(drop=True)


# Numerator/denominator pairs keep observed and bootstrap metric definitions identical.
METRIC_COMPONENTS = {
    "flip_rate": ("flipped", "n"),
    "correct_to_wrong_rate": ("correct_to_wrong", "clean_correct"),
    "correct_to_wrong_all_rate": ("correct_to_wrong", "n"),
    "wrong_to_correct_rate": ("wrong_to_correct", "clean_wrong"),
    "clean_accuracy": ("clean_correct", "n"),
    "perturbed_accuracy": ("perturbed_correct", "n"),
    "true_probability_delta_mean": ("true_probability_delta", "n"),
    "true_probability_drop_mean": ("true_probability_drop", "n"),
    "probability_l1_change_mean": ("probability_l1_change", "n"),
}


def _components(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["n"] = 1
    result["clean_wrong"] = 1 - result["clean_correct"]
    return result


def _ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, coverage), frame in predictions.groupby(["model", "coverage"], sort=True):
        totals = _components(frame).select_dtypes(include="number").sum()
        row = dict(model=model, coverage=coverage, n_images=frame.case_id.nunique(),
                   n_patients=frame.patient_id.nunique(), n_placements=frame.placement.nunique(),
                   n_observations=len(frame), n_clean_correct_observations=int(totals.clean_correct),
                   n_clean_wrong_observations=int(totals.clean_wrong))
        for name, (numerator, denominator) in METRIC_COMPONENTS.items():
            row[name] = float(_ratio(totals[numerator], totals[denominator]))
        for column in ("actual_coverage", "actual_fov_coverage", "input_coverage"):
            row[f"{column}_mean"] = float(frame[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(predictions: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Full minus baseline, with paired patient-cluster percentile bootstrap CIs.

    Resample patients, retaining ALL their images and placements. Ratios are
    recomputed in each bootstrap replicate; undefined denominators remain NaN.
    The common-clean-correct metric uses the same denominator for both models.
    """
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap cannot be negative")
    rows = []
    keys = ["case_id", "coverage", "placement"]
    for coverage, group in predictions.groupby("coverage", sort=True):
        baseline = group[group.model == "baseline"].set_index(keys)
        full = group[group.model == "full"].set_index(keys)
        if baseline.empty or baseline.index.has_duplicates or full.index.has_duplicates or set(baseline.index) != set(full.index):
            raise ValueError("Baseline/full must contain exactly one row per matched case, coverage, placement")
        full = full.loc[baseline.index]
        for column in ("patient_id", "true_label_id", "mask_sha256", "perturbation_seed"):
            if not baseline[column].equals(full[column]):
                raise ValueError(f"Unpaired baseline/full metadata: {column}")
        patient_ids = sorted(baseline.patient_id.unique())
        common = (baseline.clean_correct.astype(bool) & full.clean_correct.astype(bool)).astype(int)
        metric_components = dict(METRIC_COMPONENTS,
            common_clean_correct_to_wrong_rate=("common_correct_to_wrong", "common_clean_correct"))
        component_names = sorted({item for pair in metric_components.values() for item in pair})
        patient_arrays = {}
        for name, frame in (("baseline", baseline), ("full", full)):
            components = _components(frame)
            components["common_clean_correct"] = common
            components["common_correct_to_wrong"] = common * (1 - frame.perturbed_correct)
            patient_arrays[name] = components.groupby("patient_id")[component_names].sum().reindex(patient_ids).to_numpy(dtype=float)
        row = dict(coverage=coverage, comparison="full-minus-baseline", n_images=baseline.index.get_level_values("case_id").nunique(),
                   n_patients=len(patient_ids), n_observations=len(baseline),
                   n_common_clean_correct_images=baseline[common.eq(1)].index.get_level_values("case_id").nunique(),
                   n_common_clean_correct_observations=int(common.sum()), bootstrap_requested=n_bootstrap,
                   bootstrap_method="paired_patient_cluster_percentile_95" if n_bootstrap and len(patient_ids) > 1 else
                   ("disabled" if n_bootstrap == 0 else "insufficient_patients"))
        draws = None
        if n_bootstrap and len(patient_ids) > 1:
            draws = np.random.default_rng(seed).multinomial(len(patient_ids), [1 / len(patient_ids)] * len(patient_ids), size=n_bootstrap)
        totals = {name: array.sum(axis=0) for name, array in patient_arrays.items()}
        sampled = {name: draws @ array for name, array in patient_arrays.items()} if draws is not None else {}
        for metric, (numerator, denominator) in metric_components.items():
            ni, di = component_names.index(numerator), component_names.index(denominator)
            values = {name: float(_ratio(total[ni], total[di])) for name, total in totals.items()}
            row[f"baseline_{metric}"], row[f"full_{metric}"] = values["baseline"], values["full"]
            row[f"{metric}_difference"] = values["full"] - values["baseline"]
            differences = np.array([])
            if draws is not None:
                differences = _ratio(sampled["full"][:, ni], sampled["full"][:, di]) - _ratio(sampled["baseline"][:, ni], sampled["baseline"][:, di])
                differences = differences[np.isfinite(differences)]
            low, high = np.percentile(differences, [2.5, 97.5]) if len(differences) else (np.nan, np.nan)
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = low, high
            row[f"{metric}_bootstrap_valid"] = len(differences)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Relocate a flat dataset using each CSV path's basename")
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--coverages", type=float, nargs="+", default=[0, .05, .10, .20])
    parser.add_argument("--placements", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-attn-hires", action="store_true", help="Must match the full model's training flag")
    parser.add_argument("--no-full-crop-border", dest="full_crop_border", action="store_false", default=True)
    parser.add_argument("--baseline-crop-border", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Patient bootstrap replicates; 0 disables CIs")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.placements < 1 or args.bootstrap < 0 or args.seed < 0:
        parser.error("batch-size/placements must be positive; bootstrap/seed must be nonnegative")
    if any(not np.isfinite(c) or not 0 <= c <= 1 for c in args.coverages):
        parser.error("coverages must be finite numbers in [0, 1]")
    if len(set(args.coverages)) != len(args.coverages):
        parser.error("coverages cannot contain duplicates")
    args.coverages = sorted(args.coverages)
    return args


def load_models(args, device):
    models = {"baseline": build_resnet50(pretrained=False),
              "full": build_attention_resnet50(pretrained=False, hires=args.full_attn_hires)}
    for name, path in (("baseline", args.baseline_checkpoint), ("full", args.full_checkpoint)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError(f"{name} checkpoint must be a plain model state_dict (no wrapper)")
        models[name].load_state_dict(state, strict=True)
        models[name].to(device).eval()
    return models


def write_summary(path: Path, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    def percentage(value):
        return "NA（分母为0）" if pd.isna(value) else f"{100 * value:.3f}%"
    lines = ["# 逐图配对合成亮斑评估", "", "这是近白色方块压力测试，不等同于真实光学反光。结果不能预设完整方案更优。", "",
             "所有率均按图像×位置观测计算；每位患者的全部图像和位置共同进入配对 bootstrap。0覆盖率每张图仅记录一次。", "",
             "|模型|请求覆盖率|翻转率|原本正确→错误|准确率|真实类别概率下降|模型输入实际覆盖率|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"|{row.model}|{percentage(row.coverage)}|{percentage(row.flip_rate)}|{percentage(row.correct_to_wrong_rate)}|{percentage(row.perturbed_accuracy)}|{row.true_probability_drop_mean:+.5f}|{percentage(row.input_coverage_mean)}|")
    lines += ["", "差值为 full − baseline。正确→错误率/概率下降的负值有利于 full；准确率差值的正值有利于 full。翻转率降低只代表预测更稳定，稳定的错误预测不等于更好。", "",
              "|请求覆盖率|共同干净样本正确→错误率差|95%患者配对区间|共同干净正确图像数|", "|---|---:|---|---:|"]
    metric = "common_clean_correct_to_wrong_rate"
    for _, row in comparison.iterrows():
        low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
        interval = "NA" if pd.isna(low) else f"[{100 * low:+.3f}, {100 * high:+.3f}] 个百分点"
        difference = row[f"{metric}_difference"]
        difference_text = "NA（分母为0）" if pd.isna(difference) else f"{100 * difference:+.3f} 个百分点"
        lines.append(f"|{percentage(row.coverage)}|{difference_text}|{interval}|{int(row.n_common_clean_correct_images)}|")
    lines += ["", "正确→错误率以各模型原本正确的观测为分母；全样本率另存 correct_to_wrong_all_rate。共同样本差值仅使用两模型干净图均正确的图像。", "",
              "概率变化 = 扰动后 − 干净，概率下降 = 干净 − 扰动后；数值为概率单位（0.01 = 1个百分点）。", "",
              "请求覆盖率按原图面积和方块数换算，重叠、视野边缘截断会使实际覆盖率不同。请同时检查 actual_coverage、actual_fov_coverage、input_coverage。", "",
              "不同模型预处理可能保留不同亮斑面积，因此结果代表整个方案的抗扰动表现，不能单独归因于注意力/损失函数。", "",
              "区间反映当前测试患者抽样的不确定性，不包含训练种子、合成方式或外部中心差异；不能代替真实反光病例验证。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("output-dir must be absent or empty; existing results will not be overwritten")
    split_path = args.splits_dir / f"{args.split}.csv"
    manifest = validate_manifest(pd.read_csv(split_path, dtype=str), args.image_root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_hashes = {"baseline": sha256_file(args.baseline_checkpoint), "full": sha256_file(args.full_checkpoint)}
    models = load_models(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # If interrupted, no metadata with completed=true is written.
    predictions = collect_predictions(models, manifest, args.coverages, args.placements,
        args.seed, args.batch_size, device, {"baseline": args.baseline_crop_border, "full": args.full_crop_border})
    summary = summarize_predictions(predictions)
    comparison = compare_models(predictions, args.bootstrap, args.seed)
    predictions.to_csv(args.output_dir / "per_image_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison.csv", index=False)
    write_summary(args.output_dir / "summary.md", summary, comparison)
    members = manifest[["case_id", "patient_id", "label", "label_id"]].sort_values("case_id").to_dict("records")
    metadata = dict(completed=True, script_version=SCRIPT_VERSION, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        device=str(device), class_order=CLASS_ORDER, checkpoint_sha256=checkpoint_hashes,
        split_sha256=sha256_file(split_path),
        member_fingerprint_sha256=hashlib.sha256(json.dumps(members, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        n_images=len(manifest), n_patients=manifest.patient_id.nunique(), n_prediction_rows=len(predictions),
        versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                      numpy=np.__version__, pandas=pd.__version__, pillow=pillow_version),
        perturbation=dict(type="synthetic_near_white_square", pixel_value=252,
            patch_side="round(sqrt(original_width*original_height)*20/224), minimum 1",
            count="round(requested_coverage*original_area/side^2), minimum 1 for positive coverage",
            field="Original-resolution complement of edge-connected luminance<=24 dark pixels",
            pairing="Same original RGB perturbation and mask for both models; seed=SHA256(base_seed, case_id, placement) first 64 bits; same-seed masks nested across coverage",
            actual_coverage="Union mask pixels / original image pixels",
            actual_fov_coverage="Union mask pixels / valid field pixels",
            input_coverage="Mask mean after model-specific fixed crop and identical bilinear resize/center crop; fractional boundary pixels",
            mask_sha256="SHA256(little-endian int64 [height,width] bytes + row-major boolean mask bytes)"),
        preprocessing="Each crop bbox is derived once from clean RGB image and reused for every perturbation. Resize(256), CenterCrop(224), ImageNet normalization match get_eval_transforms.",
        metric_definitions=dict(flip_rate="changed argmax / all observations",
            correct_to_wrong_rate="clean correct AND perturbed wrong / clean correct observations",
            correct_to_wrong_all_rate="clean correct AND perturbed wrong / all observations",
            wrong_to_correct_rate="clean wrong AND perturbed correct / clean wrong observations",
            true_probability_delta="p_true(perturbed)-p_true(clean)", true_probability_drop="p_true(clean)-p_true(perturbed)",
            common_clean_correct_to_wrong_rate="perturbed wrong among cases both models classified correctly when clean / common-clean-correct observations",
            zero_denominator="NaN (empty CSV field), never replaced with zero",
            aggregation="Image-placement weighted pooled rates; zero coverage has one observation per image and model"),
        bootstrap=dict(method="paired patient-cluster percentile", interval=.95, replicates=args.bootstrap,
            sampling="Same resampled patients for both models; keep all images and placements per sampled patient; recompute ratios",
            undefined_replicates="Discard undefined denominator replicates per metric; report metric_bootstrap_valid; no CI for <2 patients or bootstrap=0"),
        limitations=["Synthetic square occlusion is not physical glare and may hide diagnostic anatomy.",
                     "Different preprocessing can retain different perturbation areas; comparison covers the entire pipeline.",
                     "No training-seed uncertainty or external validation; no guaranteed model advantage.",
                     "Image data are not exported; prediction files still contain patient identifiers and require appropriate handling.",
                     "Checkpoint state_dict cannot identify hires/training preprocessing; supplied flags must match training."])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {args.output_dir.resolve()} ({len(predictions)} prediction rows)", flush=True)


if __name__ == "__main__":
    main()
