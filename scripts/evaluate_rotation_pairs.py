"""Validation-only paired small-rotation diagnostic for matched EQ-off/EQ-on models.

Both checkpoints must use the same cropped attention ResNet50. Zero degrees is
an exact cached identity. Nonzero rotations happen AFTER ImageNet normalization:
zero padding therefore represents the ImageNet mean RGB color, not black.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision

from scripts.evaluate_glare_pairs import sha256_file, validate_manifest
from src.attention import build_attention_resnet50
from src.data import IMAGENET_MEAN, get_eval_transforms
from src.regularization import _rotate

MODEL_ORDER = ("eq_off", "eq_on")
CLASS_ORDER = ("lt", "yz", "zz")
DEFAULT_ANGLES = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
SCRIPT_VERSION = "1.0.0"
METRICS = {
    "clean_accuracy": ("clean_correct", "n"),
    "rotated_accuracy": ("rotated_correct", "n"),
    "flip_rate": ("flipped", "n"),
    "correct_to_wrong_rate": ("correct_to_wrong", "clean_correct"),
    "correct_to_wrong_all_rate": ("correct_to_wrong", "n"),
    "wrong_to_correct_rate": ("wrong_to_correct", "clean_wrong"),
    "wrong_to_correct_all_rate": ("wrong_to_correct", "n"),
    "common_clean_correct_to_wrong_rate": ("common_correct_to_wrong", "common_clean_correct"),
    "true_probability_delta_mean": ("true_probability_delta", "n"),
    "true_probability_drop_mean": ("true_probability_drop", "n"),
    "probability_l1_change_mean": ("probability_l1_change", "n"),
}
ATTENTION_COLUMNS = (
    "attention_equivariance_mse", "attention_valid_fraction",
    "clean_attention_mean", "clean_attention_std", "clean_attention_entropy",
    "rotated_attention_mean", "rotated_attention_std", "rotated_attention_entropy",
)


def validate_angles(angles):
    values = [float(value) for value in angles]
    if not values or any(not np.isfinite(value) or abs(value) > 15 for value in values):
        raise ValueError("Angles must be finite and within [-15, 15] degrees")
    if len(set(values)) != len(values) or 0.0 not in values or not any(values):
        raise ValueError("Angles must be unique and include zero and a nonzero angle")
    return sorted(values)


def rotate_inputs(inputs, angle):
    """No interpolation whatsoever for zero degrees."""
    if float(angle) == 0.0:
        return inputs
    return _rotate(inputs, inputs.new_tensor(float(angle)))


def _attention_distribution(attention):
    values = attention.flatten(1).double()
    if values.size(1) < 2 or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Attention must be finite, nonnegative, and spatial")
    total = values.sum(1, keepdim=True)
    probabilities = values / total.clamp_min(1e-300)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1)
    entropy /= np.log(values.size(1))
    entropy = torch.where(total[:, 0] > 0, entropy, torch.full_like(entropy, float("nan")))
    return values.mean(1), values.std(1, unbiased=False), entropy


def attention_diagnostics(clean_attention, rotated_attention, angle, work=56):
    """Raw MSE in a rotated valid region, plus statistics revealing constant maps."""
    if clean_attention.shape != rotated_attention.shape or clean_attention.ndim != 4:
        raise ValueError("Clean/rotated attention shapes must match (N,1,H,W)")
    if clean_attention.shape[1] != 1:
        raise ValueError("Expected one spatial attention channel")
    clean = F.interpolate(clean_attention, (work, work), mode="bilinear", align_corners=False)
    altered = F.interpolate(rotated_attention, (work, work), mode="bilinear", align_corners=False)
    target = rotate_inputs(clean, angle)
    valid = rotate_inputs(torch.ones_like(clean), angle) > .99
    denominator = valid.flatten(1).sum(1)
    if (denominator == 0).any():
        raise ValueError("Rotation has no valid attention intersection")
    mse = (((altered - target) ** 2) * valid).flatten(1).sum(1) / denominator
    result = {
        "attention_equivariance_mse": mse,
        "attention_valid_fraction": valid.float().flatten(1).mean(1),
    }
    for prefix, attention in (("clean", clean_attention), ("rotated", rotated_attention)):
        for statistic, value in zip(("mean", "std", "entropy"), _attention_distribution(attention)):
            result[f"{prefix}_attention_{statistic}"] = value
    return {name: value.detach().cpu().numpy() for name, value in result.items()}


def prediction_row(base, model, clean, rotated):
    truth = int(base["true_label_id"])
    clean_id, rotated_id = int(clean.argmax()), int(rotated.argmax())
    clean_ok, rotated_ok = clean_id == truth, rotated_id == truth
    row = dict(base, model=model, clean_pred_id=clean_id, rotated_pred_id=rotated_id,
               clean_pred=CLASS_ORDER[clean_id], rotated_pred=CLASS_ORDER[rotated_id],
               clean_correct=int(clean_ok), rotated_correct=int(rotated_ok),
               flipped=int(clean_id != rotated_id),
               correct_to_wrong=int(clean_ok and not rotated_ok),
               wrong_to_correct=int(not clean_ok and rotated_ok),
               true_probability_delta=float(rotated[truth] - clean[truth]),
               true_probability_drop=float(clean[truth] - rotated[truth]),
               probability_l1_change=float(np.abs(rotated - clean).sum()))
    for index, label in enumerate(CLASS_ORDER):
        row[f"clean_prob_{label}"] = float(clean[index])
        row[f"prob_{label}"] = float(rotated[index])
    return row


def _infer(model, inputs):
    logits, attention = model(inputs, return_attn=True)
    probabilities = logits.softmax(1)
    if probabilities.shape != (len(inputs), 3) or not torch.isfinite(probabilities).all():
        raise ValueError("Expected finite three-class probabilities")
    if attention.shape[0] != len(inputs):
        raise ValueError("Attention and probability batch sizes differ")
    return probabilities.detach().cpu().numpy(), attention


def collect_predictions(models, manifest, angles=DEFAULT_ANGLES, batch_size=32, device="cpu"):
    angles = validate_angles(angles)
    if set(models) != set(MODEL_ORDER) or batch_size < 1:
        raise ValueError("Provide eq_off and eq_on models and a positive batch size")
    for model in models.values():
        model.to(device).eval()
    ordered = manifest.sort_values("case_id").reset_index(drop=True)
    transform = get_eval_transforms(crop_border=True)
    records = []
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            cases = list(ordered.iloc[start:start + batch_size].itertuples(index=False))
            tensors = []
            for case in cases:
                with Image.open(case.path) as opened:
                    tensors.append(transform(opened.convert("RGB")))
            inputs = torch.stack(tensors).to(device)
            clean = {name: _infer(models[name], inputs) for name in MODEL_ORDER}
            for angle in angles:
                altered_inputs = rotate_inputs(inputs, angle)
                for name in MODEL_ORDER:
                    clean_prob, clean_attention = clean[name]
                    rotated_prob, rotated_attention = (
                        clean[name] if angle == 0 else _infer(models[name], altered_inputs)
                    )
                    diagnostics = attention_diagnostics(clean_attention, rotated_attention, angle)
                    for index, case in enumerate(cases):
                        base = dict(case_id=case.case_id, patient_id=case.patient_id,
                                    image_name=Path(case.path).name, true_label=case.label,
                                    true_label_id=int(case.label_id), angle_deg=float(angle))
                        base.update({key: float(values[index]) for key, values in diagnostics.items()})
                        records.append(prediction_row(base, name, clean_prob[index], rotated_prob[index]))
            print(f"Rotation evaluation: {min(start + batch_size, len(ordered))}/{len(ordered)} images",
                  flush=True)
    return pd.DataFrame(records).sort_values(["angle_deg", "case_id", "model"]).reset_index(drop=True)


def _ratio(numerator, denominator):
    num, den = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den != 0)


def _paired_groups(predictions):
    keys = ["case_id", "angle_deg"]
    if set(predictions.model.unique()) != set(MODEL_ORDER):
        raise ValueError("Prediction data require exactly eq_off and eq_on")
    tables = {}
    for name in MODEL_ORDER:
        frame = predictions[predictions.model == name].set_index(keys).sort_index()
        if frame.empty or frame.index.has_duplicates:
            raise ValueError("Expected exactly one row per model/case/angle")
        tables[name] = frame
    left, right = tables["eq_off"], tables["eq_on"]
    if not left.index.equals(right.index):
        raise ValueError("EQ models have unmatched cases/angles")
    for column in ("patient_id", "true_label_id", "true_label", "image_name"):
        if not left[column].equals(right[column]):
            raise ValueError(f"Unpaired metadata: {column}")
    angles = validate_angles(left.index.get_level_values("angle_deg").unique())
    if not left.groupby(level="case_id").size().eq(len(angles)).all():
        raise ValueError("Every case must have all requested angles")
    for name, frame in tables.items():
        stable = ["patient_id", "true_label_id", "clean_correct",
                  "clean_prob_lt", "clean_prob_yz", "clean_prob_zz"]
        if frame.groupby(level="case_id")[stable].nunique(dropna=False).gt(1).any().any():
            raise ValueError(f"Clean predictions/identity changed across angles for {name}")
        zero = frame.xs(0., level="angle_deg")
        if not zero.clean_correct.equals(zero.rotated_correct):
            raise ValueError("Zero degrees must retain original correctness")
        for label in CLASS_ORDER:
            if not np.array_equal(zero[f"clean_prob_{label}"], zero[f"prob_{label}"]):
                raise ValueError("Zero degrees must retain exact original probabilities")
    # The clean-correct flags are constant across angles, so this fixes a single
    # common cohort for all angles and both models.
    common = (left.clean_correct.astype(bool) & right.clean_correct.astype(bool)).astype(int)
    for name, frame in tables.items():
        frame = frame.copy()
        frame["n"] = 1.
        frame["clean_wrong"] = 1 - frame.clean_correct
        frame["common_clean_correct"] = common
        frame["common_correct_to_wrong"] = common * (1 - frame.rotated_correct)
        tables[name] = frame
    # A complete rectangular grid makes pooling equivalent to taking the
    # nonzero-angle mean within each image, then an image-weighted mean.
    yield "nonzero_mean", np.nan, {
        name: frame[frame.index.get_level_values("angle_deg") != 0] for name, frame in tables.items()
    }
    for angle in angles:
        yield "angle", angle, {
            name: frame[frame.index.get_level_values("angle_deg") == angle] for name, frame in tables.items()
        }


def summarize_predictions(predictions):
    rows = []
    for scope, angle, tables in _paired_groups(predictions):
        for name, frame in tables.items():
            totals = frame.select_dtypes(include="number").sum()
            row = dict(model=name, scope=scope, angle_deg=angle,
                       n_images=frame.index.get_level_values("case_id").nunique(),
                       n_patients=frame.patient_id.nunique(), n_observations=len(frame),
                       n_angles=frame.index.get_level_values("angle_deg").nunique(),
                       n_common_clean_correct_images=frame.loc[frame.common_clean_correct.eq(1)].index
                       .get_level_values("case_id").nunique())
            for metric, (numerator, denominator) in METRICS.items():
                row[metric] = float(_ratio(totals[numerator], totals[denominator]))
            for label_id, label in enumerate(CLASS_ORDER):
                selected = frame[frame.true_label_id == label_id]
                row[f"recall_{label}"] = float(selected.rotated_correct.mean()) if len(selected) else np.nan
            for column in ATTENTION_COLUMNS:
                row[f"{column}_mean"] = float(frame[column].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def compare_models(predictions, n_bootstrap=10000, seed=20261020):
    if n_bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    rows = []
    components = sorted({item for pair in METRICS.values() for item in pair})
    for scope, angle, tables in _paired_groups(predictions):
        frame = tables["eq_off"]
        patients = sorted(frame.patient_id.unique())
        arrays = {name: table.groupby("patient_id")[components].sum().reindex(patients).to_numpy(float)
                  for name, table in tables.items()}
        totals = {name: array.sum(0) for name, array in arrays.items()}
        draws = None
        if n_bootstrap and len(patients) > 1:
            draws = np.random.default_rng(seed).multinomial(
                len(patients), [1 / len(patients)] * len(patients), size=n_bootstrap)
        sampled = {name: draws @ array for name, array in arrays.items()} if draws is not None else {}
        row = dict(scope=scope, angle_deg=angle, comparison="eq_on-minus-eq_off",
                   n_images=frame.index.get_level_values("case_id").nunique(), n_patients=len(patients),
                   n_observations=len(frame), bootstrap_requested=n_bootstrap,
                   n_common_clean_correct_images=frame.loc[frame.common_clean_correct.eq(1)].index
                   .get_level_values("case_id").nunique(),
                   bootstrap_method="paired_patient_cluster_percentile_95" if draws is not None
                   else ("disabled" if n_bootstrap == 0 else "insufficient_patients"))
        for metric, (numerator, denominator) in METRICS.items():
            ni, di = components.index(numerator), components.index(denominator)
            for name in MODEL_ORDER:
                row[f"{name}_{metric}"] = float(_ratio(totals[name][ni], totals[name][di]))
            row[f"{metric}_difference"] = row[f"eq_on_{metric}"] - row[f"eq_off_{metric}"]
            differences = np.array([])
            if draws is not None:
                differences = (_ratio(sampled["eq_on"][:, ni], sampled["eq_on"][:, di])
                               - _ratio(sampled["eq_off"][:, ni], sampled["eq_off"][:, di]))
                differences = differences[np.isfinite(differences)]
            low, high = np.percentile(differences, [2.5, 97.5]) if len(differences) else (np.nan, np.nan)
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = float(low), float(high)
            row[f"{metric}_bootstrap_valid"] = len(differences)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eq-off-checkpoint", type=Path, required=True)
    parser.add_argument("--eq-on-checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--angles", nargs="+", type=float, default=list(DEFAULT_ANGLES))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--dropout", type=float, default=.3)
    args = parser.parse_args(argv)
    args.angles = validate_angles(args.angles)
    if args.batch_size < 1 or args.bootstrap < 0 or not 0 <= args.dropout < 1:
        parser.error("Positive batch size, nonnegative bootstrap, and 0 <= dropout < 1 required")
    return args


def load_models(args, device):
    result = {}
    for name, path in (("eq_off", args.eq_off_checkpoint), ("eq_on", args.eq_on_checkpoint)):
        model = build_attention_resnet50(pretrained=False, dropout=args.dropout, hires=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        result[name] = model.to(device).eval()
    return result


def _training_provenance(checkpoints, dropout, split_path=None):
    provenance = {}
    common_fields = ("crop_border", "attn_hires", "dropout", "lambda_pf", "pf_mask",
                     "max_angle", "split_fingerprints", "initial_state_sha256", "initialization_sha256")
    for name, checkpoint in checkpoints.items():
        path = checkpoint.parent / "run_config.json"
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError(f"Expected object in training config: {path}")
            settings = dict(config.get("arguments", {}), **config)
            regularization = config.get("regularization", config)
            for field, expected in (("crop_border", True), ("attn_hires", False), ("dropout", dropout)):
                if field in settings and settings[field] != expected:
                    raise ValueError(f"{name} training {field} does not match evaluation")
            if "lambda_eq" in regularization:
                if name == "eq_off" and regularization["lambda_eq"] != 0:
                    raise ValueError("eq_off checkpoint training lambda_eq must be zero")
                if name == "eq_on" and regularization["lambda_eq"] <= 0:
                    raise ValueError("eq_on checkpoint training lambda_eq must be positive")
            if config.get("protocol_version") == "eq_sham_control_v1":
                if not config.get("completed") or not config.get("eq_auxiliary_forward"):
                    raise ValueError("EQ controlled training must be complete and match auxiliary forwards")
                if config.get("fixed_architecture") != "AnatomyAttentionResNet50_layer4_7x7":
                    raise ValueError("Training architecture does not match evaluation")
                if checkpoint.name != config.get("primary_checkpoint"):
                    raise ValueError("Evaluate the fixed-budget primary checkpoint, not selected best_val")
                if config.get("checkpoint_sha256", {}).get(checkpoint.name) != sha256_file(checkpoint):
                    raise ValueError("Checkpoint does not match training provenance hash")
                if split_path is not None:
                    expected = config.get("splits", {}).get("val", {}).get("csv_sha256")
                    if expected != sha256_file(split_path):
                        raise ValueError("Validation split differs from training provenance")
            provenance[name] = dict(path=str(path.resolve()), sha256=sha256_file(path), config=config)
        else:
            provenance[name] = dict(config=None, limitation="No adjacent run_config.json; training identity unverified")
    configs = [provenance[name]["config"] for name in MODEL_ORDER]
    if all(config is not None for config in configs):
        for field in common_fields:
            if field in configs[0] and field in configs[1] and configs[0][field] != configs[1][field]:
                raise ValueError(f"Training configs differ in shared field: {field}")
        for field in ("seed", "protocol_version", "arguments", "splits", "source_sha256", "eq_auxiliary_forward",
                      "training_input_trace_sha256", "training_input_traces", "completed_epochs"):
            if field in configs[0] and field in configs[1] and configs[0][field] != configs[1][field]:
                raise ValueError(f"Training configs differ in shared field: {field}")
        if "regularization" in configs[0] and "regularization" in configs[1]:
            fixed = [{key: value for key, value in config["regularization"].items() if key != "lambda_eq"}
                     for config in configs]
            if fixed[0] != fixed[1]:
                raise ValueError("Non-EQ regularization settings differ across models")
    pair_paths = {checkpoint.parent.parent / "pair_metadata.json" for checkpoint in checkpoints.values()}
    provenance["pair_metadata"] = [
        dict(path=str(path.resolve()), sha256=sha256_file(path), content=json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(pair_paths) if path.is_file()
    ]
    for pair in provenance["pair_metadata"]:
        content = pair["content"]
        if content.get("protocol_version") != "eq_sham_control_v1":
            continue
        if not all(content.get(field) for field in
                   ("completed", "initial_states_identical", "training_input_streams_identical")):
            raise ValueError("Training pair is incomplete or its initialization/input streams differ")
        for name, checkpoint in checkpoints.items():
            arm = content.get("arms", {}).get(name, {})
            if arm.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError("Training pair metadata does not match actual checkpoint")
            if arm.get("run_config_sha256") != provenance[name].get("sha256"):
                raise ValueError("Training pair metadata does not match run_config")
            config = provenance[name]["config"]
            if config is None or arm.get("initial_state_sha256") != config.get("initial_state_sha256"):
                raise ValueError("Training pair initialization hash does not match run_config")
    return provenance


def write_summary(path, summary, comparison):
    def number(value, scale=1):
        return "NA" if pd.isna(value) else f"{value * scale:+.4f}"
    lines = [
        "# EQ 旋转约束：验证集配对诊断", "",
        "eq_off 为零 EQ 权重对照；是否匹配辅助前向、BN 和训练随机路径需结合训练记录核验。",
        "这是固定画布上的合成小角度旋转，含插值和边角裁切，不等同于真实转镜。",
        "归一化张量补 0 对应 ImageNet 均值 RGB；0° 直接复用原图预测，没有插值。", "",
        "|模型|范围|准确率|翻转率|共同正确变错率|真实类别概率下降|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        label = "非零角度均值" if row.scope == "nonzero_mean" else f"{row.angle_deg:g}°"
        lines.append(f"|{row.model}|{label}|{number(row.rotated_accuracy,100)}%|"
                     f"{number(row.flip_rate,100)}%|{number(row.common_clean_correct_to_wrong_rate,100)}%|"
                     f"{number(row.true_probability_drop_mean)}|")
    lines += ["", "主比较是所有预定非零角度的等权平均；单角度为探索性分解。",
              "差值为 eq_on − eq_off：准确率正值、共同正确变错率和概率下降负值为改善方向。", "",
              "|范围|指标|差值（百分点）|95% 患者配对区间（百分点）|",
              "|---|---|---:|---|"]
    for row in comparison.to_dict("records"):
        label = "非零角度均值" if row["scope"] == "nonzero_mean" else f'{row["angle_deg"]:g}°'
        for metric, display in (("rotated_accuracy", "准确率"), ("common_clean_correct_to_wrong_rate", "共同正确变错率")):
            lines.append(f'|{label}|{display}|{number(row[f"{metric}_difference"],100)}|'
                         f'[{number(row[f"{metric}_ci_low"],100)}, {number(row[f"{metric}_ci_high"],100)}]|')
    lines += [
        "", "共同正确队列固定为两模型在原图上均正确的同一批图；自身正确变错率使用各自干净正确队列。",
        "患者 bootstrap 保留患者的所有图像及角度，再重算比率；重复角度不增加独立患者数。",
        "零分母输出 NA，不当作零；区间仅反映当前固定模型的患者抽样不确定性。",
        "attention_equivariance_mse 是有效区原始注意力 MSE。必须结合原图/旋转后的注意力均值、"
        "标准差和空间熵检查退化；常数图也可能有很低 MSE，不能单凭此值证明解剖解释正确。",
        "概率变化 = 旋转后 − 原图；概率下降为反号的有符号量。低翻转率可能包含稳定的错误预测。",
        "本诊断仅使用验证集。当前已反复查看的测试集不用于再选参数；技术效果确认需要未参与选择的患者。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("output-dir must be absent or empty; existing results will not be overwritten")
    split_path = args.splits_dir / "val.csv"
    manifest = validate_manifest(pd.read_csv(split_path, dtype=str), args.image_root)
    checkpoints = {"eq_off": args.eq_off_checkpoint, "eq_on": args.eq_on_checkpoint}
    hashes = {name: sha256_file(path) for name, path in checkpoints.items()}
    provenance = _training_provenance(checkpoints, args.dropout, split_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = load_models(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = collect_predictions(models, manifest, args.angles, args.batch_size, device)
    summary = summarize_predictions(predictions)
    comparison = compare_models(predictions, args.bootstrap, args.seed)
    predictions.to_csv(args.output_dir / "per_image_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison.csv", index=False)
    write_summary(args.output_dir / "summary.md", summary, comparison)
    members = manifest[["case_id", "patient_id", "label", "label_id"]].sort_values("case_id").to_dict("records")
    metadata = dict(
        completed=True, script_version=SCRIPT_VERSION, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        device=str(device), class_order=CLASS_ORDER, checkpoint_sha256=hashes, training_provenance=provenance,
        split_sha256=sha256_file(split_path),
        member_fingerprint_sha256=hashlib.sha256(json.dumps(members, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        n_images=len(manifest), n_patients=int(manifest.patient_id.nunique()), n_prediction_rows=len(predictions),
        model=dict(architecture="AnatomyAttentionResNet50", crop_border=True, attn_hires=False, dropout=args.dropout),
        preprocessing="CropBlackBorder once per image, Resize(256), CenterCrop(224), ImageNet normalization; identical tensors for both models",
        perturbation=dict(type="synthetic_fixed_canvas_small_rotation", angles_deg=args.angles,
                          interpolation="bilinear", align_corners=False, padding_normalized=0.,
                          padding_rgb=IMAGENET_MEAN, zero_angle="Exact cached identity; no grid_sample",
                          limitations="Includes interpolation and corner clipping; not natural scope rotation"),
        attention=dict(work_grid=56, error="Raw MSE between A(rot(x)) and rot(A(x))",
                       valid_mask="rotate(ones on work grid)>0.99; compare only this common valid area",
                       distribution_statistics="Native-map mean, population std, normalized Shannon spatial entropy; all-zero entropy undefined",
                       limitation="Uniform or low-amplitude maps can have low MSE; no anatomy correctness established"),
        aggregation=dict(primary="nonzero_mean: equal predefined nonzero angles per image, then pooled image-weighted ratios",
                         secondary="Individual angles are exploratory",
                         common_cohort="Same cases correctly classified by both models at zero degrees, fixed across all angles",
                         delta="p_true(rotated)-p_true(clean)", drop="p_true(clean)-p_true(rotated)",
                         zero_denominator="NaN, never replaced with zero"),
        bootstrap=dict(method="paired patient-cluster percentile", replicates=args.bootstrap, seed=args.seed,
                       interval=.95, sampling="Keep all images and angles per sampled patient; recompute ratios",
                       undefined="Discard undefined replicates per metric and report count; no CI for fewer than 2 patients",
                       limitation="Conditional on these trained models, not training-seed or external-center uncertainty"),
        versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                      numpy=np.__version__, pandas=pd.__version__),
        limitations=["Validation diagnostic only; no test images were evaluated.",
                     "No expected direction is guaranteed and no parameter selection is performed.",
                     "State dict alone cannot establish training method; inspect adjacent training metadata.",
                     "Prediction files contain patient identifiers and require appropriate handling."])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {args.output_dir.resolve()} ({len(predictions)} prediction rows)", flush=True)


if __name__ == "__main__":
    main()
