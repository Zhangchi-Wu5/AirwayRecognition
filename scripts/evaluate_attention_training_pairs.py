"""Validation-only evaluation of matched learned/fixed attention training.

The primary contrast is learned_ce minus fixed_ce. The optional regularized
contrast changes EQ and PF together and is not an EQ-only ablation.
"""
from __future__ import annotations

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

from scripts import evaluate_rotation_pairs as rotation
from scripts.evaluate_glare_pairs import sha256_file, validate_manifest
from src.data import IMAGENET_MEAN, get_eval_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "attention_training_control_v1"
ARMS = ("fixed_ce", "learned_ce", "learned_regularized")
POLICIES = {
    "fixed_ce": "fixed_constant_half",
    "learned_ce": "learned_zero_initialized",
    "learned_regularized": "learned_zero_initialized",
}
PAIR_ORDER = (("fixed_ce", "learned_ce"), ("learned_ce", "learned_regularized"))
DEFAULT_ANGLES = rotation.DEFAULT_ANGLES
SCRIPT_VERSION = "1.0.0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.bootstrap < 0 or args.seed < 0:
        parser.error("Positive batch-size and nonnegative bootstrap/seed required")
    return args


def _json_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _split_membership_check(splits):
    """Inspect saved identities only; never resolve/open a test image."""
    sets = {}
    for name in ("train", "val", "test"):
        entry = splits.get(name, {})
        members = entry.get("members")
        _require(isinstance(members, list) and len(members) > 0,
                 f"Missing recorded {name} identities")
        sets[name] = {str(item["patient_id"]) for item in members}
        _require(len(members) == entry.get("n_images"), f"Invalid {name} image count")
        _require(len(sets[name]) == entry.get("n_patients"), f"Invalid {name} patient count")
        ordered = sorted(members, key=lambda row: (
            str(row["patient_id"]), str(row["label"]), str(row["image_name"])))
        fingerprint = hashlib.sha256(
            json.dumps(ordered, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        _require(fingerprint == entry.get("membership_sha256"),
                 f"Recorded {name} membership hash mismatch")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        _require(not sets[left] & sets[right], f"Patient contamination between {left} and {right}")


def _validate_execution_schedule(config, traces, execution):
    batch_size = config["arguments"].get("batch_size", 0)
    _require(isinstance(batch_size, int) and batch_size >= 2, "Invalid recorded training batch size")
    train_images = config["splits"]["train"]["n_images"]
    val_images = config["splits"]["val"]["n_images"]
    train_batches = (train_images + batch_size - 1) // batch_size
    val_batches = (val_images + batch_size - 1) // batch_size
    previous_tracked = None
    for inputs, entry in zip(traces, execution):
        _require(inputs["n_images"] == train_images and inputs["n_batches"] == train_batches,
                 "Input trace does not cover the training split")
        expected_bn = {"train": 2 * train_batches, "eval": val_batches}
        _require(entry.get("train_forward_count") == expected_bn["train"]
                 and entry.get("eval_forward_count") == expected_bn["eval"]
                 and entry.get("model_forward_count") == sum(expected_bn.values()),
                 "Execution trace has an incorrect auxiliary forward schedule")
        bn_counts = entry.get("bn_forward_counts", {})
        _require(bool(bn_counts) and all(value == expected_bn for value in bn_counts.values()),
                 "Execution trace has an incorrect BatchNorm schedule")
        tracked = entry.get("bn_num_batches_tracked", {})
        _require(set(tracked) == set(bn_counts), "Execution trace is missing BatchNorm counters")
        if previous_tracked is not None:
            _require(set(tracked) == set(previous_tracked)
                     and all(tracked[name] - previous_tracked[name] == expected_bn["train"]
                             for name in tracked), "BatchNorm cumulative counters have incorrect increments")
        previous_tracked = tracked
        for field in ("forward_rng_sha256", "end_rng_sha256"):
            value = entry.get(field, "")
            _require(isinstance(value, str) and len(value) == 64
                     and all(char in "0123456789abcdef" for char in value), f"Missing {field}")


def validate_training_run(run_dir, val_csv):
    """Require complete matching training records and the fixed-budget states."""
    pair_path = run_dir / "pair_metadata.json"
    pair = _json_object(pair_path)
    _require(pair.get("protocol_version") == PROTOCOL_VERSION, "Wrong training protocol")
    _require(pair.get("completed") is True, "Training pair is incomplete")
    _require(pair.get("primary_checkpoint") == "last_model.pt", "Expected last_model.pt, not best_val")
    _require(pair.get("test_images_loaded") is False, "Training must not load test images")
    for key in ("initial_states_identical", "training_input_streams_identical", "execution_traces_identical"):
        _require(pair.get(key) is True, f"Training pair failed {key}")
    recorded = pair.get("arms", {})
    _require(set(recorded) in (set(ARMS[:2]), set(ARMS)), "Unrecognized or missing training arms")
    arms = [arm for arm in ARMS if arm in recorded]
    provenance = {}
    shared_fields = (
        "seed", "protocol_version", "arguments", "splits", "source_sha256",
        "initial_state_sha256", "training_input_trace_sha256", "training_input_traces",
        "completed_epochs", "fixed_architecture", "crop_border", "eq_auxiliary_forward",
        "batchnorm_policy", "primary_checkpoint", "execution_trace_sha256", "execution_traces",
    )
    first = None
    for arm in arms:
        directory = run_dir / arm
        config_path = directory / "run_config.json"
        config = _json_object(config_path)
        checkpoint = directory / "last_model.pt"
        info = recorded[arm]
        _require(config.get("completed") is True, f"{arm}: training incomplete")
        _require(config.get("arm") == arm, f"{arm}: mismatched arm identity")
        _require(config.get("protocol_version") == PROTOCOL_VERSION, f"{arm}: wrong protocol")
        _require(config.get("attention_policy") == POLICIES[arm], f"{arm}: wrong attention policy")
        expected_control = {
            "policy": "fixed" if arm == "fixed_ce" else "learned",
            "initialization": "zero_spatial_convolution", "initial_constant": .5,
            "trainable": arm != "fixed_ce", "kernel_size": 3, "hires": False,
        }
        _require(config.get("attention_control") == expected_control
                 and info.get("attention_control") == expected_control
                 and info.get("attention_policy") == POLICIES[arm],
                 f"{arm}: mismatched attention control")
        _require(config.get("checkpoint_reload_verified") is True,
                 f"{arm}: checkpoint reload was not verified")
        _require(config.get("fixed_architecture") == "AnatomyAttentionResNet50_layer4_7x7",
                 f"{arm}: wrong architecture")
        _require(config.get("crop_border") is True and not config.get("attn_hires", False),
                 f"{arm}: wrong crop/attention resolution")
        _require(config.get("primary_checkpoint") == "last_model.pt", f"{arm}: selected best checkpoint")
        _require(config.get("evaluation_split") == "val" and config.get("test_images_loaded") is False,
                 f"{arm}: validation-only protocol required")
        _require(config.get("eq_auxiliary_forward") is True, f"{arm}: unmatched auxiliary forward")
        arguments = config.get("arguments", {})
        epochs = arguments.get("stage1_epochs", -1) + arguments.get("stage2_epochs", -1)
        _require(epochs > 0 and config.get("completed_epochs") == epochs,
                 f"{arm}: incomplete fixed epoch budget")
        _require(config.get("seed") == pair.get("seed"), f"{arm}: seed mismatch")
        dropout = arguments.get("dropout")
        _require(isinstance(dropout, (int, float)) and np.isfinite(dropout) and 0 <= dropout < 1,
                 f"{arm}: invalid dropout")
        regularization = config.get("regularization", {})
        _require(regularization.get("eq_auxiliary_forward") is True, f"{arm}: missing sham forward")
        expected_weights = (0., 0.) if arm != "learned_regularized" else (
            arguments.get("lambda_eq"), arguments.get("lambda_pf"))
        _require((regularization.get("lambda_eq"), regularization.get("lambda_pf")) == expected_weights,
                 f"{arm}: wrong EQ/PF coefficients")
        if arm == "learned_regularized":
            _require(all(isinstance(weight, (int, float)) and np.isfinite(weight) and weight > 0
                         for weight in expected_weights), "Regularized arm must use positive EQ and PF")
        checkpoint_hash = sha256_file(checkpoint)
        _require(checkpoint_hash == config.get("checkpoint_sha256", {}).get("last_model.pt")
                 == info.get("checkpoint_sha256"), f"{arm}: checkpoint hash mismatch")
        _require(sha256_file(config_path) == info.get("run_config_sha256"),
                 f"{arm}: run_config hash mismatch")
        _require(bool(config.get("initial_state_sha256"))
                 and info.get("initial_state_sha256") == config["initial_state_sha256"],
                 f"{arm}: initialization mismatch")
        trace_path = directory / "training_input_trace.csv"
        _require(sha256_file(trace_path) == config.get("training_input_trace_sha256")
                 == info.get("training_input_trace_sha256"), f"{arm}: input trace hash mismatch")
        traces = pd.read_csv(trace_path).to_dict("records")
        _require(len(traces) == epochs and traces == config.get("training_input_traces"),
                 f"{arm}: input traces do not cover completed epochs")
        _require([row["training_epoch"] for row in traces] == list(range(1, epochs + 1)),
                 f"{arm}: nonsequential input trace")
        execution_path = directory / "execution_trace.json"
        _require(sha256_file(execution_path) == config.get("execution_trace_sha256")
                 == info.get("execution_trace_sha256"), f"{arm}: execution trace hash mismatch")
        execution = _json_object(execution_path)
        _require(execution.get("schema_version") == 1
                 and execution.get("epochs") == config.get("execution_traces"),
                 f"{arm}: execution trace contents mismatch")
        _require(len(execution["epochs"]) == epochs
                 and [row["training_epoch"] for row in execution["epochs"]] == list(range(1, epochs + 1)),
                 f"{arm}: incomplete execution trace")
        _validate_execution_schedule(config, traces, execution["epochs"])
        checks = config.get("attention_parameter_checks", [])
        _require(len(checks) == epochs, f"{arm}: incomplete attention parameter checks")
        _require([check.get("training_epoch") for check in checks] == list(range(1, epochs + 1)),
                 f"{arm}: nonsequential attention parameter checks")
        _require(all(check.get("requires_grad") is (arm != "fixed_ce") for check in checks),
                 f"{arm}: wrong attention trainability during training")
        if arm == "fixed_ce":
            _require(all(check.get("count_nonzero") == 0 for check in checks),
                     "fixed_ce attention changed during training")
        _require(config.get("splits", {}).get("val", {}).get("csv_sha256") == sha256_file(val_csv),
                 f"{arm}: validation split hash mismatch")
        if first is None:
            first = config
            _split_membership_check(config.get("splits", {}))
        else:
            for field in shared_fields:
                _require(field in first and config.get(field) == first[field],
                         f"Arms differ in shared field: {field}")
            _require(regularization.get("max_angle") == first["regularization"].get("max_angle"),
                     "Arms use different auxiliary rotation ranges")
        provenance[arm] = {
            "config": config, "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path), "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
        }
    provenance["pair_metadata"] = {
        "path": str(pair_path.resolve()), "sha256": sha256_file(pair_path), "content": pair}
    return provenance


def load_models(provenance, device):
    from src.attention_training_control import build_controlled_attention_resnet50

    models = {}
    for arm in ARMS:
        if arm not in provenance:
            continue
        info = provenance[arm]
        policy = "fixed" if arm == "fixed_ce" else "learned"
        model = build_controlled_attention_resnet50(
            attention_policy=policy, pretrained=False,
            dropout=info["config"]["arguments"]["dropout"])
        state = torch.load(info["checkpoint_path"], map_location="cpu", weights_only=True)
        if arm == "fixed_ce":
            weight = state.get("attention.conv.weight")
            _require(weight is not None and torch.count_nonzero(weight).item() == 0,
                     "fixed_ce checkpoint has nonzero attention weights")
        model.load_state_dict(state, strict=True)
        models[arm] = model.to(device).eval()
    return models


def collect_predictions(models, manifest, batch_size=16, device="cpu"):
    _require(set(models) in (set(ARMS[:2]), set(ARMS)), "Expected matched attention training arms")
    _require(batch_size > 0, "Positive batch size required")
    ordered = manifest.sort_values("case_id").reset_index(drop=True)
    transform = get_eval_transforms(crop_border=True)
    records = []
    for model in models.values():
        model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            cases = list(ordered.iloc[start:start + batch_size].itertuples(index=False))
            tensors = []
            for case in cases:
                with Image.open(case.path) as source:
                    tensors.append(transform(source.convert("RGB")))
            inputs = torch.stack(tensors).to(device)
            clean = {arm: rotation._infer(model, inputs) for arm, model in models.items()}
            for angle in DEFAULT_ANGLES:
                altered = rotation.rotate_inputs(inputs, angle)
                for arm, model in models.items():
                    clean_prob, clean_attention = clean[arm]
                    rotated_prob, rotated_attention = clean[arm] if angle == 0 else rotation._infer(model, altered)
                    if arm == "fixed_ce":
                        _require(torch.equal(clean_attention, torch.full_like(clean_attention, .5))
                                 and torch.equal(rotated_attention, torch.full_like(rotated_attention, .5)),
                                 "fixed_ce must emit exactly constant 0.5 attention")
                    diagnostics = rotation.attention_diagnostics(clean_attention, rotated_attention, angle)
                    for index, case in enumerate(cases):
                        base = dict(case_id=case.case_id, patient_id=case.patient_id,
                                    image_name=Path(case.path).name, true_label=case.label,
                                    true_label_id=int(case.label_id), angle_deg=float(angle))
                        base.update({key: float(value[index]) for key, value in diagnostics.items()})
                        records.append(rotation.prediction_row(base, arm, clean_prob[index], rotated_prob[index]))
            print(f"Attention training evaluation: {min(start + batch_size, len(ordered))}/{len(ordered)} images",
                  flush=True)
    return pd.DataFrame(records).sort_values(["angle_deg", "case_id", "model"]).reset_index(drop=True)


def summarize_pairs(predictions, n_bootstrap=10000, seed=20261020):
    """Reuse paired rotation mathematics without leaking its legacy arm names."""
    _require(set(predictions.model.unique()) in (set(ARMS[:2]), set(ARMS)), "Invalid prediction arms")
    _require(set(predictions.angle_deg.unique()) == set(DEFAULT_ANGLES), "All seven fixed angles required")
    summaries, comparisons = [], []
    for control, treatment in PAIR_ORDER:
        if treatment not in set(predictions.model):
            continue
        label = f"{treatment}-minus-{control}"
        pair = predictions[predictions.model.isin([control, treatment])].copy()
        pair["model"] = pair.model.map({control: "eq_off", treatment: "eq_on"})
        summary = rotation.summarize_predictions(pair)
        summary["model"] = summary.model.map({"eq_off": control, "eq_on": treatment})
        summary.insert(0, "comparison", label)
        summary.insert(1, "common_cohort", f"{control}_and_{treatment}_correct_at_zero")
        comparison = rotation.compare_models(pair, n_bootstrap, seed)
        comparison["comparison"] = label
        comparison.insert(0, "control_arm", control)
        comparison.insert(1, "treatment_arm", treatment)
        comparison.insert(2, "common_cohort", f"{control}_and_{treatment}_correct_at_zero")
        comparison.rename(columns={
            column: (control + column[len("eq_off"):] if column.startswith("eq_off_")
                     else treatment + column[len("eq_on"):])
            for column in comparison.columns if column.startswith(("eq_off_", "eq_on_"))
        }, inplace=True)
        summaries.append(summary)
        comparisons.append(comparison)
    return pd.concat(summaries, ignore_index=True), pd.concat(comparisons, ignore_index=True)


def write_summary(path, summary, comparison):
    def fmt(value, scale=1):
        return "NA" if pd.isna(value) else f"{value * scale:+.4f}"

    lines = [
        "# 从训练开始学习／固定注意力：验证集匹配对照", "",
        "主比较 learned_ce − fixed_ce：两组 EQ/PF 权重均为0，辅助旋转前向仍匹配。",
        "两组从相同参数和 A=0.5 出发；fixed_ce 始终冻结注意力卷积，learned_ce 用分类损失学习。",
        "可选 learned_regularized − learned_ce 同时加入 EQ 与 PF，只能解释为联合正则贡献。",
        "所有模型使用固定训练预算的 last_model.pt。这里没有普通 ResNet 基线。", "",
        "|比较|模型|范围|准确率|翻转率|共同正确变错率|真实类别概率变化|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        if row.scope != "nonzero_mean" and row.angle_deg != 0:
            continue
        scope = "非零角度均值" if row.scope == "nonzero_mean" else "0°"
        lines.append(f"|{row.comparison}|{row.model}|{scope}|{fmt(row.rotated_accuracy,100)}%|"
                     f"{fmt(row.flip_rate,100)}%|{fmt(row.common_clean_correct_to_wrong_rate,100)}%|"
                     f"{fmt(row.true_probability_delta_mean)}|")
    lines += ["", "差值为比较名称中的前者减后者；准确率正值、翻转率与变错率负值为改善方向。", "",
              "|比较|范围|准确率差（百分点）|95%患者配对区间（百分点）|",
              "|---|---|---:|---|"]
    for row in comparison.itertuples(index=False):
        if row.scope != "nonzero_mean" and row.angle_deg != 0:
            continue
        scope = "非零角度均值" if row.scope == "nonzero_mean" else "0°"
        lines.append(f"|{row.comparison}|{scope}|{fmt(row.rotated_accuracy_difference,100)}|"
                     f"[{fmt(row.rotated_accuracy_ci_low,100)}, {fmt(row.rotated_accuracy_ci_high,100)}]|")
    lines += [
        "", "主终点为预设六个非零角度的均值；0°为原图结果，单角度仅作探索性分解。",
        "共同正确队列按每一对模型分别定义：该对在0°均正确的图。不同配对的队列不能混用。",
        "患者配对 bootstrap 保留每位患者的全部图像与角度；区间仅针对当前固定权重，不覆盖训练种子不确定性。",
        "零分母输出 NA，不能当作零。概率变化=旋转后−原图，概率下降为其反号，均为有符号量。",
        "固定 A=0.5 的注意力标准差为0、空间熵为1；很低的等变误差是退化对照，不表示解剖定位正确。",
        "固定组仍保留 global/attention 特征拼接；常数加权分支可被分类头重新缩放，不是普通 ResNet。",
        "学习注意力同时允许整体幅度和空间分布改变；本对照检验可学习注意力分支的训练贡献，不隔离纯空间定位价值。",
        "合成旋转包含插值、边角裁切，归一化补0对应 ImageNet 均值 RGB；0°直接复用预测。",
        "只评估验证集，不读取测试图像；本分析不保证任何方法更好，也不据此自动选择参数。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    _require(not args.output_dir.exists() or (args.output_dir.is_dir() and not any(args.output_dir.iterdir())),
             "output-dir must be absent or empty")
    val_csv = args.splits_dir / "val.csv"
    provenance = validate_training_run(args.run_dir, val_csv)
    manifest = validate_manifest(pd.read_csv(val_csv, dtype=str), args.image_root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = load_models(provenance, device)
    predictions = collect_predictions(models, manifest, args.batch_size, device)
    summary, comparison = summarize_pairs(predictions, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "per_image_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison.csv", index=False)
    write_summary(args.output_dir / "summary.md", summary, comparison)
    source_files = (
        "scripts/evaluate_attention_training_pairs.py", "scripts/evaluate_rotation_pairs.py",
        "scripts/evaluate_glare_pairs.py", "src/attention_training_control.py",
        "src/attention.py", "src/regularization.py", "src/data.py", "src/viz.py",
    )
    members = manifest[["case_id", "patient_id", "label", "label_id"]].sort_values("case_id").to_dict("records")
    metadata = {
        "completed": True, "script_version": SCRIPT_VERSION, "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training_provenance": provenance, "device": str(device), "class_order": rotation.CLASS_ORDER,
        "arms": list(models), "primary_comparison": "learned_ce-minus-fixed_ce",
        "secondary_comparison": "learned_regularized-minus-learned_ce" if "learned_regularized" in models else None,
        "secondary_interpretation": "joint EQ+PF regularization, not EQ-only",
        "split": "val", "test_images_loaded": False, "split_sha256": sha256_file(val_csv),
        "member_fingerprint_sha256": hashlib.sha256(
            json.dumps(members, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "n_images": len(manifest), "n_patients": int(manifest.patient_id.nunique()),
        "n_prediction_rows": len(predictions),
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in source_files},
        "preprocessing": "CropBlackBorder, Resize(256), CenterCrop(224), ImageNet normalization; shared inputs",
        "perturbation": {
            "angles_deg": DEFAULT_ANGLES, "padding_normalized": 0., "padding_rgb": IMAGENET_MEAN,
            "zero_angle": "exact cached identity", "limitation": "synthetic interpolation and corner clipping"},
        "aggregation": {
            "primary": "equal nonzero angles within each image, then image-weighted mean",
            "common_cohort": "separately defined per model pair, both correct at zero; fixed across angles",
            "probability_delta": "rotated minus clean", "zero_denominator": "NaN"},
        "bootstrap": {
            "method": "paired patient cluster percentile 95%", "replicates": args.bootstrap, "seed": args.seed,
            "sampling": "all images and angles travel together with their patient; ratios recomputed",
            "limitation": "conditional on trained models, not training-seed uncertainty"},
        "attention": {
            "statistics": "native-map mean, population std, normalized spatial Shannon entropy",
            "equivariance": "raw attention MSE on rotated valid mask >0.99, 56x56 work grid",
            "limitation": "constant A=0.5 has low error by construction; not anatomical correctness"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {args.output_dir.resolve()} ({len(predictions)} prediction rows)", flush=True)


if __name__ == "__main__":
    main()
