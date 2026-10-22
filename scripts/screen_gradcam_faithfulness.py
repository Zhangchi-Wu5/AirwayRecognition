"""Validation-only Grad-CAM perturbation screen for fixed-budget ResNet baselines.

This measures a heatmap's association with model sensitivity under specified
perturbations. It does not validate anatomy, causality, or clinical usefulness.
No training, image filtering, checkpoint selection, or weight saving occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from scripts.evaluate_region_relations import (
    PROJECT_ROOT, CLASSES, membership, read_identity_splits, read_json, require,
    resolve_split_images, sha256_file, validate_cross_seed_protocol,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD, get_eval_geometry_transforms
from src.region_relations import build_region_relation_resnet50

PROTOCOL_VERSION = "gradcam_faithfulness_screen_v1"
FILLS = ("blur", "ring_mean")
MAX_BORDER_FRACTION = .05
DARK_THRESHOLD = 24
RING_WIDTH = 8
BLUR_KERNEL = 31
BLUR_SIGMA = 8.0
EFFECTS = ("high_minus_random_prob_drop", "high_minus_low_prob_drop",
           "high_minus_random_logodds_drop", "high_minus_low_logodds_drop")


def stable_seed(*values):
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def state_digest(model):
    """Fingerprint all parameters and buffers, including frozen/unused heads."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_baseline_run(run_dir, frames):
    """Validate baseline provenance without requiring other arms' weight files."""
    run_dir = Path(run_dir)
    pair_path = run_dir / "pair_metadata.json"
    pair = read_json(pair_path)
    require(pair.get("protocol_version") == "region_relation_v1" and pair.get("completed") is True,
            "Expected a completed region_relation_v1 experiment")
    require(pair.get("primary_checkpoint") == "last_model.pt" and pair.get("test_images_loaded") is False
            and pair.get("evaluation_split") == "val", "Expected validation-only fixed-budget run")
    require(pair.get("initial_states_identical") is True and pair.get("training_input_streams_identical") is True,
            "Matched initialization and training streams were not verified")
    info = pair.get("variants", {}).get("baseline", {})
    directory = run_dir / "baseline"
    config_path, checkpoint_path = directory / "run_config.json", directory / "last_model.pt"
    config = read_json(config_path)
    require(config.get("variant") == "baseline" and config.get("protocol_version") == "region_relation_v1",
            "Wrong baseline run identity")
    require(config.get("completed") is True and config.get("checkpoint_reload_verified") is True,
            "Baseline checkpoint is incomplete or unverified")
    require(config.get("primary_checkpoint") == "last_model.pt" and config.get("test_images_loaded") is False
            and config.get("evaluation_split") == "val" and config.get("crop_border") is False,
            "Expected uncropped-border, fixed-budget validation baseline")
    require(config.get("seed") == pair.get("seed") and isinstance(config.get("seed"), int), "Seed mismatch")
    require(config.get("batchnorm_buffers_unchanged") is True, "Unverified BatchNorm policy")
    require(sha256_file(config_path) == info.get("run_config_sha256"), "Baseline config hash mismatch")
    checkpoint_hash = sha256_file(checkpoint_path)
    require(checkpoint_hash == config.get("checkpoint_sha256", {}).get("last_model.pt")
            == info.get("checkpoint_sha256"), "Baseline checkpoint hash mismatch")
    arguments = config.get("arguments", {})
    epochs = arguments.get("stage1_epochs", -1) + arguments.get("stage2_epochs", -1)
    require(epochs > 0 and config.get("completed_epochs") == epochs, "Incomplete fixed training budget")
    trace_path = directory / "training_input_trace.csv"
    require(sha256_file(trace_path) == config.get("training_input_trace_sha256")
            == info.get("training_input_trace_sha256"), "Input trace hash mismatch")
    traces = pd.read_csv(trace_path).to_dict("records")
    require(traces == config.get("training_input_traces") and len(traces) == epochs, "Incomplete training traces")
    initial = config.get("initial_state_sha256")
    require(isinstance(initial, str) and len(initial) == 64 and initial == info.get("initial_state_sha256"),
            "Invalid initialization fingerprint")
    for split, frame in frames.items():
        members, digest = membership(frame)
        recorded = config.get("splits", {}).get(split, {})
        require(recorded.get("members") == members and recorded.get("membership_sha256") == digest,
                f"{split} membership mismatch")
        require(recorded.get("n_images") == len(frame) and recorded.get("n_patients") == frame.patient_id.nunique(),
                f"{split} count mismatch")
    sources = config.get("source_sha256", {})
    expected_sources = ("src/region_candidates.py", "src/region_relations.py", "src/region_data.py",
                        "src/data.py", "src/models.py", "scripts/run_region_relation_ablation.py",
                        "scripts/evaluate_region_relations.py")
    require(all(relative in sources for relative in expected_sources), "Incomplete training source provenance")
    for relative, expected in sources.items():
        path = (PROJECT_ROOT / relative).resolve()
        require(path.is_relative_to(PROJECT_ROOT) and path.is_file() and sha256_file(path) == expected,
                f"Training source changed: {relative}")
    model_config = config.get("model_config", {})
    require(model_config.get("variant") == "baseline" and model_config.get("num_classes") == 3,
            "Wrong baseline model configuration")
    return {"config": config, "checkpoint": checkpoint_path, "checkpoint_sha256": checkpoint_hash,
            "config_sha256": sha256_file(config_path), "pair_metadata_sha256": sha256_file(pair_path)}


def load_baseline(validated, device):
    config = validated["config"]
    checkpoint = torch.load(validated["checkpoint"], map_location="cpu", weights_only=True)
    for key in ("variant", "model_config", "candidate_config", "protocol_version", "seed"):
        # Training checkpoints retain tuple fields; JSON provenance uses lists.
        require(json.dumps(checkpoint.get(key), sort_keys=True) == json.dumps(config.get(key), sort_keys=True),
                f"Checkpoint {key} mismatch")
    require(checkpoint.get("epoch") == config["completed_epochs"], "Wrong checkpoint epoch")
    expected = {name: value["membership_sha256"] for name, value in config["splits"].items()}
    require(checkpoint.get("split_membership_sha256") == expected, "Checkpoint split membership mismatch")
    model = build_region_relation_resnet50(**{**config["model_config"], "pretrained": False})
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def target_scores(logits, targets):
    """Original-class softmax probability and numerically stable one-vs-rest log odds."""
    require(logits.ndim == 2 and targets.shape == logits.shape[:1] and logits.shape[1] >= 2,
            "Invalid target-score dimensions")
    require(bool(torch.isfinite(logits).all()), "Nonfinite model logits")
    index = targets.long()[:, None]
    target = logits.gather(1, index).squeeze(1)
    others = logits.scatter(1, index, float("-inf"))
    return logits.softmax(1).gather(1, index).squeeze(1), target - others.logsumexp(1)


def gradcam(model, target_layer, images):
    """Grad-CAM for each image's original predicted class; remove hook even on failure.

    Model must already be in eval mode. Parameter .grad fields are not populated.
    """
    require(not model.training, "Grad-CAM requires eval mode")
    activations = []
    hook = target_layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    try:
        with torch.enable_grad():
            inputs = images.detach().clone().requires_grad_(True)
            logits = model(inputs, None)
            require(logits.ndim == 2 and bool(torch.isfinite(logits).all()), "Nonfinite or invalid clean logits")
            require(len(activations) == 1 and activations[0].ndim == 4, "Target layer must fire once with BCHW output")
            targets = logits.detach().argmax(1)
            features = activations[0]
            gradients = torch.autograd.grad(logits.gather(1, targets[:, None]).sum(), features)[0]
            raw = (gradients.mean((2, 3), keepdim=True) * features).sum(1).relu()
            raw = F.interpolate(raw[:, None], size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            maps, statuses = [], []
            for item in raw.detach().cpu().numpy():
                if not np.isfinite(item).all():
                    maps.append(np.full(item.shape, np.nan, dtype=np.float32)); statuses.append("nonfinite_cam")
                elif float(item.max()) <= 0:
                    maps.append(np.zeros_like(item)); statuses.append("zero_cam")
                elif float(item.max() - item.min()) <= max(1e-12, 1e-6 * float(item.max())):
                    maps.append(np.zeros_like(item)); statuses.append("flat_cam")
                else:
                    # Divide only: retain CAM mass for the border-mass diagnostic.
                    maps.append(item / item.max()); statuses.append("ok")
            return {"logits": logits.detach().cpu().numpy(), "targets": targets.cpu().numpy(),
                    "maps": np.stack(maps), "statuses": statuses}
    finally:
        hook.remove()


def window_means(array, side):
    """All stride-one square means, without changing window shape by CAM rank."""
    array = np.asarray(array, dtype=np.float64)
    require(array.ndim == 2 and 1 <= side <= min(array.shape), "Invalid square size")
    integral = np.pad(array, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side]
            + integral[:-side, :-side]) / (side * side)


def choose_windows(cam, border, coverage, random_placements, case_id, seed):
    cam, border = np.asarray(cam), np.asarray(border, dtype=bool)
    require(cam.ndim == 2 and cam.shape == border.shape, "Mismatched CAM/border shape")
    require(0 < coverage < 1 and random_placements >= 1, "Invalid window protocol")
    side = max(1, min(min(cam.shape), int(round(math.sqrt(coverage * cam.size)))))
    result = {"status": "ok", "side": side, "actual_coverage": side * side / cam.size,
              "eligible_count": 0, "windows": []}
    if not np.isfinite(cam).all():
        result["status"] = "nonfinite_cam"; return result
    if float(cam.max()) <= 0 or float(np.ptp(cam)) <= max(1e-12, 1e-6 * float(cam.max())):
        result["status"] = "flat_cam"; return result
    eligible = np.argwhere(window_means(border, side) <= MAX_BORDER_FRACTION + 1e-12)
    result["eligible_count"] = len(eligible)
    if not len(eligible):
        result["status"] = "no_eligible_window"; return result
    means = window_means(cam, side)[eligible[:, 0], eligible[:, 1]]
    if float(np.ptp(means)) <= max(1e-12, 1e-6 * float(np.max(np.abs(means)))):
        result["status"] = "flat_eligible_window_scores"; return result
    # A separate RNG ensures tie counts/CAM values cannot alter random controls.
    ties = np.random.default_rng(stable_seed(seed, case_id, side, "ties"))
    for region, value in (("high", means.max()), ("low", means.min())):
        tied = np.flatnonzero(np.isclose(means, value, atol=1e-12, rtol=1e-8))
        y, x = eligible[ties.choice(tied)]
        result["windows"].append({"region": region, "index": 0, "y": int(y), "x": int(x)})
    rng = np.random.default_rng(stable_seed(seed, case_id, side, "random"))
    for index, selection in enumerate(rng.integers(len(eligible), size=random_placements)):
        y, x = eligible[selection]
        result["windows"].append({"region": "random", "index": index, "y": int(y), "x": int(x)})
    return result


def blur_rgb(rgb):
    """One fixed full-input blur; callers reuse it for every window on this image."""
    require(min(rgb.shape[:2]) > BLUR_KERNEL // 2, "Image too small for fixed blur kernel")
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1)
    result = TF.gaussian_blur(tensor, [BLUR_KERNEL, BLUR_KERNEL], [BLUR_SIGMA, BLUR_SIGMA])
    return result.permute(1, 2, 0).numpy()


def perturb_rgb(rgb, window, side, fill, border, blurred_rgb=None):
    """Replace a square only; input/output RGB float arrays have range [0,1]."""
    rgb = np.asarray(rgb, dtype=np.float32)
    border = np.asarray(border, dtype=bool)
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.shape[:2] == border.shape,
            "Invalid RGB/border shape")
    require(np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1, "RGB must be finite in [0,1]")
    y, x = int(window["y"]), int(window["x"])
    require(side > 0 and 0 <= y <= rgb.shape[0] - side and 0 <= x <= rgb.shape[1] - side, "Invalid window")
    output = rgb.copy()
    fallback = "none"
    if fill == "blur":
        replacement = blur_rgb(rgb) if blurred_rgb is None else np.asarray(blurred_rgb)
        require(replacement.shape == rgb.shape and np.isfinite(replacement).all(), "Invalid cached blur")
        output[y:y + side, x:x + side] = replacement[y:y + side, x:x + side]
    elif fill == "ring_mean":
        ring = np.zeros(border.shape, dtype=bool)
        ring[max(0, y - RING_WIDTH):min(rgb.shape[0], y + side + RING_WIDTH),
             max(0, x - RING_WIDTH):min(rgb.shape[1], x + side + RING_WIDTH)] = True
        ring[y:y + side, x:x + side] = False
        ring &= ~border
        if not ring.any():
            ring = ~border
            fallback = "image_nonborder_mean"
        if not ring.any():
            ring = np.ones(border.shape, dtype=bool)
            fallback = "full_image_mean"
        output[y:y + side, x:x + side] = rgb[ring].mean(0)
    else:
        raise ValueError(f"Unknown fill: {fill}")
    return output, fallback


def normalize_rgb(rgb):
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def _bootstrap_means(frame, n_bootstrap, seed):
    patients = sorted(frame.patient_id.unique())
    grouped = frame.groupby("patient_id", sort=True)[list(EFFECTS)].agg(["sum", "count"])
    sums = np.stack([grouped[(effect, "sum")].reindex(patients).to_numpy() for effect in EFFECTS], axis=1)
    counts = grouped[(EFFECTS[0], "count")].reindex(patients).to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for offset in range(0, n_bootstrap, 1000):
        selected = rng.integers(len(patients), size=(min(1000, n_bootstrap - offset), len(patients)))
        draws.append(sums[selected].sum(1) / counts[selected].sum(1)[:, None])
    return np.quantile(np.concatenate(draws), [.025, .975], axis=0)


def summarize_effects(effects, per_image, n_bootstrap, seed):
    """Image-weighted paired means with patient-cluster bootstrap.

    Multi-seed means first average within image over the same seed set, then
    bootstrap patients. This exactly shares patient draws across repeated fits.
    """
    columns = ["scope", "training_seed", "coverage", "actual_coverage", "fill", "n_images", "n_patients",
               "n_training_seeds", "n_total_images", "n_excluded_images", "estimand", "metric", "mean",
               "ci_low", "ci_high"]
    if effects.empty:
        return pd.DataFrame(columns=columns)
    require(n_bootstrap >= 1 and not per_image.empty, "Invalid bootstrap/identity input")
    require(not per_image.duplicated(["seed", "case_id"]).any(), "Duplicate clean identities")
    require(not effects.duplicated(["seed", "case_id", "coverage", "fill"]).any(), "Duplicate paired effects")
    require(np.isfinite(effects[list(EFFECTS)].to_numpy()).all(), "Nonfinite paired effects")
    seeds = sorted(per_image.seed.unique())
    reference = per_image[per_image.seed == seeds[0]][["case_id", "patient_id"]].sort_values("case_id").reset_index(drop=True)
    for training_seed in seeds:
        current = per_image[per_image.seed == training_seed][["case_id", "patient_id"]].sort_values("case_id").reset_index(drop=True)
        require(current.equals(reference), "Multi-seed evaluation must use identical images and patients")
    identity = per_image[["seed", "case_id", "patient_id"]]
    checked = effects.merge(identity, on=["seed", "case_id"], how="left", suffixes=("", "_expected"), validate="many_to_one")
    require(checked.patient_id.eq(checked.patient_id_expected).all(), "Effects contain unknown/mismatched identities")
    rows = []
    for (coverage, fill), group in effects.groupby(["coverage", "fill"], sort=True):
        cohorts = [("per_seed", training_seed, group[group.seed == training_seed], 1) for training_seed in seeds]
        if len(seeds) > 1:
            common = group.groupby("case_id").seed.nunique()
            common = set(common[common == len(seeds)].index)
            paired = group[group.case_id.isin(common)]
            averaged = paired.groupby(["case_id", "patient_id"], as_index=False)[list(EFFECTS)].mean()
            cohorts.append(("across_seeds_common_images", "all", averaged, len(seeds)))
        for scope, training_seed, frame, count_seeds in cohorts:
            if frame.empty:
                continue
            intervals = _bootstrap_means(frame, n_bootstrap, stable_seed(seed, float(coverage), fill, "patient_bootstrap"))
            for index, metric in enumerate(EFFECTS):
                rows.append({"scope": scope, "training_seed": training_seed, "coverage": coverage,
                             "actual_coverage": group.actual_coverage.iloc[0], "fill": fill,
                             "n_images": len(frame), "n_patients": frame.patient_id.nunique(),
                             "n_training_seeds": count_seeds, "n_total_images": len(reference),
                             "n_excluded_images": len(reference) - len(frame),
                             "estimand": "image_mean_conditional_on_evaluable_maps_and_windows",
                             "metric": metric, "mean": frame[metric].mean(),
                             "ci_low": intervals[0, index], "ci_high": intervals[1, index]})
    return pd.DataFrame(rows, columns=columns)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _render_example(path, rgb, cam, status, plan, perturbations, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure, axes = plt.subplots(1, 5, figsize=(16, 4.1))
    for ax in axes:
        ax.imshow(rgb); ax.axis("off")
    axes[0].set_title("Actual model input")
    axes[1].set_title(f"Grad-CAM: {status}")
    if status == "ok":
        axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=.45)
    for ax, region in zip(axes[2:], ("high", "low", "random")):
        windows = [window for window in plan.get("windows", []) if window["region"] == region]
        ax.set_title(region if windows else f"{region}: unavailable")
        if windows:
            window = windows[0]
            ax.add_patch(Rectangle((window["x"] - .5, window["y"] - .5), plan["side"], plan["side"],
                                   fill=False, edgecolor="cyan", linewidth=2))
            scores = [row for row in perturbations if row["region"] == region and row["placement_index"] == window["index"]]
            ax.set_title(region + "\n" + " / ".join(f"{row['fill']}: Δp={row['prob_drop']:+.3f}" for row in scores), fontsize=9)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .88))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _summary_markdown(summary, per_image, windows, args):
    lines = ["# Grad-CAM 热图依赖初筛", "",
             "仅使用已有 ResNet baseline 的 last_model.pt，在原 val 图像上增加推理；未训练、未保存新权重、未筛除原始图片。",
             "目标始终为原图预测类别。正差表示遮挡高热区域比对应对照更降低该类别得分；不设置通过阈值。",
             "热图和干预均在实际 Resize(256)、CenterCrop(224) 输入上计算。近黑且与图像边界连通的区域按固定规则标记；方窗中该区域占比不超过 5%。",
             "因此结果只描述当前视野内可放置方窗的区域，不能据此排除模型利用黑边。规则黑区也不是人工确认的视野分割。", "",
             "|训练种子|全部图像|有效热图|零热图|平坦热图|非有限热图|", "|---|---:|---:|---:|---:|---:|"]
    for training_seed, group in per_image.groupby("seed", sort=True):
        counts = group.cam_status.value_counts()
        lines.append(f"|{training_seed}|{len(group)}|{counts.get('ok', 0)}|{counts.get('zero_cam', 0)}|{counts.get('flat_cam', 0)}|{counts.get('nonfinite_cam', 0)}|")
    lines += ["", "|训练种子|名义覆盖率|可评估图像|无合格方窗|合格窗分数平坦|", "|---|---:|---:|---:|---:|"]
    for (training_seed, coverage), group in windows.groupby(["seed", "coverage"], sort=True):
        lines.append(f"|{training_seed}|{coverage:.0%}|{group.status.eq('ok').sum()}|{group.status.eq('no_eligible_window').sum()}|{group.status.eq('flat_eligible_window_scores').sum()}|")
    if per_image.seed.nunique() > 1:
        lines += ["", "|名义覆盖率|全部种子共同可评估/请求图像|", "|---:|---:|"]
        for coverage, group in windows.groupby("coverage", sort=True):
            common = group.groupby("case_id").status.apply(lambda values: values.eq("ok").all())
            lines.append(f"|{coverage:.0%}|{int(common.sum())}/{len(common)}|")
    lines += ["", "下表是高热窗 − 随机窗平均的原类别概率下降差；单位为百分点。高热窗 − 低热窗及 log-odds 结果见 summary.csv，均完整输出。",
              "", "|范围|种子|填充|名义覆盖率|可评估图像/全部图像/患者|平均差（百分点）|95% 患者区间|", "|---|---|---|---:|---|---:|---|"]
    selected = summary[summary.metric.eq(EFFECTS[0])] if not summary.empty else summary
    for row in selected.itertuples(index=False):
        lines.append(f"|{row.scope}|{row.training_seed}|{row.fill}|{row.coverage:.0%}|{row.n_images}/{row.n_total_images}/{row.n_patients}|{100*row.mean:+.3f}|[{100*row.ci_low:+.3f}, {100*row.ci_high:+.3f}]|")
    if summary.empty:
        lines.append("\n没有可评估的热图/方窗组合；所有输入仍记录在 per_image.csv 和 window_status.csv。")
    lines += ["", f"每图每档有 {args.random_placements} 个等面积、同形状随机方窗（有放回），先在图内平均；{args.bootstrap} 次患者成簇重采样。",
              "多种子只在所有种子均可评估的共同图像上先平均种子内差值，再共享患者重采样；训练种子和随机位置不会增加独立患者数。",
              "区间条件于已训练模型和当前可评估图像，不能代表独立新数据验证。重复使用的开发集仅供探索。",
              "blur 使用 31×31、σ=8 的高斯模糊；ring_mean 使用窗外 8 像素环带且排除规则黑区，回退情况逐次记录。仅替换方窗内像素。",
              "两种填充和全部面积档位分别报告，不能只选择较有利的一项。模糊可能保留结构；两类干预都可能造成分布偏移。",
              "本轮只做扰动初筛，尚未进行参数随机化检验；热图有效不等于解剖解释正确，无人工 ROI 不能验证解剖定位。",
              "示例按类别和固定身份散列在计算热图前选择，包含无效热图；不是按结果挑选。实际覆盖率、黑边内热图响应占比、输入哈希和模型状态校验见明细与 metadata.json。", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    output = Path(args.output_dir)
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "output-dir must be absent or empty; existing results are not overwritten")
    require(args.batch_size > 0 and args.random_placements > 0 and args.bootstrap > 0 and args.render_per_class >= 0,
            "Batch size, random placements and bootstrap must be positive")
    require(len(args.coverages) > 0 and len(set(args.coverages)) == len(args.coverages)
            and all(0 < value < 1 for value in args.coverages), "Coverages must be distinct values between 0 and 1")
    require(len(args.run_dirs) > 0, "At least one run directory is required")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable; run on the server or use --device cpu")
    frames = read_identity_splits(args.splits_dir)
    validated = [validate_baseline_run(directory, frames) for directory in args.run_dirs]
    require(len({item["config"]["seed"] for item in validated}) == len(validated), "Duplicate training seeds")
    for item in validated[1:]:
        validate_cross_seed_protocol(validated[0]["config"], item["config"])
    validation = resolve_split_images(frames, "val", args.image_root).sort_values(
        ["patient_id", "image_name"]).reset_index(drop=True)
    validation["case_id"] = validation.patient_id + ":" + validation.image_name
    require(not validation.case_id.duplicated().any(), "Duplicate validation case IDs")
    selected_examples = set()
    for _, group in validation.groupby("label", sort=True):
        selected_examples.update(sorted(group.case_id, key=lambda case: stable_seed(args.seed, case, "examples"))[:args.render_per_class])
    output.mkdir(parents=True, exist_ok=True)
    (output / "examples").mkdir()
    metadata = {"protocol_version": PROTOCOL_VERSION, "completed": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {key: [str(v) for v in value] if key == "run_dirs" else str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "training_performed": False, "weights_saved": False, "test_images_loaded": False,
                "evaluation_split": "val", "target_class": "original_predicted_class_fixed_for_all_perturbations",
                "cam_layer": "backbone.layer4", "cam_gradient_target": "original_predicted_class_logit",
                "input_geometry": "Resize(256), CenterCrop(224), RGB / 255, ImageNet normalization; crop_border=False",
                "fills": {"blur": {"kernel": BLUR_KERNEL, "sigma": BLUR_SIGMA},
                          "ring_mean": {"width": RING_WIDTH, "exclude_rule_border": True,
                                        "fallback_order": ["image_nonborder_mean", "full_image_mean"]}},
                "border_rule": {"dark_luminance_threshold_0_255": DARK_THRESHOLD, "edge_connected_only": True,
                                "max_fraction_in_window": MAX_BORDER_FRACTION},
                "random_windows": "uniform among eligible top-left positions, with replacement; independent of CAM values",
                "example_selection": "per class lowest stable identity hashes, selected before model inference",
                "selected_example_cases": sorted(selected_examples),
                "bootstrap_estimand": "image-weighted paired effect, conditional on evaluable CAM/window; shared patients across seeds",
                "split_csv_sha256": {name: sha256_file(Path(args.splits_dir) / f"{name}.csv") for name in frames},
                "split_membership_sha256": {name: membership(frame)[1] for name, frame in frames.items()},
                "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in
                                  ("scripts/screen_gradcam_faithfulness.py", "src/viz.py", "src/data.py", "src/region_relations.py")},
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": str(device)},
                "runs": [], "image_sha256": {row.case_id: sha256_file(row.path) for row in validation.itertuples(index=False)}}
    _write_json(output / "metadata.json", metadata)
    from src.viz import detect_dark_border_mask
    geometry = get_eval_geometry_transforms(crop_border=False)
    # Geometry only once; targets/CAMs remain model-specific. No train/test image is opened.
    images = []
    for row in validation.itertuples(index=False):
        with Image.open(row.path) as source:
            rgb = np.asarray(geometry(source.convert("RGB")), dtype=np.float32) / 255.0
        border = detect_dark_border_mask((rgb * 255).round().astype(np.uint8), dark_threshold=DARK_THRESHOLD)
        images.append((row, rgb, border, blur_rgb(rgb)))
    clean_rows, perturbation_rows, effect_rows, window_rows = [], [], [], []
    for item in validated:
        training_seed = item["config"]["seed"]
        model = load_baseline(item, device)
        before = state_digest(model)
        for image_index, (row, rgb, border, blurred) in enumerate(images):
            result = gradcam(model, model.backbone.layer4, normalize_rgb(rgb)[None].to(device))
            logits = torch.from_numpy(result["logits"])
            target = int(result["targets"][0])
            probability, logodds = target_scores(logits, torch.tensor([target]))
            clean_probability, clean_logodds = float(probability[0]), float(logodds[0])
            cam, status = result["maps"][0], result["statuses"][0]
            common = {"seed": training_seed, "case_id": row.case_id, "patient_id": row.patient_id,
                      "label": row.label, "label_id": int(row.label_id), "target": target,
                      "target_label": CLASSES[target], "clean_correct": target == row.label_id,
                      "clean_probability": clean_probability, "clean_logodds": clean_logodds}
            clean_rows.append({**common, "image_name": row.image_name, "cam_status": status,
                               "border_fraction": float(border.mean()),
                               "cam_border_mass": float(cam[border].sum() / cam.sum()) if status == "ok" else float("nan")})
            for coverage in args.coverages:
                plan = choose_windows(cam, border, coverage, args.random_placements, row.case_id, args.seed)
                if status != "ok":
                    plan["status"] = status
                window_rows.append({**common, "coverage": coverage, "actual_coverage": plan["actual_coverage"],
                                    "side": plan["side"], "status": plan["status"], "eligible_count": plan["eligible_count"]})
                current_rows = []
                if plan["status"] == "ok":
                    for fill in FILLS:
                        jobs = plan["windows"]
                        for offset in range(0, len(jobs), args.batch_size):
                            batch_jobs = jobs[offset:offset + args.batch_size]
                            modified = [perturb_rgb(rgb, window, plan["side"], fill, border, blurred_rgb=blurred) for window in batch_jobs]
                            batch = torch.stack([normalize_rgb(value[0]) for value in modified]).to(device)
                            with torch.inference_mode():
                                changed_logits = model(batch, None)
                                changed_probability, changed_logodds = target_scores(
                                    changed_logits, torch.full((len(batch_jobs),), target, device=device, dtype=torch.long))
                            for index, (window, (_, fallback)) in enumerate(zip(batch_jobs, modified)):
                                y, x, side = window["y"], window["x"], plan["side"]
                                current_rows.append({**common, "coverage": coverage, "actual_coverage": plan["actual_coverage"],
                                                     "fill": fill, "region": window["region"], "placement_index": window["index"],
                                                     "y": y, "x": x, "side": side, "fill_fallback": fallback,
                                                     "window_border_fraction": float(border[y:y+side, x:x+side].mean()),
                                                     "window_cam_mean": float(cam[y:y+side, x:x+side].mean()),
                                                     "perturbed_probability": float(changed_probability[index]),
                                                     "perturbed_logodds": float(changed_logodds[index]),
                                                     "prob_drop": clean_probability - float(changed_probability[index]),
                                                     "logodds_drop": clean_logodds - float(changed_logodds[index]),
                                                     "prediction_flipped": int(changed_logits[index].argmax()) != target})
                        by_region = {region: [entry for entry in current_rows if entry["fill"] == fill and entry["region"] == region]
                                     for region in ("high", "low", "random")}
                        effects = {}
                        for score in ("prob_drop", "logodds_drop"):
                            high = by_region["high"][0][score]
                            effects[f"high_minus_random_{score}"] = high - np.mean([entry[score] for entry in by_region["random"]])
                            effects[f"high_minus_low_{score}"] = high - by_region["low"][0][score]
                        effect_rows.append({**common, "coverage": coverage, "actual_coverage": plan["actual_coverage"], "fill": fill, **effects})
                    perturbation_rows.extend(current_rows)
                if row.case_id in selected_examples:
                    token = hashlib.sha256(row.case_id.encode()).hexdigest()[:12]
                    _render_example(output / "examples" / f"seed_{training_seed}_{row.label}_{token}_coverage_{coverage:g}.png",
                                    rgb, cam, status, plan, current_rows,
                                    f"Seed {training_seed}; label={row.label}; original prediction={CLASSES[target]}; "
                                    f"p={clean_probability:.3f}; nominal area={coverage:.0%}; actual={plan['actual_coverage']:.3%}")
            if (image_index + 1) % 10 == 0 or image_index + 1 == len(images):
                print(f"Grad-CAM screen seed {training_seed}: {image_index + 1}/{len(images)} images", flush=True)
        after = state_digest(model)
        require(before == after, "Model parameters or buffers changed during diagnostic")
        metadata["runs"].append({"seed": training_seed, "checkpoint": str(item["checkpoint"]),
                                 "checkpoint_sha256": item["checkpoint_sha256"], "config_sha256": item["config_sha256"],
                                 "pair_metadata_sha256": item["pair_metadata_sha256"],
                                 "state_sha256_before": before, "state_sha256_after": after,
                                 "model_state_unchanged": True, "completed_epochs": item["config"]["completed_epochs"]})
        _write_json(output / "metadata.json", metadata)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    clean = pd.DataFrame(clean_rows)
    perturbations = pd.DataFrame(perturbation_rows)
    effects = pd.DataFrame(effect_rows)
    windows = pd.DataFrame(window_rows)
    if perturbations.empty:
        perturbations = pd.DataFrame(columns=[*clean.columns, "coverage", "fill", "region", "placement_index", "prob_drop", "logodds_drop"])
    if effects.empty:
        effects = pd.DataFrame(columns=[*clean.columns, "coverage", "actual_coverage", "fill", *EFFECTS])
    summary = summarize_effects(effects, clean, args.bootstrap, args.seed)
    for filename, frame in (("per_image.csv", clean), ("perturbations.csv", perturbations), ("paired_effects.csv", effects),
                            ("window_status.csv", windows), ("summary.csv", summary)):
        frame.to_csv(output / filename, index=False)
    (output / "summary.md").write_text(_summary_markdown(summary, clean, windows, args), encoding="utf-8")
    metadata.update(completed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, n_images=len(validation),
                    n_patients=int(validation.patient_id.nunique()), n_training_seeds=len(validated),
                    n_perturbations=len(perturbations), n_paired_effect_rows=len(effects),
                    output_sha256={name: sha256_file(output / name) for name in
                                   ("per_image.csv", "perturbations.csv", "paired_effects.csv", "window_status.csv", "summary.csv", "summary.md")})
    _write_json(output / "metadata.json", metadata)
    print(f"Completed Grad-CAM screen: {output.resolve()}", flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coverages", nargs="+", type=float, default=[.05, .10, .20])
    parser.add_argument("--random-placements", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--render-per-class", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/airway_gradcam_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/airway_gradcam_cache")
    run(parse_args())
