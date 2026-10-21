"""Validation-only, patient-paired evaluation of the candidate-region experiment.

Training seeds are repeated fits of the same patients, never independent samples.
This reports an exploratory technical experiment, not anatomical ROI validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import LABEL_TO_ID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "region_relation_v1"
VARIANTS = ("baseline", "regions", "relations", "no_geometry_relations")
CLASSES = ("lt", "yz", "zz")
COMPARISONS = (("relations", "regions"), ("relations", "baseline"),
               ("relations", "no_geometry_relations"), ("regions", "baseline"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def read_identity_splits(splits_dir):
    """Read identities in all CSVs, without opening or resolving any image."""
    frames = {}
    for name in ("train", "val", "test"):
        frame = pd.read_csv(Path(splits_dir) / f"{name}.csv", dtype=str)
        needed = ["patient_id", "label", "label_id", "path"]
        require(not frame.empty and set(needed).issubset(frame.columns), f"Invalid {name} CSV")
        require(not frame[needed].isna().any().any(), f"Missing {name} identity")
        for field in needed:
            frame[field] = frame[field].str.strip()
            require(not frame[field].eq("").any(), f"Empty {name} {field}")
        expected = frame.label.map(LABEL_TO_ID)
        require(not expected.isna().any() and pd.to_numeric(frame.label_id, errors="coerce").eq(expected).all(),
                f"Invalid {name} label mapping")
        frame["label_id"] = expected.astype(int)
        frame["image_name"] = frame.path.str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
        require(not frame.path.duplicated().any() and not frame.duplicated(["patient_id", "label", "image_name"]).any(),
                f"Duplicate {name} images")
        frames[name] = frame.reset_index(drop=True)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not set(frames[left].patient_id) & set(frames[right].patient_id), f"Patient overlap {left}/{right}")
        require(not set(frames[left].path) & set(frames[right].path), f"Image overlap {left}/{right}")
    return frames


def membership(frame):
    members = frame[["patient_id", "label", "label_id", "image_name"]].sort_values(
        ["patient_id", "label", "image_name"]).to_dict("records")
    digest = hashlib.sha256(json.dumps(members, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return members, digest


def resolve_split_images(frames, split, image_root=None):
    """Resolve all identity paths for overlap; require files only in requested split."""
    mapped = {}
    for name, frame in frames.items():
        paths = [(Path(image_root) / row.image_name if image_root else Path(row.path)).expanduser().resolve()
                 for row in frame.itertuples(index=False)]
        require(len(set(paths)) == len(paths), f"Duplicate resolved {name} image paths")
        mapped[name] = paths
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not set(mapped[left]) & set(mapped[right]), f"Resolved image overlap {left}/{right}")
    frame = frames[split].copy()
    require(all(path.is_file() for path in mapped[split]), f"Missing {split} images; set --image-root")
    frame["path"] = [str(path) for path in mapped[split]]
    return frame


def validate_training_run(run_dir, frames):
    """Reject incomplete, unmatched, stale-source or mismatched-split runs."""
    run_dir = Path(run_dir)
    pair = read_json(run_dir / "pair_metadata.json")
    require(pair.get("protocol_version") == PROTOCOL_VERSION and pair.get("completed") is True,
            "Expected a completed region_relation_v1 experiment")
    require(pair.get("primary_checkpoint") == "last_model.pt" and pair.get("test_images_loaded") is False,
            "Expected fixed-budget validation-only experiment")
    require(pair.get("initial_states_identical") is True and pair.get("training_input_streams_identical") is True,
            "Matched initialization and training input streams must be verified")
    recorded = pair.get("variants", {})
    require(set(recorded) == set(VARIANTS), "All four training variants are required")
    result = {}
    first = None
    for variant in VARIANTS:
        directory = run_dir / variant
        config_path, checkpoint = directory / "run_config.json", directory / "last_model.pt"
        config = read_json(config_path)
        require(config.get("variant") == variant and config.get("protocol_version") == PROTOCOL_VERSION,
                f"{variant}: wrong run identity")
        require(config.get("completed") is True and config.get("checkpoint_reload_verified") is True,
                f"{variant}: incomplete/unverified checkpoint")
        require(config.get("primary_checkpoint") == "last_model.pt" and config.get("test_images_loaded") is False,
                f"{variant}: not fixed-budget validation-only")
        info = recorded[variant]
        require(sha256_file(config_path) == info.get("run_config_sha256"), f"{variant}: run config hash mismatch")
        require(sha256_file(checkpoint) == config.get("checkpoint_sha256", {}).get("last_model.pt")
                == info.get("checkpoint_sha256"), f"{variant}: checkpoint hash mismatch")
        require(config.get("seed") == pair.get("seed"), f"{variant}: seed mismatch")
        require(config.get("batchnorm_buffers_unchanged") is True and config.get("evaluation_split") == "val",
                f"{variant}: unverified BatchNorm policy or non-validation evaluation")
        arguments = config.get("arguments", {})
        epochs = arguments.get("stage1_epochs", -1) + arguments.get("stage2_epochs", -1)
        require(epochs > 0 and config.get("completed_epochs") == epochs, f"{variant}: incomplete fixed budget")
        traces_path = directory / "training_input_trace.csv"
        require(sha256_file(traces_path) == config.get("training_input_trace_sha256")
                == info.get("training_input_trace_sha256"), f"{variant}: input trace hash mismatch")
        traces = pd.read_csv(traces_path).to_dict("records")
        require(traces == config.get("training_input_traces") and len(traces) == epochs,
                f"{variant}: incomplete input traces")
        require(config.get("initial_state_sha256") == info.get("initial_state_sha256")
                and isinstance(config.get("initial_state_sha256"), str)
                and len(config["initial_state_sha256"]) == 64, f"{variant}: invalid initialization hash")
        for name, frame in frames.items():
            members, digest = membership(frame)
            entry = config.get("splits", {}).get(name, {})
            require(entry.get("members") == members and entry.get("membership_sha256") == digest,
                    f"{variant}: {name} membership mismatch")
            require(entry.get("n_images") == len(frame) and entry.get("n_patients") == frame.patient_id.nunique(),
                    f"{variant}: {name} counts mismatch")
        sources = config.get("source_sha256", {})
        require(all(name in sources for name in ("src/region_candidates.py", "src/region_relations.py", "src/region_data.py",
                                                "src/data.py", "src/models.py", "scripts/run_region_relation_ablation.py",
                                                "scripts/evaluate_region_relations.py")),
                f"{variant}: incomplete source provenance")
        for relative, expected in sources.items():
            path = (PROJECT_ROOT / relative).resolve()
            require(path.is_relative_to(PROJECT_ROOT) and path.is_file() and sha256_file(path) == expected,
                    f"{variant}: source changed since training: {relative}")
        model_config = config.get("model_config", {})
        require(model_config.get("variant") == variant and model_config.get("num_classes") == 3,
                f"{variant}: invalid model config")
        if first is not None:
            for field in ("seed", "candidate_config", "splits", "source_sha256", "initial_state_sha256",
                          "training_input_traces", "arguments", "batchnorm_policy", "training_budget",
                          "loss", "pf_loss", "eq_loss", "crop_border", "image_transforms"):
                require(field in first and config.get(field) == first[field], f"Unmatched variant field: {field}")
            for key in ("dropout", "max_candidates", "pretrained", "num_classes"):
                require(model_config.get(key) == first["model_config"].get(key), f"Unmatched model setting: {key}")
        else:
            first = config
        result[variant] = {"config": config, "checkpoint": checkpoint,
                           "checkpoint_sha256": sha256_file(checkpoint), "config_sha256": sha256_file(config_path)}
    return result


def validate_cross_seed_protocol(first, current):
    """Only repeated fits of the same technical protocol may be averaged."""
    for key in ("candidate_config", "model_config", "source_sha256", "image_transforms", "crop_border",
                "batchnorm_policy", "training_budget", "loss", "pf_loss", "eq_loss"):
        require(key in first and first[key] == current.get(key), f"Across-seed protocol mismatch: {key}")
    technical_args = ("batch_size", "stage1_epochs", "stage2_epochs", "stage1_lr", "stage2_lr",
                      "weight_decay", "dropout", "no_pretrained")
    for key in technical_args:
        require(key in first.get("arguments", {})
                and first["arguments"][key] == current.get("arguments", {}).get(key),
                f"Across-seed training setting mismatch: {key}")


def _auc_binary(labels, scores):
    """Mann-Whitney AUC with average ranks for ties; missing class => NaN."""
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion_counts(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    require(labels.shape == predictions.shape and labels.ndim == 1 and len(labels) > 0, "Empty or mismatched predictions")
    require(np.isin(labels, [0, 1, 2]).all() and np.isin(predictions, [0, 1, 2]).all(), "Invalid class ID")
    return np.bincount(3 * labels + predictions, minlength=9).reshape(3, 3)


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=float)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return {"accuracy": tp.sum(axis=-1) / matrix.sum(axis=(-1, -2)), "macro_f1": f1.mean(axis=-1)}


def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    require(probabilities.shape == (len(labels), 3) and np.isfinite(probabilities).all(), "Invalid probability shape/values")
    require((probabilities >= 0).all() and (probabilities <= 1).all()
            and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5), "Invalid probability distribution")
    pred = probabilities.argmax(axis=1)
    matrix = confusion_counts(labels, pred)
    metrics = {key: float(value) for key, value in metrics_from_confusion(matrix).items()}
    aucs = [_auc_binary(labels == idx, probabilities[:, idx]) for idx in range(3)]
    metrics["macro_ovr_auc"] = float(np.mean(aucs)) if np.isfinite(aucs).all() else float("nan")
    metrics["n_images"] = len(labels)
    metrics["n_errors"] = int((labels != pred).sum())
    metrics["yz_to_zz"] = int(matrix[1, 2])
    metrics["zz_to_yz"] = int(matrix[2, 1])
    for idx, label in enumerate(CLASSES):
        metrics[f"{label}_n"] = int(matrix[idx].sum())
        metrics[f"{label}_errors"] = int(matrix[idx].sum() - matrix[idx, idx])
        metrics[f"{label}_auc"] = aucs[idx]
    return metrics


def validate_prediction_pairs(predictions):
    needed = {"seed", "variant", "patient_id", "case_id", "label", "label_id", "pred", "prob_lt", "prob_yz", "prob_zz"}
    require(needed.issubset(predictions.columns) and not predictions.empty, "Missing prediction columns")
    require(not predictions.duplicated(["seed", "variant", "case_id"]).any(), "Duplicate case in a model")
    reference = None
    for seed, seed_frame in predictions.groupby("seed", sort=True):
        require(set(seed_frame.variant) == set(VARIANTS), f"Seed {seed}: incomplete variants")
        for variant in VARIANTS:
            frame = seed_frame[seed_frame.variant == variant].sort_values("case_id")
            identity = frame[["case_id", "patient_id", "label", "label_id"]].reset_index(drop=True)
            require(frame.label.map(LABEL_TO_ID).eq(frame.label_id).all(), "Inconsistent labels")
            classification_metrics(frame.label_id.to_numpy(), frame[[f"prob_{label}" for label in CLASSES]].to_numpy())
            require(np.array_equal(frame.pred.to_numpy(), frame[[f"prob_{label}" for label in CLASSES]].to_numpy().argmax(1)),
                    "Prediction does not match probabilities")
            if reference is None:
                reference = identity
            else:
                require(identity.equals(reference), "Paired comparisons require identical cases, patients and labels in every seed/variant")
    return reference


def paired_comparisons(predictions, bootstrap=10000, seed=20261021):
    """Same patient-cluster draws shared across variants AND training seeds."""
    require(bootstrap >= 0, "bootstrap must be nonnegative")
    reference = validate_prediction_pairs(predictions)
    patients = sorted(reference.patient_id.astype(str).unique())
    seeds = sorted(predictions.seed.unique())
    patient_index = {patient: idx for idx, patient in enumerate(patients)}
    matrices, correctness = {}, {}
    for training_seed in seeds:
        for variant in VARIANTS:
            frame = predictions[(predictions.seed == training_seed) & (predictions.variant == variant)].sort_values("case_id")
            counts = np.zeros((len(patients), 3, 3), dtype=np.int64)
            for row in frame.itertuples(index=False):
                counts[patient_index[str(row.patient_id)], int(row.label_id), int(row.pred)] += 1
            matrices[training_seed, variant] = counts
            correctness[training_seed, variant] = frame.label_id.to_numpy() == frame.pred.to_numpy()
    rng = np.random.default_rng(seed)
    weights = np.zeros((bootstrap, len(patients)), dtype=np.int64)
    for begin in range(0, bootstrap, 1000):
        draws = rng.integers(0, len(patients), size=(min(1000, bootstrap - begin), len(patients)))
        for offset, draw in enumerate(draws):
            weights[begin + offset] = np.bincount(draw, minlength=len(patients))
    point, boot = {}, {}
    for key, counts in matrices.items():
        point[key] = metrics_from_confusion(counts.sum(axis=0))
        boot[key] = metrics_from_confusion(np.einsum("bp,pij->bij", weights, counts)) if bootstrap else {}
    rows = []
    scopes = [(str(s), [s]) for s in seeds]
    if len(seeds) > 1:
        scopes.append(("mean_across_seeds", seeds))
    for scope, included in scopes:
        for proposed, reference_variant in COMPARISONS:
            base_correct = np.stack([correctness[s, reference_variant] for s in included])
            new_correct = np.stack([correctness[s, proposed] for s in included])
            for metric in ("accuracy", "macro_f1"):
                delta = float(np.mean([point[s, proposed][metric] - point[s, reference_variant][metric] for s in included]))
                if bootstrap:
                    differences = np.mean([boot[s, proposed][metric] - boot[s, reference_variant][metric] for s in included], axis=0)
                    low, high = np.quantile(differences, [.025, .975])
                else:
                    low = high = float("nan")
                rows.append({"seed": scope, "comparison": f"{proposed}_minus_{reference_variant}", "metric": metric,
                             "delta": delta, "ci95_low": float(low), "ci95_high": float(high),
                             "n_patients": len(patients), "n_images_per_seed": len(reference), "n_seeds": len(included),
                             "corrected_images_mean_per_seed": float((~base_correct & new_correct).sum(axis=1).mean()),
                             "broken_images_mean_per_seed": float((base_correct & ~new_correct).sum(axis=1).mean()),
                             "bootstrap_unit": "patient", "bootstrap": bootstrap})
    return pd.DataFrame(rows)


def make_summary(predictions):
    validate_prediction_pairs(predictions)
    rows = []
    for (seed, variant), frame in predictions.groupby(["seed", "variant"], sort=True):
        row = {"seed": seed, "variant": variant, "n_patients": frame.patient_id.nunique()}
        row.update(classification_metrics(frame.label_id.to_numpy(), frame[[f"prob_{label}" for label in CLASSES]].to_numpy()))
        row["images_with_candidates"] = int((frame.candidate_count > 0).sum())
        row["images_with_two_candidates"] = int((frame.candidate_count >= 2).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary_markdown(output_dir, summary, comparisons):
    lines = ["# 候选区域关系模型：开发集探索性结果", "",
             "本次仅评估已有 val 划分。候选暗区及周围环带是算法区域，不是医生确认的管腔或解剖 ROI。",
             "数据量、患者数见下表；这些开发数据已多次参与探索，不能作为独立最终验证。",
             "全部模型按固定训练预算使用 last_model.pt；本轮基线也采用本轮共同预处理与 BN 策略，不能替代历史基线结果。", "",
             "|种子|模型|图片/患者|准确率|宏 F1|宏 OVR AUC|右→左/左→右错误|至少两个候选|",
             "|---|---|---|---|---|---|---|---|"]
    for row in summary.itertuples(index=False):
        auc = f"{row.macro_ovr_auc:.6f}" if np.isfinite(row.macro_ovr_auc) else "未定义（类别缺失）"
        lines.append(f"|{row.seed}|{row.variant}|{row.n_images}/{row.n_patients}|{100*row.accuracy:.3f}%|{row.macro_f1:.5f}|{auc}|{row.yz_to_zz}/{row.zz_to_yz}|{row.images_with_two_candidates}|")
    lines += ["", "准确率与宏 F1 的差值、患者成簇配对 bootstrap 95% 区间见 paired_comparison.csv。",
              "多种子汇总是在同一组患者重采样后平均种子内差值；不会把重复训练当作新增患者。区间只描述这些已训练模型在当前患者上的采样不确定性。",
              "corrected 是参考模型错、候选模型对；broken 是参考模型对、候选模型错；多种子行是每种子的平均图片数。",
              "宏 F1 固定覆盖 lt/yz/zz 三类，缺失类别按 0 计；宏 AUC 任一类别缺失时留空，不用剩余类别冒充三类结果。", "",
              "relations−regions 用于判断区域间关系分支的增量；relations−no_geometry_relations 检查显式位置、尺寸和距离特征的增量。",
              "no_geometry_relations 保留相同区域特征和关系网络尺寸，从训练开始到推理均把相对中心位置、面积、宽高五维及距离置零，保留对比度差。",
              "区域外观、掩膜形状和卷积特征仍可能包含空间信息；此对照仅移除显式位置、尺寸和距离输入，不等于去除所有空间信息。",
              "单轮收益、热图或候选外观均不证明发现新解剖区域、临床效益、因果机制或专利创造性。"]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    from src.region_candidates import CandidateConfig
    from src.region_data import RegionDataset
    from src.region_relations import build_region_relation_resnet50
    frames = read_identity_splits(args.splits_dir)
    val = resolve_split_images(frames, "val", args.image_root)
    output = args.output_dir.expanduser().resolve()
    require(not output.exists() or not any(output.iterdir()), "Output directory must be absent or empty")
    run_dirs = args.run_dirs if args.run_dirs else [args.run_dir]
    records, provenance, seen_seeds = [], [], set()
    device = torch.device(args.device)
    first_seed_protocol = None
    for run_dir in run_dirs:
        provenance_by_variant = validate_training_run(run_dir, frames)
        first = provenance_by_variant["baseline"]["config"]
        if first_seed_protocol is None:
            first_seed_protocol = first
        else:
            validate_cross_seed_protocol(first_seed_protocol, first)
        training_seed = first["seed"]
        require(training_seed not in seen_seeds, "Duplicate training seed across run directories")
        seen_seeds.add(training_seed)
        candidate_config = CandidateConfig(**first["candidate_config"])
        dataset = RegionDataset(val, training=False, candidate_config=candidate_config, seed=training_seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for variant in VARIANTS:
            entry = provenance_by_variant[variant]
            model_config = dict(entry["config"]["model_config"])
            model_config["pretrained"] = False
            model = build_region_relation_resnet50(**model_config).to(device)
            checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
            for key, expected in (("variant", variant), ("protocol_version", PROTOCOL_VERSION),
                                  ("seed", training_seed), ("candidate_config", asdict(candidate_config)),
                                  ("model_config", entry["config"]["model_config"]),
                                  ("epoch", entry["config"]["completed_epochs"])):
                # JSON config turns tuples into lists; preserve equivalent values.
                require(json.dumps(checkpoint.get(key), sort_keys=True) == json.dumps(expected, sort_keys=True),
                        f"{variant}: checkpoint metadata mismatch: {key}")
            expected_membership = {name: membership(frame)[1] for name, frame in frames.items()}
            require(checkpoint.get("split_membership_sha256") == expected_membership, "Checkpoint split membership mismatch")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    candidates = {key: value.to(device) for key, value in batch["candidates"].items()}
                    logits = model(batch["image"].to(device), candidates)
                    probabilities = logits.softmax(dim=1).cpu().numpy()
                    for offset, index in enumerate(batch["index"].tolist()):
                        identity = val.iloc[index]
                        probability = probabilities[offset]
                        count = int(batch["candidates"]["valid"][offset].sum().item())
                        pred = int(probability.argmax())
                        records.append({"seed": training_seed, "variant": variant, "patient_id": str(identity.patient_id),
                                        "case_id": str(batch["case_id"][offset]), "image_name": identity.image_name,
                                        "label": identity.label, "label_id": int(identity.label_id), "pred": pred,
                                        "pred_label": CLASSES[pred], "correct": int(pred == identity.label_id),
                                        "candidate_count": count, "has_two_candidates": int(count >= 2),
                                        **{f"prob_{label}": float(probability[idx]) for idx, label in enumerate(CLASSES)}})
            provenance.append({"seed": training_seed, "variant": variant, "run_dir": str(Path(run_dir).resolve()),
                               "checkpoint_sha256": entry["checkpoint_sha256"], "config_sha256": entry["config_sha256"]})
            del model, checkpoint
    predictions = pd.DataFrame(records)
    summary = make_summary(predictions)
    comparisons = paired_comparisons(predictions, args.bootstrap, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "per_image_predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparison.csv", index=False)
    metadata = {"protocol_version": PROTOCOL_VERSION, "split": "val", "test_images_loaded": False,
                "bootstrap": args.bootstrap, "evaluation_seed": args.seed, "runs": provenance,
                "evaluation_source_sha256": sha256_file(Path(__file__)), "class_order": list(CLASSES)}
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_markdown(output, summary, comparisons)
    print(f"Completed validation evaluation: {output} ({len(predictions)} prediction rows)")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--run-dirs", type=Path, nargs="+")
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.bootstrap < 0 or not 0 <= args.seed < 2**32:
        parser.error("batch-size must be positive; bootstrap nonnegative; seed in [0,2**32)")
    return args


if __name__ == "__main__":
    run(parse_args())
