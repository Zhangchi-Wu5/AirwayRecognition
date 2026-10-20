"""Patient-paired statistics for within-model spatial-attention interventions.

Shuffle repetitions are averaged within image before any patient resampling.
The same bootstrap patient counts are used for every model, seed and condition.
Intervals therefore condition on the supplied fitted models; training-seed
variation is described separately by the sample standard deviation.
"""
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_ORDER = ("lt", "yz", "zz")
MODEL_ORDER = ("eq_off", "eq_on")
INTERVENTIONS = ("original", "mean_constant", "spatial_shuffle", "ones_constant")
METRICS = {
    "original_accuracy": ("original_correct", "n"),
    "intervened_accuracy": ("intervened_correct", "n"),
    "accuracy_difference": ("accuracy_difference", "n"),
    "flip_rate": ("flipped", "n"),
    "correct_to_wrong_rate": ("correct_to_wrong", "original_correct"),
    "wrong_to_correct_rate": ("wrong_to_correct", "original_wrong"),
    "common_original_correct_to_wrong_rate": ("common_correct_to_wrong", "common_correct"),
    "true_probability_delta_mean": ("true_probability_delta", "n"),
    "original_pred_probability_drop_mean": ("original_pred_probability_drop", "n"),
    "probability_l1_change_mean": ("probability_l1_change", "n"),
    "attention_l1_change_mean": ("attention_l1_change", "n"),
}
DERIVED_COLUMNS = (
    "original_pred_id", "intervened_pred_id", "original_correct",
    "intervened_correct", "flipped", "correct_to_wrong", "wrong_to_correct",
    "true_probability_delta", "original_pred_probability_drop", "probability_l1_change",
)
PROBABILITY_ATOL = 1e-6


def _integer_column(frame, column):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} must contain finite integers")
    frame[column] = values.astype(np.int64)


def _validate_and_recompute(predictions):
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("Predictions must be a nonempty DataFrame")
    required = {
        "seed", "model", "case_id", "patient_id", "image_name", "true_label",
        "true_label_id", "intervention", "repeat", "attention_l1_change",
        *[f"original_prob_{label}" for label in CLASS_ORDER],
        *[f"prob_{label}" for label in CLASS_ORDER],
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    frame = predictions.copy().reset_index(drop=True)
    for column in ("case_id", "patient_id", "image_name", "true_label", "model", "intervention"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains missing or empty values")
        frame[column] = frame[column].astype(str)
    for column in ("seed", "true_label_id", "repeat"):
        _integer_column(frame, column)
    if (frame.seed < 0).any():
        raise ValueError("Training seeds must be nonnegative")
    labels = frame.true_label_id.to_numpy()
    if ((labels < 0) | (labels >= len(CLASS_ORDER))).any():
        raise ValueError("true_label_id must be 0, 1 or 2")
    if not np.array_equal(frame.true_label.to_numpy(), np.asarray(CLASS_ORDER)[labels]):
        raise ValueError("true_label and true_label_id disagree")
    if set(frame.model) != set(MODEL_ORDER):
        raise ValueError("Expected exactly eq_off and eq_on")
    if set(frame.intervention) != set(INTERVENTIONS):
        raise ValueError("All four intervention modes are required")
    keys = ["seed", "model", "case_id", "intervention", "repeat"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate seed/model/case/intervention/repeat")
    fixed = frame.intervention.ne("spatial_shuffle")
    if not frame.loc[fixed, "repeat"].eq(-1).all():
        raise ValueError("Nonshuffle interventions require repeat=-1")
    shuffle_repeats = sorted(frame.loc[~fixed, "repeat"].unique().tolist())
    if not shuffle_repeats or shuffle_repeats != list(range(len(shuffle_repeats))):
        raise ValueError("Shuffle repeats must be contiguous starting at zero")
    n_repeats = len(shuffle_repeats)
    expected_modes = {(mode, -1) for mode in INTERVENTIONS if mode != "spatial_shuffle"}
    expected_modes |= {("spatial_shuffle", repeat) for repeat in shuffle_repeats}
    for key, group in frame.groupby(["seed", "model", "case_id"], sort=False):
        if set(zip(group.intervention, group.repeat)) != expected_modes:
            raise ValueError(f"Incomplete intervention/repetition grid for {key}")
    identity = ["patient_id", "image_name", "true_label", "true_label_id"]
    if frame.groupby("case_id")[identity].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Case identities/labels differ across models, seeds or repetitions")
    expected_cases = set(frame.case_id)
    expected_seeds = set(frame.seed)
    grouped = frame.groupby(["seed", "model"], sort=False)
    if set(grouped.groups) != {(seed, model) for seed in expected_seeds for model in MODEL_ORDER}:
        raise ValueError("Each training seed must contain both EQ models")
    for key, group in grouped:
        if set(group.case_id) != expected_cases:
            raise ValueError(f"Unmatched cases across models or seeds: {key}")

    probabilities = {}
    for prefix in ("original_prob_", "prob_"):
        values = frame[[prefix + label for label in CLASS_ORDER]].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{prefix} values must be finite probabilities within [0, 1]")
        if not np.allclose(values.sum(1), 1, atol=PROBABILITY_ATOL, rtol=0):
            raise ValueError(f"{prefix} rows do not sum to one")
        probabilities[prefix] = values
    original, altered = probabilities["original_prob_"], probabilities["prob_"]
    original_columns = ["original_prob_" + label for label in CLASS_ORDER]
    if frame.groupby(["seed", "model", "case_id"])[original_columns].nunique().gt(1).any().any():
        raise ValueError("Original probabilities change across intervention rows")
    identity_rows = frame.intervention.eq("original").to_numpy()
    if not np.array_equal(original[identity_rows], altered[identity_rows]):
        raise ValueError("Original intervention must retain exact original probabilities")
    attention = frame.attention_l1_change.to_numpy(float)
    if not np.isfinite(attention).all() or ((attention < 0) | (attention > 1 + 1e-6)).any():
        raise ValueError("attention_l1_change must be finite and within [0, 1]")
    if not np.equal(attention[identity_rows], 0).all():
        raise ValueError("Original intervention must have zero attention change")
    frame["attention_l1_change"] = attention
    old_id, new_id = original.argmax(1), altered.argmax(1)
    old_correct, new_correct = old_id == labels, new_id == labels
    row_ids = np.arange(len(frame))
    recomputed = {
        "original_pred_id": old_id,
        "intervened_pred_id": new_id,
        "original_correct": old_correct.astype(int),
        "intervened_correct": new_correct.astype(int),
        "flipped": (old_id != new_id).astype(int),
        "correct_to_wrong": (old_correct & ~new_correct).astype(int),
        "wrong_to_correct": (~old_correct & new_correct).astype(int),
        "true_probability_delta": altered[row_ids, labels] - original[row_ids, labels],
        "original_pred_probability_drop": original[row_ids, old_id] - altered[row_ids, old_id],
        "probability_l1_change": np.abs(altered - original).sum(1),
    }
    for column, values in recomputed.items():
        if column in frame:
            supplied = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            tolerance = 1e-6 if column.endswith(("delta", "drop", "change")) else 0
            if not np.isfinite(supplied).all() or not np.allclose(
                supplied, values, atol=tolerance, rtol=0
            ):
                raise ValueError(f"Stored derived column disagrees with probabilities: {column}")
        frame[column] = values
    frame["original_wrong"] = 1 - frame.original_correct
    frame["accuracy_difference"] = frame.intervened_correct - frame.original_correct
    originals = frame[frame.intervention.eq("original")].pivot(
        index=["seed", "case_id"], columns="model", values="original_correct"
    )
    common = originals.eq(1).all(axis=1).astype(int)
    indices = pd.MultiIndex.from_frame(frame[["seed", "case_id"]])
    frame["common_correct"] = common.reindex(indices).to_numpy()
    frame["common_correct_to_wrong"] = frame.common_correct * (1 - frame.intervened_correct)
    frame["n"] = 1.
    return frame, n_repeats


def _ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator != 0)


def _bootstrap_patient_counts(n_patients, n_bootstrap, seed):
    """Single shared resampling schedule; kept separate for exact-oracle tests."""
    return np.random.default_rng(seed).multinomial(
        n_patients, np.full(n_patients, 1 / n_patients), size=n_bootstrap
    )


def _interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    lower, upper = np.quantile(finite, [.025, .975])
    return float(lower), float(upper), int(len(finite))


def summarize_interventions(predictions, n_bootstrap=10000, seed=20261020):
    """Return per-seed and equal-seed-average summaries with patient intervals.

    Rates use images as the observational unit and patients as resampling clusters.
    A bootstrap draw with a zero conditional denominator is undefined. Combined
    draws are used only if every seed is defined, never by silently omitting a seed.
    """
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    n_bootstrap = int(n_bootstrap)
    frame, shuffle_repeats = _validate_and_recompute(predictions)
    components = sorted({item for values in METRICS.values() for item in values})
    image_keys = ["seed", "model", "case_id", "patient_id", "intervention"]
    # Each image has one value per intervention, regardless of shuffle repeat count.
    images = frame.groupby(image_keys, sort=True)[components].mean().reset_index()
    patients = sorted(images.patient_id.unique())
    n_patients = len(patients)
    counts = (_bootstrap_patient_counts(n_patients, n_bootstrap, seed)
              if n_bootstrap and n_patients > 1 else np.empty((0, n_patients), dtype=int))
    seeds = sorted(images.seed.unique())
    bootstrap_metrics = {}
    rows = []
    for training_seed in seeds:
        for model in MODEL_ORDER:
            for intervention in INTERVENTIONS:
                selected = images[
                    images.seed.eq(training_seed) & images.model.eq(model)
                    & images.intervention.eq(intervention)
                ]
                arrays = selected.groupby("patient_id")[components].sum().reindex(patients).to_numpy()
                totals, sampled = arrays.sum(0), counts @ arrays
                indexes = {column: index for index, column in enumerate(components)}
                row = dict(seed=int(training_seed), model=model, intervention=intervention,
                           n_images=len(selected), n_patients=n_patients,
                           n_repeats=shuffle_repeats if intervention == "spatial_shuffle" else 1,
                           n_original_correct_images=int(selected.original_correct.sum()),
                           n_original_wrong_images=int(selected.original_wrong.sum()),
                           n_common_original_correct_images=int(selected.common_correct.sum()),
                           n_bootstrap_requested=n_bootstrap)
                draws = {}
                for metric, (numerator, denominator) in METRICS.items():
                    ni, di = indexes[numerator], indexes[denominator]
                    row[metric] = float(_ratio(totals[ni], totals[di]))
                    draws[metric] = _ratio(sampled[:, ni], sampled[:, di])
                    lo, hi, valid = _interval(draws[metric])
                    row[metric + "_ci_low"] = lo
                    row[metric + "_ci_high"] = hi
                    row[metric + "_ci_n_valid"] = valid
                rows.append(row)
                bootstrap_metrics[(training_seed, model, intervention)] = draws
    summary = pd.DataFrame(rows)
    combined_rows = []
    for model in MODEL_ORDER:
        for intervention in INTERVENTIONS:
            selected = summary[summary.model.eq(model) & summary.intervention.eq(intervention)]
            row = dict(model=model, intervention=intervention, n_seeds=len(seeds),
                       seeds=",".join(str(value) for value in seeds),
                       n_images=int(selected.n_images.iloc[0]), n_patients=n_patients,
                       n_repeats=int(selected.n_repeats.iloc[0]), n_bootstrap_requested=n_bootstrap)
            for metric in METRICS:
                values = selected[metric].to_numpy(float)
                row[metric] = float(values.mean())
                row[metric + "_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                row[metric + "_n_defined_seeds"] = int(np.isfinite(values).sum())
                draws = np.stack([
                    bootstrap_metrics[(training_seed, model, intervention)][metric]
                    for training_seed in seeds
                ]).mean(0)
                lo, hi, valid = _interval(draws)
                row[metric + "_ci_low"] = lo
                row[metric + "_ci_high"] = hi
                row[metric + "_ci_n_valid"] = valid
            combined_rows.append(row)
    return summary, pd.DataFrame(combined_rows)


def _number(value, percent=False, signed=False):
    if not np.isfinite(value):
        return "NA"
    return format(100 * value if percent else value,
                  ("+" if signed else "") + (".2f" if percent else ".5f"))


def _estimate(row, metric, percent=False, signed=False):
    value = _number(row[metric], percent, signed)
    low = _number(row[metric + "_ci_low"], percent, signed)
    high = _number(row[metric + "_ci_high"], percent, signed)
    return f"{value} [{low}, {high}]"


def report_markdown(summary, combined):
    """Explain what these interventions can and cannot establish."""
    lines = [
        "# 注意力空间干预诊断",
        "",
        "仅使用验证集原图；每项干预与同一权重、同一模型的原始注意力输出配对。",
        "预测差异由概率独立复算。空间打乱的各次重复先在每张图内平均，"
        "不把重复次数当成额外图片或患者。",
        "",
        "- original：原始注意力，作为恒等检查。",
        "- mean_constant：每张注意力图替换为其自身均值，保留平均幅度，去除空间位置差异。",
        "- spatial_shuffle：打乱原始注意力值的空间位置，保留数值分布。",
        "- ones_constant：注意力全部置为 1，同时改变空间差异和幅度；不能单独归因于空间位置。",
        "",
        "下表为训练种子等权均值；方括号是配对患者 bootstrap 95% 区间。"
        "所有种子、模型和干预使用相同的患者重采样权重。区间条件于本次已训练权重，"
        "不包含训练种子不确定性；CSV 另列各种子结果及种子间样本标准差。",
        "准确率、翻转率以 % 表示，准确率差以百分点表示；概率变化为干预后减原始，"
        "概率 L1 为三类绝对差之和。",
        "",
        "|模型|干预|原始准确率|干预准确率|准确率差及区间|翻转率及区间|真实类概率变化及区间|概率 L1 及区间|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in combined.iterrows():
        lines.append(
            f"|{row.model}|{row.intervention}|{_number(row.original_accuracy, True)}|"
            f"{_number(row.intervened_accuracy, True)}|"
            f"{_estimate(row, 'accuracy_difference', True, True)}|"
            f"{_estimate(row, 'flip_rate', True)}|"
            f"{_estimate(row, 'true_probability_delta_mean', signed=True)}|"
            f"{_estimate(row, 'probability_l1_change_mean')}|"
        )
    lines += [
        "",
        f"共 {int(combined.n_images.iloc[0])} 张图、{int(combined.n_patients.iloc[0])} 位患者、"
        f"{int(combined.n_seeds.iloc[0])} 个训练种子；每图空间打乱 "
        f"{int(combined.loc[combined.intervention.eq('spatial_shuffle'), 'n_repeats'].iloc[0])} 次。",
        "",
        "详细 CSV 同时包含：本模型原图正确队列的正确变错误率、原图错误队列的错误变正确率、"
        "同一种子两模型原图共同正确队列的变错率、原预测类别概率下降和注意力平均绝对变化。",
        "本模型正确队列与共同正确队列分母不同，不可混用；也不能直接相减两个模型的"
        "各自正确队列变错率，解释为 EQ 的因果交互效应。",
        "",
        "NA 表示对应分母为零、单种子无样本标准差、未进行 bootstrap 或只有一位患者。"
        "各指标的 ci_n_valid 列给出有效重采样次数；条件分母为零的抽样不参与区间。"
        "跨种子统计只使用所有种子都可定义的抽样，不会静默删掉不可定义的种子。",
        "",
        "若打乱或抹平后变化很小，只能说明这些权重在本批图片上对相应空间干预不敏感；"
        "若变化明显，只能说明输出使用了这些注意力值的空间配置。"
        "两者均不能单独证明注意力对应正确的解剖依据；后者仍需医生标注或独立核验。",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(predictions, output_dir, n_bootstrap=10000, seed=20261020):
    """Write three new report files, preserving all existing runner artifacts."""
    output_dir = Path(output_dir)
    destinations = [output_dir / filename for filename in (
        "intervention_summary.csv", "intervention_combined.csv", "summary.md"
    )]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite existing attention statistics outputs")
    summary, combined = summarize_interventions(predictions, n_bootstrap, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destinations[0], index=False)
    combined.to_csv(destinations[1], index=False)
    destinations[2].write_text(report_markdown(summary, combined), encoding="utf-8")
    return summary, combined
