"""Paired patient statistics for fixed-angle attention interventions.

Each prediction is classified before averaging: shuffle repetitions, then angles
within an image, then images. Patient resampling keeps every image, seed, angle,
model and intervention together. Confidence intervals condition on fitted models.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.attention_intervention_stats import (
    CLASS_ORDER,
    INTERVENTIONS,
    MODEL_ORDER,
    _bootstrap_patient_counts,
    _interval,
    _validate_and_recompute,
)


ANGLES = (-15., -10., -5., 0., 5., 10., 15.)
MODEL_METRICS = {
    "original_accuracy": "original_correct",
    "intervened_accuracy": "intervened_correct",
    "accuracy_difference": "accuracy_difference",
    "flip_rate": "flipped",
    "correct_to_wrong_fraction": "correct_to_wrong",
    "wrong_to_correct_fraction": "wrong_to_correct",
    "true_probability_delta_mean": "true_probability_delta",
    "original_pred_probability_drop_mean": "original_pred_probability_drop",
    "probability_l1_change_mean": "probability_l1_change",
    "attention_l1_change_mean": "attention_l1_change",
}
GAIN_METRICS = (
    "eq_gain",
    "eq_gain_change_vs_original",
    "rotation_minus_zero_gain",
)
OUTPUT_NAMES = (
    "model_summary.csv",
    "model_combined.csv",
    "eq_gain_summary.csv",
    "eq_gain_combined.csv",
    "summary.md",
)


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a nonnegative integer") from None
    if not np.isfinite(number) or number < 0 or number != np.floor(number):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


def validate_predictions(predictions):
    """Validate the complete grid and independently recompute prediction metrics.

    Original probabilities refer to original attention at the *same angle*.
    They may differ across angles. A case's identity and label may not.
    """
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("Predictions must be a nonempty DataFrame")
    if "angle_deg" not in predictions:
        raise ValueError("Missing prediction column: angle_deg")
    frame = predictions.copy().reset_index(drop=True)
    angles = pd.to_numeric(frame.angle_deg, errors="raise").to_numpy(float)
    if not np.isfinite(angles).all() or set(angles) != set(ANGLES):
        raise ValueError(f"Expected exactly the seven fixed angles {ANGLES}")
    frame["angle_deg"] = angles
    validated = []
    expected_grid = None
    expected_repeats = None
    for angle in ANGLES:
        part, repeats = _validate_and_recompute(frame[frame.angle_deg.eq(angle)])
        grid = set(zip(part.seed, part.model, part.case_id, part.intervention, part.repeat))
        if expected_grid is not None and grid != expected_grid:
            raise ValueError("Unmatched seed/model/case/intervention/repetition grid across angles")
        if expected_repeats is not None and repeats != expected_repeats:
            raise ValueError("Shuffle repeat counts differ across angles")
        expected_grid, expected_repeats = grid, repeats
        validated.append(part)
    result = pd.concat(validated, ignore_index=True)
    identity = ["patient_id", "image_name", "true_label", "true_label_id"]
    if result.groupby("case_id")[identity].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Case identities/labels differ across angles")
    return result, expected_repeats


def _scopes():
    # Zero is reported separately and never enters the primary six-angle mean.
    yield "nonzero_mean", None, len(ANGLES) - 1
    for angle in ANGLES:
        yield "angle", angle, 1


def _add_estimate(row, metric, value, draws):
    row[metric] = float(value)
    low, high, valid = _interval(draws)
    row[metric + "_ci_low"] = low
    row[metric + "_ci_high"] = high
    row[metric + "_ci_n_valid"] = valid


def _combined_estimate(row, metric, values, draws):
    values = np.asarray(values, dtype=float)
    _add_estimate(row, metric, values.mean(), np.asarray(draws).mean(axis=0))
    row[metric + "_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan


def summarize_rotation_interventions(predictions, n_bootstrap=10000, seed=20261020):
    """Return model and EQ contrast summaries, per seed and averaged over seeds.

    G(mode) = accuracy(eq_on, mode) - accuracy(eq_off, mode).
    D(mode) = G(mode) - G(original).
    R(mode) = G(mode, nonzero_mean) - G(mode, angle=0).

    R is only reported on the nonzero_mean rows. These are paired descriptive
    contrasts, not estimates of a causal mediation effect.
    """
    n_bootstrap = _nonnegative_integer(n_bootstrap, "n_bootstrap")
    seed = _nonnegative_integer(seed, "seed")
    frame, repeats = validate_predictions(predictions)
    components = list(dict.fromkeys(MODEL_METRICS.values()))
    image_keys = ["seed", "model", "case_id", "patient_id", "intervention"]
    by_angle = frame.groupby(image_keys + ["angle_deg"], sort=True)[components].mean().reset_index()
    nonzero = by_angle[by_angle.angle_deg.ne(0)].groupby(
        image_keys, sort=True
    )[components].mean().reset_index()
    patients = sorted(by_angle.patient_id.unique())
    seeds = sorted(int(value) for value in by_angle.seed.unique())
    n_patients = len(patients)
    n_images = by_angle.case_id.nunique()
    counts = (
        _bootstrap_patient_counts(n_patients, n_bootstrap, seed)
        if n_bootstrap and n_patients > 1 else np.empty((0, n_patients), dtype=int)
    )
    # Values and bootstrap draws share the exact same patient schedule.
    estimates = {}
    model_rows = []
    for scope, angle, n_angles in _scopes():
        selected_scope = nonzero if angle is None else by_angle[by_angle.angle_deg.eq(angle)]
        for training_seed in seeds:
            for model in MODEL_ORDER:
                for intervention in INTERVENTIONS:
                    selected = selected_scope[
                        selected_scope.seed.eq(training_seed)
                        & selected_scope.model.eq(model)
                        & selected_scope.intervention.eq(intervention)
                    ].copy()
                    selected["n"] = 1.
                    arrays = selected.groupby("patient_id")[components + ["n"]].sum().reindex(
                        patients
                    ).to_numpy(float)
                    total, samples = arrays.sum(axis=0), counts @ arrays
                    values = total[:-1] / total[-1]
                    draws = samples[:, :-1] / samples[:, -1:]
                    row = dict(
                        seed=training_seed, model=model, intervention=intervention,
                        scope=scope, angle_deg=np.nan if angle is None else angle,
                        n_angles=n_angles, n_images=n_images, n_patients=n_patients,
                        n_repeats=repeats if intervention == "spatial_shuffle" else 1,
                        n_bootstrap_requested=n_bootstrap,
                    )
                    node = {}
                    for metric, component in MODEL_METRICS.items():
                        index = components.index(component)
                        node[metric] = (values[index], draws[:, index])
                        _add_estimate(row, metric, *node[metric])
                    estimates[(training_seed, model, scope, angle, intervention)] = node
                    model_rows.append(row)

    model_combined_rows = []
    for scope, angle, n_angles in _scopes():
        for model in MODEL_ORDER:
            for intervention in INTERVENTIONS:
                row = _combined_metadata(
                    seeds, intervention, scope, angle, n_angles, n_images,
                    n_patients, repeats, n_bootstrap, model=model,
                )
                for metric in MODEL_METRICS:
                    pairs = [
                        estimates[(training_seed, model, scope, angle, intervention)][metric]
                        for training_seed in seeds
                    ]
                    _combined_estimate(row, metric, [item[0] for item in pairs],
                                       [item[1] for item in pairs])
                model_combined_rows.append(row)

    gains = {}
    for scope, angle, _ in _scopes():
        for training_seed in seeds:
            for intervention in INTERVENTIONS:
                off = estimates[(training_seed, "eq_off", scope, angle, intervention)][
                    "intervened_accuracy"
                ]
                on = estimates[(training_seed, "eq_on", scope, angle, intervention)][
                    "intervened_accuracy"
                ]
                gains[(training_seed, scope, angle, intervention)] = (
                    on[0] - off[0], on[1] - off[1]
                )

    contrasts = {}
    gain_rows = []
    for scope, angle, n_angles in _scopes():
        for training_seed in seeds:
            original = gains[(training_seed, scope, angle, "original")]
            for intervention in INTERVENTIONS:
                gain = gains[(training_seed, scope, angle, intervention)]
                row = dict(
                    seed=training_seed, intervention=intervention, scope=scope,
                    angle_deg=np.nan if angle is None else angle,
                    n_angles=n_angles, n_images=n_images, n_patients=n_patients,
                    n_repeats=repeats if intervention == "spatial_shuffle" else 1,
                    n_bootstrap_requested=n_bootstrap,
                )
                node = {
                    "eq_gain": gain,
                    "eq_gain_change_vs_original": (
                        gain[0] - original[0], gain[1] - original[1]
                    ),
                }
                if scope == "nonzero_mean":
                    zero = gains[(training_seed, "angle", 0., intervention)]
                    node["rotation_minus_zero_gain"] = (
                        gain[0] - zero[0], gain[1] - zero[1]
                    )
                else:
                    node["rotation_minus_zero_gain"] = (np.nan, np.full(len(counts), np.nan))
                for metric in GAIN_METRICS:
                    _add_estimate(row, metric, *node[metric])
                contrasts[(training_seed, scope, angle, intervention)] = node
                gain_rows.append(row)

    gain_combined_rows = []
    for scope, angle, n_angles in _scopes():
        for intervention in INTERVENTIONS:
            row = _combined_metadata(
                seeds, intervention, scope, angle, n_angles, n_images,
                n_patients, repeats, n_bootstrap,
            )
            for metric in GAIN_METRICS:
                pairs = [contrasts[(training_seed, scope, angle, intervention)][metric]
                         for training_seed in seeds]
                _combined_estimate(row, metric, [item[0] for item in pairs],
                                   [item[1] for item in pairs])
            gain_combined_rows.append(row)
    return (
        pd.DataFrame(model_rows), pd.DataFrame(model_combined_rows),
        pd.DataFrame(gain_rows), pd.DataFrame(gain_combined_rows),
    )


def _combined_metadata(seeds, intervention, scope, angle, n_angles, n_images,
                       n_patients, repeats, n_bootstrap, **extra):
    return dict(
        **extra, intervention=intervention, scope=scope,
        angle_deg=np.nan if angle is None else angle, n_seeds=len(seeds),
        seeds=",".join(str(value) for value in seeds), n_angles=n_angles,
        n_images=n_images, n_patients=n_patients,
        n_repeats=repeats if intervention == "spatial_shuffle" else 1,
        n_bootstrap_requested=n_bootstrap,
    )


def _number(value, percent=False, signed=False):
    if not np.isfinite(value):
        return "NA"
    return format(100 * value if percent else value,
                  ("+" if signed else "") + (".3f" if percent else ".5f"))


def _estimate(row, metric, percent=False, signed=False):
    return (
        f"{_number(row[metric], percent, signed)} "
        f"[{_number(row[metric + '_ci_low'], percent, signed)}, "
        f"{_number(row[metric + '_ci_high'], percent, signed)}]"
    )


def report_markdown(model_summary, model_combined, gain_summary, gain_combined):
    """Report the predeclared contrasts without automatic positive/negative verdicts."""
    first = gain_combined.iloc[0]
    lines = [
        "# 旋转下的注意力空间干预诊断", "",
        "使用同一批验证图片与已有权重，在固定角度 −15、−10、−5、0、5、10、15° 下，"
        "分别保留原始注意力、替换为该图均值、打乱空间位置、全部置为 1。",
        "每项干预与同一模型、同一图片、同一角度的原始注意力输出配对；original_prob_* "
        "不是干净图概率。不会将旋转、打乱重复或训练种子当作新增病例。", "",
        f"共 {int(first.n_images)} 张图、{int(first.n_patients)} 位患者、"
        f"{int(first.n_seeds)} 个训练种子。主结果先在每张图内平均打乱重复，再等权平均"
        "六个非零角度，再计算图像加权准确率，最后对训练种子等权平均。0° 不纳入主结果。", "",
        "方括号为共享患者重采样的 bootstrap 95% 区间，所有角度、干预、模型和种子使用"
        "同一组患者权重。差值区间直接在每次配对重采样内计算，未相减区间端点。"
        "区间条件于这些已训练权重，不包含重新训练的不确定性；CSV 另给各种子结果和"
        "种子间样本标准差。", "",
        "## 六个非零角度的主结果", "",
        "G = EQ 开启准确率 − EQ 关闭准确率；D = 当前干预 G − 原始注意力 G。"
        "准确率以 % 表示，所有准确率差以百分点表示。", "",
        "|干预|EQ 关闭准确率|EQ 开启准确率|G 及区间|D 及区间|",
        "|---|---:|---:|---:|---:|",
    ]
    primary = gain_combined[gain_combined.scope.eq("nonzero_mean")]
    for _, row in primary.iterrows():
        selected = model_combined[
            model_combined.scope.eq("nonzero_mean")
            & model_combined.intervention.eq(row.intervention)
        ].set_index("model")
        lines.append(
            f"|{row.intervention}|{_number(selected.loc['eq_off', 'intervened_accuracy'], True)}|"
            f"{_number(selected.loc['eq_on', 'intervened_accuracy'], True)}|"
            f"{_estimate(row, 'eq_gain', True, True)}|"
            f"{_estimate(row, 'eq_gain_change_vs_original', True, True)}|"
        )
    lines += [
        "", "## 两个模型各自对干预的变化", "",
        "下列准确率差为干预后减同角度原始注意力。翻转是预测类别改变，分母为所有图片；"
        "概率变化为真实类别概率的有符号变化，L1 为三类概率绝对差之和。", "",
        "|模型|干预|准确率差及区间|翻转率及区间|真实类概率变化及区间|概率 L1 及区间|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in model_combined[model_combined.scope.eq("nonzero_mean")].iterrows():
        lines.append(
            f"|{row.model}|{row.intervention}|{_estimate(row, 'accuracy_difference', True, True)}|"
            f"{_estimate(row, 'flip_rate', True)}|"
            f"{_estimate(row, 'true_probability_delta_mean', signed=True)}|"
            f"{_estimate(row, 'probability_l1_change_mean')}|"
        )
    lines += [
        "", "## 0° 和相对 0° 的辅助对照", "",
        "R = 非零角度的 G − 0° 的 G，描述相对干净图差异；它不是额外训练效果或因果机制证明。", "",
        "|干预|0° EQ 关闭准确率|0° EQ 开启准确率|0° G 及区间|R 及区间|",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        selected = model_combined[
            model_combined.scope.eq("angle") & model_combined.angle_deg.eq(0)
            & model_combined.intervention.eq(row.intervention)
        ].set_index("model")
        zero = gain_combined[
            gain_combined.scope.eq("angle") & gain_combined.angle_deg.eq(0)
            & gain_combined.intervention.eq(row.intervention)
        ].iloc[0]
        lines.append(
            f"|{row.intervention}|{_number(selected.loc['eq_off', 'intervened_accuracy'], True)}|"
            f"{_number(selected.loc['eq_on', 'intervened_accuracy'], True)}|"
            f"{_estimate(zero, 'eq_gain', True, True)}|"
            f"{_estimate(row, 'rotation_minus_zero_gain', True, True)}|"
        )
    lines += [
        "", "## 解读边界", "",
        "- 若 G 基本保留，也要检查两模型各自准确率；两者可能同时受到相近程度的损害，"
        "使差值不变。G 保留不能直接推出空间注意力没有作用。",
        "- D 的区间包含 0 不能证明等价或无影响；本试验未预设等价界值。"
        "不计算收益保留百分比，以免小分母放大波动。",
        "- mean_constant 保留单张图注意力均值并去除空间差异；spatial_shuffle 保留数值分布"
        "并打乱位置。ones_constant 同时改变平均幅度，是辅助对照。",
        "- 推理期替换不能排除注意力或 EQ 在训练阶段对骨干特征、分类头产生的影响，"
        "也不能单独证明注意力对应正确解剖结构。",
        "- 各角度结果为辅助分析，均见 CSV。correct_to_wrong_fraction 与 "
        "wrong_to_correct_fraction 的分母是所有图片/角度，均不是只在原始正确或错误队列中的条件比例。",
        "- NA 表示不适用的 R（逐角度行）、单种子没有种子间标准差、未进行 bootstrap "
        "或只有一位患者；ci_n_valid 列给出有效重采样次数。",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(predictions, output_dir, n_bootstrap=10000, seed=20261020):
    """Write four CSV summaries and a report; do not overwrite existing outputs."""
    output_dir = Path(output_dir)
    destinations = [output_dir / name for name in OUTPUT_NAMES]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite existing rotation attention statistics outputs")
    tables = summarize_rotation_interventions(predictions, n_bootstrap, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for table, path in zip(tables, destinations[:4]):
        table.to_csv(path, index=False)
    destinations[-1].write_text(report_markdown(*tables), encoding="utf-8")
    return tables
