"""Independent correctness and patient-cluster oracles; no torch required."""
import numpy as np
import pandas as pd
import pytest

from scripts import rotation_attention_stats as stats


CORRECT = np.array([.8, .1, .1])
WRONG = np.array([.1, .8, .1])


def predictions(cases=(("a", "patient_1"), ("b", "patient_1"), ("c", "patient_2")),
                seeds=(42,), repeats=2, original_fn=None, intervention_fn=None):
    rows = []
    for seed in seeds:
        for model in stats.MODEL_ORDER:
            for case, patient in cases:
                for angle in stats.ANGLES:
                    original = (CORRECT if original_fn is None
                                else original_fn(seed, model, case, angle))
                    for intervention in stats.INTERVENTIONS:
                        for repeat in range(repeats) if intervention == "spatial_shuffle" else (-1,):
                            altered = original.copy()
                            if intervention != "original" and intervention_fn is not None:
                                altered = intervention_fn(seed, model, case, angle, intervention, repeat)
                            row = dict(
                                seed=seed, model=model, case_id=case, patient_id=patient,
                                image_name=case + ".png", true_label="lt", true_label_id=0,
                                intervention=intervention, repeat=repeat, angle_deg=angle,
                                attention_l1_change=0. if intervention == "original" else .2,
                            )
                            for index, label in enumerate(stats.CLASS_ORDER):
                                row["original_prob_" + label] = float(original[index])
                                row["prob_" + label] = float(altered[index])
                            rows.append(row)
    return pd.DataFrame(rows)


def select(frame, intervention="mean_constant", scope="nonzero_mean", angle=None, **conditions):
    selected = frame[frame.intervention.eq(intervention) & frame.scope.eq(scope)]
    if angle is not None:
        selected = selected[selected.angle_deg.eq(angle)]
    for name, value in conditions.items():
        selected = selected[selected[name].eq(value)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_difference_in_differences_matches_complete_patient_enumeration(monkeypatch):
    def original(seed, model, case, angle):
        if angle == 0:
            return CORRECT
        return CORRECT if case == "a" or (model == "eq_on" and case == "b") else WRONG

    def intervention(seed, model, case, angle, mode, repeat):
        if angle == 0:
            return CORRECT
        return CORRECT if case == "a" or (
            (model == "eq_off" and case == "b") or (model == "eq_on" and case == "c")
        ) else WRONG

    # Two patients sampled twice: four equally likely ordered bootstrap samples.
    counts = np.array([[2, 0], [1, 1], [1, 1], [0, 2]])
    monkeypatch.setattr(stats, "_bootstrap_patient_counts", lambda *args: counts)
    models, combined_models, gains, combined_gains = stats.summarize_rotation_interventions(
        predictions(original_fn=original, intervention_fn=intervention), n_bootstrap=4
    )
    for table in (gains, combined_gains):
        before = select(table, intervention="original")
        after = select(table)
        assert before.eq_gain == pytest.approx(1 / 3)
        assert after.eq_gain == 0
        assert after.eq_gain_change_vs_original == pytest.approx(-1 / 3)
        assert after.rotation_minus_zero_gain == 0
        # Patient 1: original gain 1/2, altered gain -1/2 => D=-1.
        # Patient 2: original gain 0, altered gain 1 => D=1.
        # Image weighting (2 versus 1 image) makes the middle draws D=-1/3.
        lo, hi = np.quantile([-1, -1 / 3, -1 / 3, 1], [.025, .975])
        assert after.eq_gain_change_vs_original_ci_low == pytest.approx(lo)
        assert after.eq_gain_change_vs_original_ci_high == pytest.approx(hi)
        assert after.eq_gain_change_vs_original_ci_n_valid == 4
        assert after.n_images == 3 and after.n_patients == 2 and after.n_angles == 6
        # The contrast interval is independently derived from paired sample gains.
        lo, hi = np.quantile([-.5, 0, 0, 1], [.025, .975])
        assert after.eq_gain_ci_low == pytest.approx(lo)
        assert after.eq_gain_ci_high == pytest.approx(hi)
    for table in (models, combined_models):
        off = select(table, model="eq_off")
        on = select(table, model="eq_on")
        assert off.original_accuracy == pytest.approx(1 / 3)
        assert off.intervened_accuracy == pytest.approx(2 / 3)
        assert off.accuracy_difference == pytest.approx(1 / 3)
        assert on.accuracy_difference == 0
        assert on.flip_rate == pytest.approx(2 / 3)
        assert on.correct_to_wrong_fraction == pytest.approx(1 / 3)
        assert on.wrong_to_correct_fraction == pytest.approx(1 / 3)
    assert np.isnan(select(combined_gains).eq_gain_seed_sd)


def test_zero_is_excluded_and_secondary_is_directly_paired():
    frame = predictions(
        original_fn=lambda seed, model, case, angle:
        CORRECT if model == "eq_on" or angle == 0 else WRONG
    )
    _, _, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=100)
    primary = select(gains, intervention="original")
    zero = select(gains, intervention="original", scope="angle", angle=0)
    assert primary.eq_gain == primary.rotation_minus_zero_gain == 1
    assert primary.rotation_minus_zero_gain_ci_low == primary.rotation_minus_zero_gain_ci_high == 1
    assert zero.eq_gain == 0
    assert np.isnan(zero.rotation_minus_zero_gain)
    assert zero.rotation_minus_zero_gain_ci_n_valid == 0


def test_shared_patient_schedule_across_seeds_does_not_create_artificial_uncertainty():
    frame = predictions(
        cases=(("a", "patient_1"), ("b", "patient_2")), seeds=(42, 2026),
        intervention_fn=lambda seed, model, case, angle, mode, repeat:
        WRONG if model == "eq_on" and (
            (seed == 42 and case == "a") or (seed == 2026 and case == "b")
        ) else CORRECT,
    )
    _, _, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=1000)
    result = select(gains)
    assert result.eq_gain == result.eq_gain_change_vs_original == -.5
    assert result.eq_gain_ci_low == result.eq_gain_ci_high == -.5
    assert result.eq_gain_change_vs_original_ci_low == result.eq_gain_change_vs_original_ci_high == -.5
    assert result.n_images == 2 and result.n_patients == 2 and result.n_seeds == 2
    assert result.eq_gain_seed_sd == 0


def test_direct_gain_change_interval_preserves_cross_intervention_covariance():
    frame = predictions(
        cases=(("a", "patient_1"), ("b", "patient_2")),
        original_fn=lambda seed, model, case, angle:
        CORRECT if model == "eq_on" and case == "a" else WRONG,
        intervention_fn=lambda seed, model, case, angle, mode, repeat:
        CORRECT if model == "eq_off" or case == "a" else WRONG,
    )
    _, _, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=1000)
    result = select(gains)
    assert result.eq_gain == -.5
    assert result.eq_gain_change_vs_original == -1
    assert result.eq_gain_change_vs_original_ci_low == -1
    assert result.eq_gain_change_vs_original_ci_high == -1
    assert result.eq_gain_ci_high > result.eq_gain_ci_low


def test_shared_patient_schedule_across_angles_for_secondary_contrast():
    frame = predictions(
        cases=(("a", "patient_1"), ("b", "patient_2")),
        original_fn=lambda seed, model, case, angle:
        CORRECT if (model == "eq_on" and case == "a")
        or (model == "eq_off" and angle == 0) else WRONG,
    )
    _, _, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=1000)
    result = select(gains, intervention="original")
    # Nonzero G is [1, 0] and zero G is [0, -1] by patient, so R=1 every draw.
    assert result.rotation_minus_zero_gain == 1
    assert result.rotation_minus_zero_gain_ci_low == result.rotation_minus_zero_gain_ci_high == 1
    assert result.eq_gain_ci_high > result.eq_gain_ci_low


def test_equal_damage_to_both_models_is_visible_even_when_eq_gain_is_unchanged():
    frame = predictions(
        intervention_fn=lambda seed, model, case, angle, mode, repeat:
        WRONG if case == "a" else CORRECT
    )
    _, models, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=30)
    result = select(gains)
    assert result.eq_gain == result.eq_gain_change_vs_original == 0
    for model in stats.MODEL_ORDER:
        changed = select(models, model=model)
        assert changed.accuracy_difference == pytest.approx(-1 / 3)
        assert changed.flip_rate == pytest.approx(1 / 3)
        assert changed.true_probability_delta_mean == pytest.approx(-.7 / 3)
        assert changed.probability_l1_change_mean == pytest.approx(1.4 / 3)


def test_repeating_shuffle_draws_does_not_inflate_sample_or_change_interval():
    def change(seed, model, case, angle, mode, repeat):
        return WRONG if model == "eq_on" and mode == "spatial_shuffle" and repeat % 2 == 0 else CORRECT

    two = stats.summarize_rotation_interventions(predictions(repeats=2, intervention_fn=change),
                                                n_bootstrap=100)
    four = stats.summarize_rotation_interventions(predictions(repeats=4, intervention_fn=change),
                                                 n_bootstrap=100)
    for left, right in zip(two, four):
        columns = [column for column in left if column != "n_repeats"]
        pd.testing.assert_frame_equal(left[columns], right[columns])
    shuffled = select(two[1], intervention="spatial_shuffle", model="eq_on")
    assert shuffled.intervened_accuracy == shuffled.flip_rate == .5
    assert shuffled.n_images == 3 and shuffled.n_patients == 2


def test_angles_are_averaged_within_image_and_classification_precedes_averaging():
    frame = predictions(
        original_fn=lambda seed, model, case, angle:
        CORRECT if model == "eq_on" or angle == -15 else WRONG
    )
    _, models, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=0)
    result = select(models, intervention="original", model="eq_off")
    assert result.intervened_accuracy == pytest.approx(1 / 6)
    assert select(gains, intervention="original").eq_gain == pytest.approx(5 / 6)
    assert select(gains, intervention="original", scope="angle", angle=-15).eq_gain == 0
    assert result.intervened_accuracy_ci_n_valid == 0


def test_seed_sd_uses_sample_standard_deviation():
    frame = predictions(
        seeds=(42, 2026, 3407),
        original_fn=lambda seed, model, case, angle:
        CORRECT if model == "eq_off" or seed == 42 or (seed == 2026 and case == "a") else WRONG
    )
    _, _, summary, combined = stats.summarize_rotation_interventions(frame, n_bootstrap=0)
    assert select(combined, intervention="original").eq_gain_seed_sd == pytest.approx(
        np.std([0, -2 / 3, -1], ddof=1)
    )
    assert set(summary.seed) == {42, 2026, 3407}


def test_row_order_does_not_affect_results():
    frame = predictions(
        seeds=(42, 2026),
        intervention_fn=lambda seed, model, case, angle, mode, repeat:
        WRONG if case == "a" and angle < 0 else CORRECT,
    )
    original = stats.summarize_rotation_interventions(frame, n_bootstrap=53, seed=8)
    reordered = stats.summarize_rotation_interventions(
        frame.sample(frac=1, random_state=61), n_bootstrap=53, seed=8
    )
    for left, right in zip(original, reordered):
        pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("mutation", [
    lambda f: f[f.angle_deg.ne(15)],
    lambda f: f[~(f.angle_deg.eq(15) & f.case_id.eq("a"))],
    lambda f: f[~(f.angle_deg.eq(15) & f.seed.eq(2026))],
    lambda f: f[~(f.angle_deg.eq(15) & f.repeat.eq(1))],
    lambda f: pd.concat([f, f.iloc[:1]], ignore_index=True),
    lambda f: f.assign(angle_deg=f.angle_deg.replace({15.: 20.})),
    lambda f: f.assign(angle_deg=f.angle_deg.replace({15.: np.nan})),
])
def test_rejects_incomplete_or_invalid_angle_grid(mutation):
    with pytest.raises(ValueError):
        stats.summarize_rotation_interventions(mutation(predictions(seeds=(42, 2026))),
                                               n_bootstrap=0)


@pytest.mark.parametrize("column,value", [
    ("patient_id", "different_patient"),
    ("image_name", "different_image.png"),
])
def test_rejects_cross_angle_identity_changes(column, value):
    frame = predictions()
    frame.loc[frame.angle_deg.eq(15) & frame.case_id.eq("a"), column] = value
    with pytest.raises(ValueError, match="across angles"):
        stats.summarize_rotation_interventions(frame, n_bootstrap=0)


def test_rejects_cross_angle_label_changes():
    frame = predictions()
    changed = frame.angle_deg.eq(15) & frame.case_id.eq("a")
    frame.loc[changed, ["true_label", "true_label_id"]] = ["yz", 1]
    with pytest.raises(ValueError, match="across angles"):
        stats.summarize_rotation_interventions(frame, n_bootstrap=0)


def test_recomputes_derived_columns_and_rejects_corrupt_stored_values():
    frame = predictions()
    frame["flipped"] = 0
    frame.loc[0, "flipped"] = 1
    with pytest.raises(ValueError, match="Stored derived column"):
        stats.summarize_rotation_interventions(frame, n_bootstrap=0)


@pytest.mark.parametrize("argument,value", [
    ("n_bootstrap", -1), ("n_bootstrap", .5), ("n_bootstrap", True),
    ("seed", -1), ("seed", np.nan), ("seed", False),
])
def test_rejects_invalid_statistical_parameters(argument, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        stats.summarize_rotation_interventions(predictions(), **{argument: value})


def test_single_patient_suppresses_bootstrap_intervals():
    frame = predictions(cases=(("a", "patient_1"), ("b", "patient_1")))
    _, _, _, gains = stats.summarize_rotation_interventions(frame, n_bootstrap=100)
    assert gains.eq_gain.eq(0).all()
    assert gains.eq_gain_ci_low.isna().all()
    assert gains.eq_gain_ci_n_valid.eq(0).all()


def test_writes_four_csvs_and_explanatory_report_and_refuses_overwrite(tmp_path):
    tables = stats.write_outputs(predictions(), tmp_path, n_bootstrap=0)
    assert len(tables) == 4
    assert {path.name for path in tmp_path.iterdir()} == set(stats.OUTPUT_NAMES)
    report = (tmp_path / "summary.md").read_text()
    assert "D = 当前干预 G − 原始注意力 G" in report
    assert "两者可能同时" in report
    assert "区间包含 0 不能证明等价" in report
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        stats.write_outputs(predictions(), tmp_path, n_bootstrap=0)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
