"""Known-answer and patient-cluster oracles without torch or image dependencies."""
import numpy as np
import pandas as pd
import pytest

from scripts import attention_intervention_stats as stats


CORRECT = np.array([.8, .1, .1])
WRONG = np.array([.1, .8, .1])


def predictions(cases=(("a", "patient_1"), ("b", "patient_1"), ("c", "patient_2")),
                seeds=(42,), repeats=2, original_fn=None, intervention_fn=None):
    rows = []
    for seed in seeds:
        for model in stats.MODEL_ORDER:
            for case, patient in cases:
                original = CORRECT if original_fn is None else original_fn(seed, model, case)
                for intervention in stats.INTERVENTIONS:
                    for repeat in range(repeats) if intervention == "spatial_shuffle" else (-1,):
                        altered = original.copy()
                        if intervention != "original" and intervention_fn is not None:
                            altered = intervention_fn(seed, model, case, intervention, repeat)
                        row = dict(seed=seed, model=model, case_id=case, patient_id=patient,
                                   image_name=case + ".png", true_label="lt", true_label_id=0,
                                   intervention=intervention, repeat=repeat,
                                   attention_l1_change=0. if intervention == "original" else .2)
                        for label_id, label in enumerate(stats.CLASS_ORDER):
                            row["original_prob_" + label] = float(original[label_id])
                            row["prob_" + label] = float(altered[label_id])
                        rows.append(row)
    return pd.DataFrame(rows)


def row_for(frame, model="eq_off", intervention="mean_constant", seed=None):
    selected = frame[frame.model.eq(model) & frame.intervention.eq(intervention)]
    if seed is not None:
        selected = selected[selected.seed.eq(seed)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_image_weighted_cluster_bootstrap_matches_complete_patient_enumeration(monkeypatch):
    frame = predictions(intervention_fn=lambda seed, model, case, mode, repeat:
                        WRONG if case == "a" else CORRECT)
    # Two patients sampled twice have four equally likely ordered outcomes.
    # patient_1 contains two images (one flip); patient_2 one image (no flip).
    counts = np.array([[2, 0], [1, 1], [1, 1], [0, 2]])
    monkeypatch.setattr(stats, "_bootstrap_patient_counts", lambda *args: counts)
    summary, combined = stats.summarize_interventions(frame, n_bootstrap=4)
    result = row_for(summary)
    assert result.n_images == 3 and result.n_patients == 2
    assert result.original_accuracy == 1
    assert result.intervened_accuracy == pytest.approx(2 / 3)
    assert result.flip_rate == pytest.approx(1 / 3)  # Not the patient mean 1/4.
    assert result.correct_to_wrong_rate == pytest.approx(1 / 3)
    low, high = np.quantile([.5, 1 / 3, 1 / 3, 0], [.025, .975])
    assert result.flip_rate_ci_low == pytest.approx(low)
    assert result.flip_rate_ci_high == pytest.approx(high)
    assert result.flip_rate_ci_n_valid == 4
    assert result.true_probability_delta_mean == pytest.approx(-.7 / 3)
    assert result.original_pred_probability_drop_mean == pytest.approx(.7 / 3)
    assert result.probability_l1_change_mean == pytest.approx(1.4 / 3)
    assert np.isnan(row_for(combined).flip_rate_seed_sd)


def test_shuffle_repeated_identical_patterns_do_not_inflate_sample_or_change_interval():
    change = lambda seed, model, case, mode, repeat: (
        WRONG if mode == "spatial_shuffle" and repeat % 2 == 0 else CORRECT
    )
    two, combined_two = stats.summarize_interventions(
        predictions(repeats=2, intervention_fn=change), n_bootstrap=173
    )
    four, combined_four = stats.summarize_interventions(
        predictions(repeats=4, intervention_fn=change), n_bootstrap=173
    )
    for left, right in ((two, four), (combined_two, combined_four)):
        columns = [column for column in left if column != "n_repeats"]
        pd.testing.assert_frame_equal(left[columns], right[columns])
        result = row_for(left, intervention="spatial_shuffle")
        # Classification is computed per repeat before averaging, not from
        # averaged class probabilities, which could turn this into a tie.
        assert result.flip_rate == .5
        assert result.intervened_accuracy == .5
        assert result.n_images == 3


def test_cross_seed_bootstrap_uses_same_patients_and_equal_seed_rates():
    frame = predictions(
        cases=(("a", "patient_1"), ("b", "patient_2")), seeds=(42, 2026),
        intervention_fn=lambda seed, model, case, mode, repeat:
        WRONG if (seed == 42 and case == "a") or (seed == 2026 and case == "b") else CORRECT
    )
    _, combined = stats.summarize_interventions(frame, n_bootstrap=1000)
    result = row_for(combined)
    # For every paired patient draw, the two seeds' flip rates sum to one.
    # Independently resampling seeds would incorrectly widen this interval.
    assert result.flip_rate == .5
    assert result.flip_rate_ci_low == .5
    assert result.flip_rate_ci_high == .5
    assert result.accuracy_difference_ci_low == -.5
    assert result.accuracy_difference_ci_high == -.5
    assert result.n_images == 2 and result.n_patients == 2 and result.n_seeds == 2


def test_own_and_common_original_correct_cohorts_have_distinct_denominators():
    frame = predictions(
        original_fn=lambda seed, model, case:
        WRONG if (model == "eq_off" and case == "c") or (model == "eq_on" and case == "a")
        else CORRECT,
        intervention_fn=lambda seed, model, case, mode, repeat:
        CORRECT if case == "b" else WRONG,
    )
    summary, _ = stats.summarize_interventions(frame, n_bootstrap=30)
    result = row_for(summary)
    assert result.n_original_correct_images == 2
    assert result.n_common_original_correct_images == 1
    assert result.correct_to_wrong_rate == .5
    assert result.common_original_correct_to_wrong_rate == 0
    assert result.flip_rate == pytest.approx(1 / 3)


def test_zero_conditional_denominator_is_na_not_zero_or_dropped_seed():
    frame = predictions(
        seeds=(42, 2026),
        original_fn=lambda seed, model, case: WRONG if seed == 42 else CORRECT,
        intervention_fn=lambda *args: CORRECT,
    )
    summary, combined = stats.summarize_interventions(frame, n_bootstrap=100)
    first = row_for(summary, seed=42)
    assert np.isnan(first.correct_to_wrong_rate)
    assert np.isnan(first.correct_to_wrong_rate_ci_low)
    assert first.correct_to_wrong_rate_ci_n_valid == 0
    assert first.wrong_to_correct_rate == 1
    result = row_for(combined)
    assert result.correct_to_wrong_rate_n_defined_seeds == 1
    assert np.isnan(result.correct_to_wrong_rate)
    assert np.isnan(result.correct_to_wrong_rate_seed_sd)
    assert result.correct_to_wrong_rate_ci_n_valid == 0


def test_equal_seed_mean_of_conditional_rates_is_not_pooled_image_rate():
    frame = predictions(
        seeds=(42, 2026),
        original_fn=lambda seed, model, case:
        WRONG if seed == 42 and case != "a" else CORRECT,
        intervention_fn=lambda seed, model, case, mode, repeat:
        WRONG if case == "a" else CORRECT,
    )
    _, combined = stats.summarize_interventions(frame, n_bootstrap=100)
    result = row_for(combined)
    # Seed 42: 1/1. Seed 2026: 1/3. Mean 2/3, pooled count would be 2/4.
    assert result.correct_to_wrong_rate == pytest.approx(2 / 3)
    assert result.correct_to_wrong_rate_seed_sd == pytest.approx(np.std([1., 1 / 3], ddof=1))


def test_undefined_bootstrap_draws_are_counted_explicitly(monkeypatch):
    frame = predictions(
        cases=(("a", "patient_1"), ("b", "patient_2")),
        original_fn=lambda seed, model, case: CORRECT if case == "a" else WRONG,
        intervention_fn=lambda *args: WRONG,
    )
    monkeypatch.setattr(stats, "_bootstrap_patient_counts",
                        lambda *args: np.array([[2, 0], [1, 1], [1, 1], [0, 2]]))
    summary, combined = stats.summarize_interventions(frame, n_bootstrap=4)
    for data in (summary, combined):
        result = row_for(data)
        assert result.correct_to_wrong_rate_ci_n_valid == 3
        assert result.correct_to_wrong_rate_ci_low == result.correct_to_wrong_rate_ci_high == 1
        assert result.flip_rate_ci_n_valid == 4


@pytest.mark.parametrize("column,value", [
    ("prob_lt", np.nan),
    ("prob_lt", -.1),
    ("prob_lt", .2),  # Sum is no longer one.
    ("true_label_id", 0.5),
    ("true_label_id", 4),
    ("true_label", "zz"),
    ("patient_id", ""),
    ("patient_id", None),
    ("attention_l1_change", -.01),
    ("attention_l1_change", np.inf),
])
def test_rejects_invalid_values(column, value):
    frame = predictions()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        stats.summarize_interventions(frame, n_bootstrap=0)


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.iloc[1:],
    lambda frame: pd.concat([frame, frame.iloc[:1]], ignore_index=True),
    lambda frame: frame[~(frame.model.eq("eq_on") & frame.case_id.eq("a"))],
    lambda frame: frame[frame.model.ne("eq_on")],
    lambda frame: frame[frame.intervention.ne("ones_constant")],
    lambda frame: frame.assign(repeat=frame.repeat.replace({1: 2})),
    lambda frame: frame.assign(patient_id=np.where(frame.index == 1, "changed", frame.patient_id)),
    lambda frame: frame.assign(image_name=np.where(frame.index == 1, "changed.png", frame.image_name)),
])
def test_rejects_missing_duplicate_or_unpaired_rows(mutation):
    with pytest.raises(ValueError):
        stats.summarize_interventions(mutation(predictions()), n_bootstrap=0)


def test_rejects_reference_probability_changes_and_nonidentity_original():
    frame = predictions()
    for label_id, label in enumerate(stats.CLASS_ORDER):
        frame.loc[1, "original_prob_" + label] = WRONG[label_id]
    with pytest.raises(ValueError, match="Original probabilities change"):
        stats.summarize_interventions(frame, n_bootstrap=0)
    frame = predictions()
    for label_id, label in enumerate(stats.CLASS_ORDER):
        frame.loc[0, "prob_" + label] = WRONG[label_id]
    with pytest.raises(ValueError, match="retain exact"):
        stats.summarize_interventions(frame, n_bootstrap=0)


def test_derived_columns_are_verified_and_do_not_override_probability_oracle():
    frame = predictions()
    frame["flipped"] = 0
    stats.summarize_interventions(frame, n_bootstrap=0)
    frame.loc[1, "flipped"] = 1
    with pytest.raises(ValueError, match="Stored derived column"):
        stats.summarize_interventions(frame, n_bootstrap=0)


def test_single_patient_and_zero_bootstrap_have_undefined_intervals():
    for frame, n_bootstrap in ((predictions(cases=(("a", "one_patient"),)), 100),
                               (predictions(), 0)):
        summary, _ = stats.summarize_interventions(frame, n_bootstrap=n_bootstrap)
        assert summary.flip_rate_ci_low.isna().all()
        assert summary.flip_rate_ci_n_valid.eq(0).all()
        assert summary.flip_rate.eq(0).all()


def test_output_files_preserve_runner_artifacts_and_refuse_overwrite(tmp_path):
    marker = tmp_path / "raw_predictions.csv"
    marker.write_text("existing", encoding="utf-8")
    summary, combined = stats.write_outputs(predictions(), tmp_path, n_bootstrap=20)
    assert marker.read_text() == "existing"
    assert len(summary) == len(combined) == 8
    for filename in ("intervention_summary.csv", "intervention_combined.csv", "summary.md"):
        assert (tmp_path / filename).exists()
    report = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "不包含训练种子不确定性" in report
    assert "不可混用" in report
    with pytest.raises(FileExistsError):
        stats.write_outputs(predictions(), tmp_path, n_bootstrap=20)
