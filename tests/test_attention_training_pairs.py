"""Tests of the matched attention-training evaluator; no pretrained downloads."""
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from scripts import evaluate_attention_training_pairs as evaluation
from scripts import evaluate_rotation_pairs as rotation


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def split_record(patient, image="absent.png"):
    members = [dict(patient_id=patient, label="lt", label_id=0, image_name=image)]
    return dict(
        members=members, n_images=1, n_patients=1, csv_sha256="unused",
        membership_sha256=hashlib.sha256(json.dumps(
            members, ensure_ascii=False, sort_keys=True).encode()).hexdigest())


def make_run(tmp_path, include_regularized=True):
    run_dir = tmp_path / "seed_42"
    run_dir.mkdir()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    val_csv = split_dir / "val.csv"
    val_csv.write_text("fixture", encoding="utf-8")
    splits = {name: split_record(name) for name in ("train", "val", "test")}
    splits["val"]["csv_sha256"] = evaluation.sha256_file(val_csv)
    arguments = dict(stage1_epochs=0, stage2_epochs=1, batch_size=2, dropout=.3, lambda_eq=.1, lambda_pf=.1)
    pair = dict(
        completed=True, protocol_version=evaluation.PROTOCOL_VERSION, seed=42,
        primary_checkpoint="last_model.pt", test_images_loaded=False,
        initial_states_identical=True, training_input_streams_identical=True,
        execution_traces_identical=True, arms={})
    for arm in evaluation.ARMS if include_regularized else evaluation.ARMS[:2]:
        directory = run_dir / arm
        directory.mkdir()
        checkpoint = directory / "last_model.pt"
        checkpoint.write_bytes(arm.encode())
        traces = [dict(training_epoch=1, n_batches=1, n_images=1, augmented_input_sha256="same")]
        trace_path = directory / "training_input_trace.csv"
        pd.DataFrame(traces).to_csv(trace_path, index=False)
        execution_traces = [dict(
            training_epoch=1, model_forward_count=3, train_forward_count=2, eval_forward_count=1,
            forward_rng_sha256="a" * 64, end_rng_sha256="b" * 64,
            bn_forward_counts={"backbone.bn": {"train": 2, "eval": 1}},
            bn_num_batches_tracked={"backbone.bn": 2})]
        execution_path = directory / "execution_trace.json"
        write_json(execution_path, dict(schema_version=1, epochs=execution_traces, policy="matched"))
        policy = evaluation.POLICIES[arm]
        control = dict(policy="fixed" if arm == "fixed_ce" else "learned",
                       initialization="zero_spatial_convolution", initial_constant=.5,
                       trainable=arm != "fixed_ce", kernel_size=3, hires=False)
        config = dict(
            arm=arm, seed=42, completed=True, protocol_version=evaluation.PROTOCOL_VERSION,
            attention_policy=policy, attention_control=control,
            checkpoint_reload_verified=True,
            fixed_architecture="AnatomyAttentionResNet50_layer4_7x7",
            crop_border=True, attn_hires=False, primary_checkpoint="last_model.pt",
            evaluation_split="val", test_images_loaded=False, eq_auxiliary_forward=True,
            arguments=arguments, completed_epochs=1, initial_state_sha256="same",
            regularization=dict(lambda_eq=.1 if arm == "learned_regularized" else 0.,
                                lambda_pf=.1 if arm == "learned_regularized" else 0.,
                                max_angle=15., eq_auxiliary_forward=True),
            checkpoint_sha256={"last_model.pt": evaluation.sha256_file(checkpoint)},
            training_input_trace_sha256=evaluation.sha256_file(trace_path), training_input_traces=traces,
            execution_trace_sha256=evaluation.sha256_file(execution_path), execution_traces=execution_traces,
            attention_parameter_checks=[dict(training_epoch=1, count_nonzero=0, requires_grad=arm != "fixed_ce")],
            splits=splits, source_sha256={"src/attention.py": "same"}, batchnorm_policy="matched")
        config_path = directory / "run_config.json"
        write_json(config_path, config)
        pair["arms"][arm] = dict(
            initial_state_sha256="same", checkpoint_sha256=evaluation.sha256_file(checkpoint),
            run_config_sha256=evaluation.sha256_file(config_path),
            training_input_trace_sha256=config["training_input_trace_sha256"],
            execution_trace_sha256=config["execution_trace_sha256"],
            attention_policy=policy, attention_control=control)
    write_json(run_dir / "pair_metadata.json", pair)
    return run_dir, val_csv


def update_config(run_dir, arm, mutate):
    path = run_dir / arm / "run_config.json"
    config = json.loads(path.read_text())
    mutate(config)
    write_json(path, config)
    pair_path = run_dir / "pair_metadata.json"
    pair = json.loads(pair_path.read_text())
    pair["arms"][arm]["run_config_sha256"] = evaluation.sha256_file(path)
    write_json(pair_path, pair)


def analytical_predictions():
    rows = []
    correct, wrong = np.array([.8, .1, .1]), np.array([.1, .8, .1])
    for model in evaluation.ARMS:
        for case in "ABC":
            clean = wrong if ((model == "learned_ce" and case == "C")
                              or (model == "learned_regularized" and case == "B")) else correct
            for angle in evaluation.DEFAULT_ANGLES:
                if angle == 0:
                    probs = clean
                elif model == "fixed_ce":
                    probs = correct if case == "B" else wrong
                elif model == "learned_ce":
                    probs = correct if case == "A" or (case == "B" and angle > 0) else wrong
                else:
                    probs = correct if case != "B" else wrong
                base = dict(case_id=case, patient_id="p1" if case == "A" else "p2",
                            true_label="lt", true_label_id=0, image_name=case, angle_deg=angle)
                base.update({column: 0. for column in rotation.ATTENTION_COLUMNS})
                rows.append(rotation.prediction_row(base, model, clean, probs))
    return pd.DataFrame(rows)


def test_pairs_have_true_names_direction_and_separate_common_cohorts():
    summary, comparisons = evaluation.summarize_pairs(analytical_predictions(), 10000, seed=31)
    primary = comparisons[comparisons.scope.eq("nonzero_mean")].set_index("comparison")
    first = primary.loc["learned_ce-minus-fixed_ce"]
    assert first.rotated_accuracy_difference == pytest.approx(1 / 6)
    assert first.common_clean_correct_to_wrong_rate_difference == -.25
    assert first.n_common_clean_correct_images == 2
    assert first.rotated_accuracy_ci_low == -.25
    assert first.rotated_accuracy_ci_high == 1
    assert first.fixed_ce_rotated_accuracy == pytest.approx(1 / 3)
    assert first.learned_ce_rotated_accuracy == .5
    second = primary.loc["learned_regularized-minus-learned_ce"]
    assert second.n_common_clean_correct_images == 1
    assert second.common_clean_correct_to_wrong_rate_difference == 0
    assert len(summary) == 2 * 2 * 8
    assert not any("eq_off" in column or "eq_on" in column for column in comparisons)
    assert not comparisons.astype(str).apply(lambda col: col.str.contains("eq_off|eq_on")).any().any()
    assert set(summary.model) == set(evaluation.ARMS)
    zeros = summary[summary.angle_deg.eq(0)]
    assert zeros.flip_rate.eq(0).all()
    assert zeros.true_probability_delta_mean.eq(0).all()


def test_zero_common_denominator_stays_nan_and_patient_bootstrap_is_paired():
    frame = analytical_predictions()
    for arm in ("fixed_ce", "learned_ce"):
        selected = frame.model.eq(arm)
        frame.loc[selected, "clean_correct"] = 0
        frame.loc[selected, "clean_prob_lt"] = .1
        frame.loc[selected, "clean_prob_yz"] = .8
        zero = selected & frame.angle_deg.eq(0)
        frame.loc[zero, "rotated_correct"] = 0
        frame.loc[zero, "prob_lt"] = .1
        frame.loc[zero, "prob_yz"] = .8
    _, comparison = evaluation.summarize_pairs(frame, 20)
    primary = comparison[comparison.comparison.eq("learned_ce-minus-fixed_ce")]
    assert primary.common_clean_correct_to_wrong_rate_difference.isna().all()
    assert primary.common_clean_correct_to_wrong_rate_bootstrap_valid.eq(0).all()
    assert primary.common_clean_correct_to_wrong_rate_ci_low.isna().all()


def test_missing_fixed_angle_rejected():
    with pytest.raises(ValueError, match="seven"):
        evaluation.summarize_pairs(analytical_predictions().query("angle_deg != -15"))


def test_valid_two_and_three_arm_runs(tmp_path):
    run_dir, val_csv = make_run(tmp_path)
    provenance = evaluation.validate_training_run(run_dir, val_csv)
    assert set(provenance) == set(evaluation.ARMS) | {"pair_metadata"}


@pytest.mark.parametrize("fault", ["aux_forward", "bn_count", "bn_increment", "dropped_images"])
def test_matching_but_wrong_schedule_is_still_rejected(tmp_path, fault):
    run_dir, _ = make_run(tmp_path)
    config = json.loads((run_dir / "fixed_ce" / "run_config.json").read_text())
    traces, execution = config["training_input_traces"], config["execution_traces"]
    if fault == "aux_forward":
        execution[0]["train_forward_count"] = 1
    elif fault == "bn_count":
        execution[0]["bn_forward_counts"]["backbone.bn"]["train"] = 1
    elif fault == "dropped_images":
        traces[0]["n_images"] = 0
    else:
        traces.append(dict(traces[0], training_epoch=2))
        execution.append(dict(execution[0], training_epoch=2))
    with pytest.raises(ValueError):
        evaluation._validate_execution_schedule(config, traces, execution)


@pytest.mark.parametrize("fault", [
    "incomplete", "best_checkpoint", "policy", "loss", "seed", "epochs",
    "input_stream", "execution", "checkpoint", "test_loaded", "frozen_changed", "source",
])
def test_rejects_invalid_or_unmatched_training(tmp_path, fault):
    run_dir, val_csv = make_run(tmp_path)
    if fault == "checkpoint":
        (run_dir / "learned_ce" / "last_model.pt").write_bytes(b"replaced")
    elif fault == "frozen_changed":
        update_config(run_dir, "fixed_ce", lambda c: c["attention_parameter_checks"][0].update(count_nonzero=1))
    else:
        def mutate(config):
            if fault == "incomplete":
                config["completed"] = False
            elif fault == "best_checkpoint":
                config["primary_checkpoint"] = "best_val.pt"
            elif fault == "policy":
                config["attention_policy"] = "fixed_constant_half"
            elif fault == "loss":
                config["regularization"]["lambda_eq"] = .1
            elif fault == "seed":
                config["seed"] = 43
            elif fault == "epochs":
                config["completed_epochs"] = 0
            elif fault == "input_stream":
                config["training_input_traces"][0]["augmented_input_sha256"] = "different"
            elif fault == "execution":
                config["execution_traces"][0]["model_forward_count"] = 99
            elif fault == "test_loaded":
                config["test_images_loaded"] = True
            elif fault == "source":
                config["source_sha256"] = {"src/attention.py": "changed"}
        update_config(run_dir, "learned_ce", mutate)
    with pytest.raises(ValueError):
        evaluation.validate_training_run(run_dir, val_csv)


def test_validation_manifest_replacement_and_patient_overlap_rejected(tmp_path):
    run_dir, val_csv = make_run(tmp_path)
    original = val_csv.read_text()
    val_csv.write_text("replaced")
    with pytest.raises(ValueError, match="validation split"):
        evaluation.validate_training_run(run_dir, val_csv)
    val_csv.write_text(original)
    for arm in evaluation.ARMS:
        update_config(run_dir, arm, lambda c: c["splits"].update(test=split_record("val")))
    with pytest.raises(ValueError, match="contamination"):
        evaluation.validate_training_run(run_dir, val_csv)


def test_cli_cannot_select_test_or_angles_or_best_checkpoint(tmp_path):
    args = ["--run-dir", str(tmp_path), "--splits-dir", str(tmp_path), "--output-dir", "out"]
    for disallowed in (["--split", "test"], ["--angles", "0", "5"],
                       ["--checkpoint", "best_val.pt"]):
        with pytest.raises(SystemExit):
            evaluation.parse_args(args + disallowed)


class TinyAttention(torch.nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        self.fixed = fixed

    def forward(self, inputs, return_attn=False):
        logits = inputs.mean((2, 3))
        attention = (torch.full((len(inputs), 1, 7, 7), .5, device=inputs.device)
                     if self.fixed else F.adaptive_avg_pool2d(inputs[:, :1], (7, 7)).sigmoid())
        return (logits, attention) if return_attn else logits


def test_collection_requires_fixed_half_and_zero_is_exact(tmp_path):
    image_path = tmp_path / "val.png"
    Image.fromarray(np.full((80, 100, 3), 128, dtype=np.uint8)).save(image_path)
    manifest = validate_image_manifest(image_path)
    with pytest.raises(ValueError, match="constant"):
        evaluation.collect_predictions({"fixed_ce": TinyAttention(), "learned_ce": TinyAttention()}, manifest)
    predictions = evaluation.collect_predictions(
        {"fixed_ce": TinyAttention(True), "learned_ce": TinyAttention()}, manifest)
    assert len(predictions) == 14
    zero = predictions[predictions.angle_deg.eq(0)]
    assert zero.true_probability_delta.eq(0).all()
    fixed = predictions[predictions.model.eq("fixed_ce")]
    np.testing.assert_allclose(fixed.rotated_attention_std, 0)
    np.testing.assert_allclose(fixed.rotated_attention_entropy, 1)


def validate_image_manifest(image_path):
    return evaluation.validate_manifest(pd.DataFrame([
        dict(path=str(image_path), patient_id="val", label="lt", label_id=0)]))


def test_main_opens_only_validation_images_and_outputs_real_arm_names(tmp_path, monkeypatch):
    run_dir, val_csv = make_run(tmp_path, include_regularized=False)
    image_path = tmp_path / "val.png"
    Image.fromarray(np.full((80, 100, 3), 128, dtype=np.uint8)).save(image_path)
    manifest = validate_image_manifest(image_path)
    manifest.drop(columns="case_id").to_csv(val_csv, index=False)
    for arm in evaluation.ARMS[:2]:
        def set_val(config):
            config["splits"]["val"] = split_record("val", image_path.name)
            config["splits"]["val"]["csv_sha256"] = evaluation.sha256_file(val_csv)
        update_config(run_dir, arm, set_val)
    real_open = Image.open
    opened = []
    def guarded_open(path, *args, **kwargs):
        assert str(path) == str(image_path)
        opened.append(str(path))
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", guarded_open)
    monkeypatch.setattr(evaluation, "load_models", lambda provenance, device: {
        "fixed_ce": TinyAttention(True), "learned_ce": TinyAttention()})
    output = tmp_path / "evaluation"
    evaluation.main(["--run-dir", str(run_dir), "--splits-dir", str(val_csv.parent),
                     "--output-dir", str(output), "--device", "cpu", "--bootstrap", "0"])
    assert opened == [str(image_path)]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["test_images_loaded"] is False
    assert metadata["n_prediction_rows"] == 14
    comparison = pd.read_csv(output / "paired_comparison.csv")
    assert comparison.comparison.eq("learned_ce-minus-fixed_ce").all()
    assert "fixed_ce_rotated_accuracy" in comparison
    assert "eq_off_rotated_accuracy" not in comparison
    assert "退化对照" in (output / "summary.md").read_text()
    with pytest.raises(ValueError, match="absent or empty"):
        evaluation.main(["--run-dir", str(run_dir), "--splits-dir", str(val_csv.parent),
                         "--output-dir", str(output)])
