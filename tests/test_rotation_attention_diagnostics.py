"""Fixed-rotation attention diagnostic: protocol equivalence and known-answer checks."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import diagnose_attention as clean_diagnostic
from scripts import diagnose_rotation_attention as diagnostic
from scripts import evaluate_rotation_pairs as rotation
from tests.test_attention_diagnostics import TinyAttention, manifest, write_controlled_run


def write_rotation_run(tmp_path, manifest, seed=42):
    directory, split = write_controlled_run(tmp_path, manifest, seed)
    pair_path = directory / "pair_metadata.json"
    pair = json.loads(pair_path.read_text())
    for name in diagnostic.MODELS:
        path = directory / name / "run_config.json"
        config = json.loads(path.read_text())
        config["source_sha256"]["src/regularization.py"] = diagnostic.sha256_file(
            diagnostic.PROJECT_ROOT / "src/regularization.py")
        clean_diagnostic._write_json(path, config)
        pair["arms"][name]["run_config_sha256"] = diagnostic.sha256_file(path)
    clean_diagnostic._write_json(pair_path, pair)
    return directory, split


def make_models():
    torch.manual_seed(23)
    return {name: TinyAttention().eval() for name in diagnostic.MODELS}


def test_original_matches_previous_rotation_evaluator_at_all_seven_angles(manifest):
    models = make_models()
    old = rotation.collect_predictions(models, manifest, batch_size=2)
    new, stats, raw, cases, verification = diagnostic.collect_pair(
        models, manifest, 42, shuffles=2, batch_size=2)
    keys = ["model", "case_id", "angle_deg"]
    probabilities = [f"prob_{label}" for label in diagnostic.CLASSES]
    original = new[new.intervention.eq("original")].set_index(keys).sort_index()
    previous = old.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(original[probabilities], previous[probabilities],
                                  rtol=1e-6, atol=1e-7)
    assert (original.intervened_pred_id == previous.rotated_pred_id).all()
    assert set(original.index.get_level_values("angle_deg")) == set(diagnostic.ANGLES)
    assert len(new) == 3 * 2 * 7 * (3 + 2)
    assert len(stats) == 3 * 2 * 7
    assert list(cases) == sorted(manifest.case_id)
    assert all(values.shape == (3, 7, 7, 7) for values in raw.values())
    assert verification["state_unchanged"]
    assert verification["state_sha256_before"] == verification["state_sha256_after"]
    assert all(count == 2 for arm in verification["verified_forward_batches_per_angle"].values()
               for count in arm.values())
    assert not original.flipped.any()
    assert not original.attention_l1_change.any()
    assert not original.probability_l1_change.any()


def test_zero_angle_matches_original_image_diagnostic_exactly(manifest):
    models = make_models()
    clean = clean_diagnostic.collect_pair(models, manifest, 42, shuffles=2, batch_size=3)
    rotated = diagnostic.collect_pair(models, manifest, 42, shuffles=2, batch_size=3)
    zero = rotated[0].loc[rotated[0].angle_deg.eq(0)].drop(columns="angle_deg").reset_index(drop=True)
    pd.testing.assert_frame_equal(zero, clean[0], check_exact=True)
    zero_index = diagnostic.ANGLES.index(0.)
    for name in diagnostic.MODELS:
        np.testing.assert_array_equal(rotated[2][name][:, zero_index], clean[2][name])


def test_rotated_mean_uses_same_angle_native_map_and_reuses_zero(manifest, monkeypatch):
    calls = []
    original_rotate = diagnostic.rotate_inputs

    def record(inputs, angle):
        result = original_rotate(inputs, angle)
        calls.append((angle, result is inputs))
        return result

    monkeypatch.setattr(diagnostic, "rotate_inputs", record)
    frame, _, maps, case_ids, _ = diagnostic.collect_pair(
        make_models(), manifest, 42, shuffles=1, batch_size=3)
    assert calls == [(angle, angle == 0.) for angle in diagnostic.ANGLES]
    for name, values in maps.items():
        selected = frame[(frame.model == name) & (frame.intervention == "mean_constant")]
        for i, case_id in enumerate(case_ids):
            for j, angle in enumerate(diagnostic.ANGLES):
                raw = values[i, j].astype(float)
                expected_difference = np.abs(raw - raw.mean()).mean()
                row = selected[(selected.case_id == case_id) & (selected.angle_deg == angle)].iloc[0]
                assert row.attention_l1_change == pytest.approx(expected_difference, abs=1e-7)
        # The fixture's rotated native means differ; a cached zero-angle mean would be wrong.
        assert not np.allclose(values[:, 0].mean((1, 2)),
                               values[:, diagnostic.ANGLES.index(0.)].mean((1, 2)), atol=1e-5)


def test_case_order_batch_size_and_shuffle_repeats_are_paired(manifest):
    first = diagnostic.collect_pair(make_models(), manifest, 42, shuffles=2, batch_size=1)
    second = diagnostic.collect_pair(make_models(), manifest.iloc[::-1], 42, shuffles=2, batch_size=3)
    pd.testing.assert_frame_equal(first[0], second[0], rtol=1e-5, atol=1e-7)
    assert first[0].groupby(["model", "case_id", "angle_deg"]).size().eq(5).all()
    shuffled = first[0][first[0].intervention.eq("spatial_shuffle")]
    assert shuffled.groupby(["model", "case_id", "angle_deg"])["repeat"].apply(
        lambda values: set(values) == {0, 1}).all()


@pytest.mark.parametrize("args", [
    ["--split", "test"], ["--angles", "-5", "0", "5"], ["--shuffles", "0"],
    ["--batch-size", "0"], ["--bootstrap", "-1"], ["--seed", "-1"],
])
def test_cli_rejects_new_angles_test_split_and_invalid_counts(args):
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--run-dirs", "run", "--output-dir", "out"] + args)


def test_rotation_source_provenance_is_required(manifest, tmp_path):
    directory, split = write_controlled_run(tmp_path, manifest)
    with pytest.raises(ValueError, match="Rotation source differs"):
        diagnostic.load_run_provenance([directory], split / "val.csv", manifest)


def test_rotation_provenance_accepts_matching_source(manifest, tmp_path):
    directory, split = write_rotation_run(tmp_path, manifest)
    runs = diagnostic.load_run_provenance([directory], split / "val.csv", manifest)
    assert len(runs) == 1 and runs[0]["seed"] == 42


def test_historical_comparison_reports_changes_without_substitution(manifest, tmp_path, capsys):
    models = make_models()
    old = rotation.collect_predictions(models, manifest, batch_size=3)
    current = diagnostic.collect_pair(models, manifest, 42, shuffles=1, batch_size=3)[0]
    original_current = current.copy(deep=True)
    missing = diagnostic.compare_historical_original(current, tmp_path)
    assert missing["status"] == "missing"
    directory = tmp_path / "rotation_val"
    directory.mkdir()
    path = directory / "per_image_predictions.csv"
    old.to_csv(path, index=False)
    same = diagnostic.compare_historical_original(current, tmp_path)
    assert same["status"] == "compared" and same["labels_identical"]
    assert same["probability_max_abs_difference"] < 1e-7
    probabilities = [f"prob_{label}" for label in diagnostic.CLASSES]
    # Differences beyond a tight forward tolerance must be recorded, not rejected.
    old.loc[0, probabilities] = [.333, .333, .334]
    old.to_csv(path, index=False)
    changed = diagnostic.compare_historical_original(current, tmp_path)
    assert changed["probability_max_abs_difference"] > .001
    pd.testing.assert_frame_equal(current, original_current, check_exact=True)
    old.loc[0, probabilities] = np.roll(current.loc[
        current.intervention.eq("original") & current.case_id.eq(old.loc[0, "case_id"])
        & current.model.eq(old.loc[0, "model"]) & current.angle_deg.eq(old.loc[0, "angle_deg"]),
        probabilities].to_numpy()[0], 1)
    old.to_csv(path, index=False)
    changed = diagnostic.compare_historical_original(current, tmp_path)
    assert changed["n_label_disagreements"] >= 1
    assert "WARNING:" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["identity", "grid", "duplicate", "probabilities"])
def test_historical_comparison_rejects_mismatched_records(manifest, tmp_path, fault):
    models = make_models()
    old = rotation.collect_predictions(models, manifest, batch_size=3)
    current = diagnostic.collect_pair(models, manifest, 42, shuffles=1, batch_size=3)[0]
    directory = tmp_path / "rotation_val"
    directory.mkdir()
    if fault == "identity":
        old.loc[0, "patient_id"] = "wrong"
    elif fault == "grid":
        old = old.iloc[1:]
    elif fault == "duplicate":
        old = pd.concat([old, old.iloc[:1]])
    else:
        old.loc[0, "prob_lt"] = float("nan")
    old.to_csv(directory / "per_image_predictions.csv", index=False)
    with pytest.raises(ValueError):
        diagnostic.compare_historical_original(current, tmp_path)


def test_complete_cli_saves_all_angles_raw_maps_and_metadata(manifest, tmp_path, monkeypatch):
    directory, split = write_rotation_run(tmp_path, manifest)
    monkeypatch.setattr(clean_diagnostic, "build_attention_resnet50", TinyAttention)
    output = tmp_path / "rotation_attention"
    args = ["--run-dirs", str(directory), "--splits-dir", str(split),
            "--output-dir", str(output), "--device", "cpu", "--batch-size", "3",
            "--shuffles", "2", "--bootstrap", "20"]
    diagnostic.main(args)
    for filename in ("model_summary.csv", "model_combined.csv", "eq_gain_summary.csv",
                     "eq_gain_combined.csv", "summary.md", "per_image_predictions.csv",
                     "attention_stats.csv"):
        assert (output / filename).is_file()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["completed"] and metadata["split"] == "val"
    assert metadata["test_images_loaded"] is False
    assert metadata["n_prediction_rows"] == 210
    assert metadata["rotation"]["angles_deg"] == list(diagnostic.ANGLES)
    assert "src/regularization.py" in metadata["source_sha256"]
    assert "scripts/diagnose_attention.py" in metadata["source_sha256"]
    assert metadata["runs"][0]["historical_original_comparison"]["status"] == "missing"
    with np.load(output / "seed_42/raw_maps/eq_off.npz", allow_pickle=False) as raw:
        assert raw["maps"].shape == (3, 7, 7, 7)
        assert raw["maps"].dtype == np.float32
        assert raw["case_ids"].dtype.kind == "U"
        np.testing.assert_array_equal(raw["angles_deg"], diagnostic.ANGLES)
    with pytest.raises(ValueError, match="not overwritten"):
        diagnostic.main(args)
