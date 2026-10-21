"""Statistical pairing, missing classes, identity integrity and safe preview tests."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.evaluate_region_relations import (
    VARIANTS, classification_metrics, paired_comparisons, read_identity_splits,
    resolve_split_images, validate_cross_seed_protocol, validate_prediction_pairs,
)
from scripts.preview_region_candidates import choose_examples, preview_tile, render_candidate_overlay
from src.region_candidates import CandidateConfig, extract_candidates


def predictions(seeds=(42,)):
    rows = []
    for seed in seeds:
        for variant in VARIANTS:
            for patient, label in [("001", 0), ("001", 1), ("002", 2), ("003", 0), ("003", 1), ("003", 2)]:
                pred = label
                if variant in ("baseline", "no_geometry_relations") and label == 1:
                    pred = 2
                probability = np.full(3, .05)
                probability[pred] = .9
                rows.append({"seed": seed, "variant": variant, "patient_id": patient,
                             "case_id": f"{patient}:{label}", "label": ("lt", "yz", "zz")[label],
                             "label_id": label, "pred": pred, "candidate_count": 2,
                             **{f"prob_{name}": probability[index] for index, name in enumerate(("lt", "yz", "zz"))}})
    return pd.DataFrame(rows)


def test_perfect_predictions_and_missing_class_auc():
    metrics = classification_metrics(np.array([0, 1, 2]), np.eye(3))
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["macro_ovr_auc"] == 1
    assert metrics["n_errors"] == metrics["yz_to_zz"] == metrics["zz_to_yz"] == 0
    missing = classification_metrics(np.array([0, 1]), np.array([[1., 0., 0.], [0., 1., 0.]]))
    assert np.isnan(missing["macro_ovr_auc"])
    assert missing["macro_f1"] == pytest.approx(2 / 3)
    assert missing["lt_auc"] == missing["yz_auc"] == 1
    assert np.isnan(missing["zz_auc"])


def test_auc_ties_and_probability_validation():
    result = classification_metrics([0, 1, 2], np.full((3, 3), 1 / 3))
    assert result["macro_ovr_auc"] == .5
    with pytest.raises(ValueError, match="distribution"):
        classification_metrics([0], [[.8, .8, .1]])


def test_patient_bootstrap_is_shared_across_seeds_not_extra_patients():
    single = paired_comparisons(predictions(), bootstrap=500, seed=7)
    repeated = paired_comparisons(predictions((42, 2026)), bootstrap=500, seed=7)
    aggregate = repeated[repeated.seed == "mean_across_seeds"].reset_index(drop=True)
    assert np.allclose(single.delta, aggregate.delta)
    assert np.allclose(single.ci95_low, aggregate.ci95_low)
    assert np.allclose(single.ci95_high, aggregate.ci95_high)
    assert aggregate.n_patients.eq(3).all()
    assert aggregate.n_seeds.eq(2).all()
    row = aggregate[(aggregate.comparison == "relations_minus_baseline") & (aggregate.metric == "accuracy")].iloc[0]
    assert row.delta == pytest.approx(2 / 6)
    assert row.corrected_images_mean_per_seed == 2
    assert row.broken_images_mean_per_seed == 0


def test_cluster_sampling_preserves_all_patient_images():
    data = predictions()
    # Three clusters contribute 2,1,3 images. Enumerate the same bootstrap RNG
    # draws explicitly: an image-level bootstrap would produce a different CI.
    got = paired_comparisons(data, bootstrap=1000, seed=17)
    draws = np.random.default_rng(17).integers(0, 3, size=(1000, 3))
    counts = np.array([2, 1, 3])
    corrected = np.array([1, 0, 1])
    expected = corrected[draws].sum(1) / counts[draws].sum(1)
    row = got[(got.comparison == "relations_minus_baseline") & (got.metric == "accuracy")].iloc[0]
    assert row.ci95_low == pytest.approx(np.quantile(expected, .025))
    assert row.ci95_high == pytest.approx(np.quantile(expected, .975))


def test_prediction_pair_integrity_rejects_missing_duplicate_or_relabelled_cases():
    data = predictions()
    with pytest.raises(ValueError, match="identical cases"):
        validate_prediction_pairs(data.drop(data.index[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prediction_pairs(pd.concat([data, data.iloc[[0]]]))
    changed = data.copy()
    changed.loc[0, "patient_id"] = "different"
    with pytest.raises(ValueError, match="identical cases"):
        validate_prediction_pairs(changed)


def test_zero_bootstrap_keeps_delta_and_leaves_ci_undefined():
    output = paired_comparisons(predictions(), bootstrap=0)
    assert output.ci95_low.isna().all() and output.ci95_high.isna().all()
    assert output.delta.notna().all()


def test_preview_empty_candidate_and_fixed_selection():
    image = Image.new("RGB", (224, 224), (150, 150, 150))
    candidates = extract_candidates(image, CandidateConfig())
    assert not candidates["valid"].any()
    assert np.array_equal(np.asarray(render_candidate_overlay(image, candidates)), np.asarray(image))
    tile = preview_tile(image, candidates, "example lt")
    assert tile.size == (464, 294)
    frame = pd.DataFrame({"label": ["yz", "lt", "lt", "zz"], "patient_id": ["002", "003", "001", "001"],
                          "image_name": ["2yz.png", "3lt.png", "1lt.png", "1zz.png"]})
    assert choose_examples(frame, 1) == [2, 0, 3]


def test_split_validation_and_resolution_do_not_access_test_images(tmp_path, monkeypatch):
    for split, patient in (("train", "001"), ("val", "002"), ("test", "003")):
        pd.DataFrame([{"patient_id": patient, "label": "lt", "label_id": 0,
                       "path": f"/nonexistent/{patient}lt.png"}]).to_csv(tmp_path / f"{split}.csv", index=False)
    Image.new("RGB", (20, 20)).save(tmp_path / "002lt.png")
    frames = read_identity_splits(tmp_path)
    original = Path.is_file
    checked = []
    def tracked_is_file(path):
        checked.append(path.name)
        return original(path)
    monkeypatch.setattr(Path, "is_file", tracked_is_file)
    resolved = resolve_split_images(frames, "val", tmp_path)
    assert resolved.patient_id.tolist() == ["002"]
    assert checked == ["002lt.png"]
    pd.DataFrame([{"patient_id": "001", "label": "lt", "label_id": 0,
                   "path": "/other/test.png"}]).to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="Patient overlap"):
        read_identity_splits(tmp_path)


def test_completed_training_evaluation_and_provenance_rejection(tmp_path, monkeypatch):
    from scripts import evaluate_region_relations as evaluator
    from scripts import run_region_relation_ablation as runner
    from src import region_relations
    from tests.test_region_relation_training import TinyRegionModel, _args, _splits
    splits, images = _splits(tmp_path)
    monkeypatch.setattr(runner, "build_region_relation_resnet50", TinyRegionModel)
    monkeypatch.setattr(region_relations, "build_region_relation_resnet50", TinyRegionModel)
    args = _args(tmp_path, splits, images, stage1_epochs=0)
    completed = runner.run(args)
    evaluation_args = evaluator.parse_args([
        "--run-dir", str(completed[0]), "--splits-dir", str(splits),
        "--image-root", str(images), "--output-dir", str(tmp_path / "evaluation"),
        "--device", "cpu", "--bootstrap", "25",
    ])
    evaluator.run(evaluation_args)
    exported = pd.read_csv(tmp_path / "evaluation" / "per_image_predictions.csv", dtype={"patient_id": str})
    assert len(exported) == 2 * 4
    assert set(exported.patient_id) == {"003"}
    assert set(exported.variant) == set(VARIANTS)
    assert len(pd.read_csv(tmp_path / "evaluation" / "paired_comparison.csv")) == 8
    assert (tmp_path / "evaluation" / "summary.md").is_file()
    import copy
    import json
    first_config = json.loads((completed[0] / "baseline" / "run_config.json").read_text())
    next_config = copy.deepcopy(first_config)
    next_config["seed"] = 2026
    next_config["arguments"]["output_root"] = "/other/seed/output"
    validate_cross_seed_protocol(first_config, next_config)
    next_config["arguments"]["stage2_epochs"] += 1
    with pytest.raises(ValueError, match="Across-seed training setting mismatch"):
        validate_cross_seed_protocol(first_config, next_config)
    next_config = copy.deepcopy(first_config)
    next_config["model_config"]["pretrained"] = not first_config["model_config"]["pretrained"]
    with pytest.raises(ValueError, match="Across-seed protocol mismatch: model_config"):
        validate_cross_seed_protocol(first_config, next_config)
    config_path = completed[0] / "baseline" / "run_config.json"
    config_path.write_text(config_path.read_text() + " ")
    with pytest.raises(ValueError, match="config hash mismatch"):
        evaluator.validate_training_run(completed[0], evaluator.read_identity_splits(splits))
