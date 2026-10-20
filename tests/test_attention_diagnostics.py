"""Known-answer tests for fixed-weight attention interventions; CPU, no downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from scripts import diagnose_attention as diagnostic


class ChannelAttention(torch.nn.Module):
    def forward(self, features):
        return features[:, :1].sigmoid()


class TinyAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hires = False
        self.backbone = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.AdaptiveAvgPool2d((7, 7)))
        self.attention = ChannelAttention()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(6, 3))

    def forward(self, inputs, return_attn=False):
        features = self.backbone(inputs)
        attention = self.attention(features)
        global_features = self.pool(features).flatten(1)
        attended = self.pool(features * attention).flatten(1)
        logits = self.fc(torch.cat((global_features, attended), dim=1))
        return (logits, attention) if return_attn else logits


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i, label in enumerate(diagnostic.CLASSES):
        pixels = np.zeros((90, 110, 3), dtype=np.uint8)
        pixels[8:82, 12:100] = (65, 120, 80 + i * 10)
        pixels[22:65, 32:61] = (210 - i * 20, 50, 120)
        path = tmp_path / f"{i:03}{label}.png"
        Image.fromarray(pixels).save(path)
        rows.append(dict(patient_id=f"{i:03}", label=label, label_id=i, path=str(path)))
    return diagnostic.validate_manifest(pd.DataFrame(rows))


def write_controlled_run(tmp_path, manifest, seed=42):
    split = tmp_path / "splits"
    split.mkdir(exist_ok=True)
    manifest.drop(columns="case_id").to_csv(split / "val.csv", index=False)
    # Deliberately no train.csv, test.csv or test images: only val may be opened.
    directory = tmp_path / f"seed_{seed}"
    directory.mkdir()
    members = manifest.assign(image_name=manifest.path.map(lambda p: Path(p).name))[
        ["patient_id", "label", "label_id", "image_name"]
    ].sort_values(["patient_id", "label", "image_name"]).to_dict("records")
    configs, pair_arms = {}, {}
    for model in diagnostic.MODELS:
        arm = directory / model
        arm.mkdir()
        torch.manual_seed(seed + int(model == "eq_on"))
        net = TinyAttention().eval()
        checkpoint = arm / "last_model.pt"
        torch.save(net.state_dict(), checkpoint)
        config = dict(
            protocol_version="eq_sham_control_v1", completed=True, seed=seed, arm=model,
            fixed_architecture="AnatomyAttentionResNet50_layer4_7x7", crop_border=True,
            evaluation_split="val", eq_auxiliary_forward=True, primary_checkpoint="last_model.pt",
            arguments=dict(dropout=.3, seeds=[seed], output_root=f"output_{seed}", batch_size=32),
            regularization=dict(lambda_eq=.1 if model == "eq_on" else 0, lambda_pf=.1,
                                max_angle=15, eq_auxiliary_forward=True),
            splits={"val": dict(csv_sha256=diagnostic.sha256_file(split / "val.csv"), members=members)},
            source_sha256={name: diagnostic.sha256_file(diagnostic.PROJECT_ROOT / name)
                           for name in ("src/attention.py", "src/data.py", "src/viz.py")},
            initial_state_sha256=f"initial-{seed}",
            final_state_sha256=diagnostic.state_fingerprint(net), completed_epochs=25,
            checkpoint_sha256={"last_model.pt": diagnostic.sha256_file(checkpoint)},
            training_input_trace_sha256=f"trace-{seed}", training_input_traces=[seed])
        diagnostic._write_json(arm / "run_config.json", config)
        configs[model] = config
        pair_arms[model] = dict(
            checkpoint_sha256=diagnostic.sha256_file(checkpoint),
            run_config_sha256=diagnostic.sha256_file(arm / "run_config.json"),
            initial_state_sha256=config["initial_state_sha256"])
    diagnostic._write_json(directory / "pair_metadata.json", dict(
        protocol_version="eq_sham_control_v1", seed=seed, primary_checkpoint="last_model.pt",
        completed=True, initial_states_identical=True, training_input_streams_identical=True, arms=pair_arms))
    return directory, split


def test_interventions_preserve_identity_mean_and_shuffle_histogram():
    attention = torch.linspace(.01, .99, 98).reshape(2, 1, 7, 7)
    outputs = list(diagnostic.intervention_maps(attention, ["a", "b"], 3, 11))
    assert outputs[0][2] is attention
    constant = outputs[1][2]
    torch.testing.assert_close(constant.mean((-2, -1)), attention.mean((-2, -1)))
    torch.testing.assert_close(constant.flatten(1).std(1), torch.zeros(2), atol=1e-7, rtol=0)
    for kind, repeat, output in outputs[2:-1]:
        assert kind == "spatial_shuffle" and repeat >= 0
        torch.testing.assert_close(output.flatten(1).sort().values, attention.flatten(1).sort().values,
                                   atol=0, rtol=0)
    assert torch.equal(outputs[-1][2], torch.ones_like(attention))


def test_shuffles_are_case_batch_and_model_independent():
    attention = torch.arange(98).reshape(2, 1, 7, 7).float()
    first = diagnostic.shuffled_attention(attention, ["a", "b"], 4, 22)
    reversed_result = diagnostic.shuffled_attention(attention.flip(0), ["b", "a"], 4, 22).flip(0)
    torch.testing.assert_close(first, reversed_result, rtol=0, atol=0)
    singles = torch.cat([diagnostic.shuffled_attention(attention[i:i+1], [case], 4, 22)
                         for i, case in enumerate(["a", "b"])])
    torch.testing.assert_close(first, singles, rtol=0, atol=0)
    # A different model's values receive the same indices.
    shifted = diagnostic.shuffled_attention(attention + 100, ["a", "b"], 4, 22)
    torch.testing.assert_close(first + 100, shifted, rtol=0, atol=0)
    assert not torch.equal(first, diagnostic.shuffled_attention(attention, ["a", "b"], 5, 22))


def test_constant_maps_remain_constant_and_preserve_logits():
    model = TinyAttention().eval()
    features = torch.arange(147).reshape(1, 3, 7, 7).float() / 100
    global_features = model.pool(features).flatten(1)
    attention = torch.full((1, 1, 7, 7), .5)
    original = diagnostic.intervention_logits(model, features, global_features, attention)
    for kind, _, changed in diagnostic.intervention_maps(attention, ["a"], 2, 12):
        if kind == "ones_constant":
            continue
        torch.testing.assert_close(changed, attention, rtol=0, atol=0)
        torch.testing.assert_close(
            diagnostic.intervention_logits(model, features, global_features, changed), original,
            rtol=0, atol=0)


def test_known_position_dependent_head_changes_after_spatial_intervention():
    # Independent four-cell oracle: attended mean is (4*1 + 0*.0 + 0*.0 + 0*.0)/4 = 1.
    model = SimpleNamespace(pool=torch.nn.AdaptiveAvgPool2d(1), fc=torch.nn.Linear(2, 3, bias=False))
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[0., 1.], [0., -1.], [0., 0.]]))
    features = torch.tensor([[[[4., 0.], [0., 0.]]]])
    attention = torch.tensor([[[[1., 0.], [0., 0.]]]])
    global_features = model.pool(features).flatten(1)
    original = diagnostic.intervention_logits(model, features, global_features, attention)
    moved = diagnostic.intervention_logits(model, features, global_features, attention.flip(-1))
    uniform = diagnostic.intervention_logits(
        model, features, global_features, torch.full_like(attention, .25))
    torch.testing.assert_close(original, torch.tensor([[1., -1., 0.]]))
    torch.testing.assert_close(moved, torch.zeros(1, 3))
    torch.testing.assert_close(uniform, torch.tensor([[.25, -.25, 0.]]))


def test_cached_forward_matches_model_and_keeps_batchnorm_state():
    model = TinyAttention().eval()
    before = diagnostic.state_fingerprint(model)
    with torch.inference_mode():
        *_, errors = diagnostic.cached_forward(model, torch.randn(2, 3, 224, 224))
    assert errors == dict(logits_max_abs_error=0., attention_max_abs_error=0.)
    assert before == diagnostic.state_fingerprint(model)
    with pytest.raises(ValueError, match="eval"):
        diagnostic.cached_forward(model.train(), torch.randn(2, 3, 224, 224))


def test_forward_mismatch_is_detected():
    class Mismatched(TinyAttention):
        def forward(self, inputs, return_attn=False):
            logits, attention = super().forward(inputs, return_attn=True)
            return (logits + .1, attention) if return_attn else logits + .1
    with torch.inference_mode(), pytest.raises(AssertionError):
        diagnostic.cached_forward(Mismatched().eval(), torch.randn(1, 3, 224, 224))


def test_prediction_metric_known_answer_and_negative_drops():
    case = SimpleNamespace(label_id=0, label="lt", path="/unused.png", case_id="a", patient_id="001")
    row = diagnostic.prediction_rows(
        [case], 42, "eq_on", "mean_constant", -1, np.array([[.7, .2, .1]]),
        np.array([[.3, .6, .1]]), [.2])[0]
    assert row["flipped"] == row["correct_to_wrong"] == 1
    assert row["wrong_to_correct"] == 0
    assert row["true_probability_delta"] == pytest.approx(-.4)
    assert row["original_pred_probability_drop"] == pytest.approx(.4)
    assert row["probability_l1_change"] == pytest.approx(.8)
    improved = diagnostic.prediction_rows(
        [case], 42, "eq_on", "mean_constant", -1, np.array([[.7, .2, .1]]),
        np.array([[.9, .05, .05]]), [.2])[0]
    assert improved["original_pred_probability_drop"] == pytest.approx(-.2)


def test_collection_reordering_batch_invariance_and_identity(manifest):
    torch.manual_seed(22)
    models = {"eq_off": TinyAttention(), "eq_on": TinyAttention()}
    first = diagnostic.collect_pair(models, manifest, 42, shuffles=2, batch_size=1,
                                    render_ids=diagnostic.selected_case_ids(manifest, 1))
    second = diagnostic.collect_pair(models, manifest.iloc[::-1], 42, shuffles=2, batch_size=3)
    pd.testing.assert_frame_equal(first[0], second[0], rtol=1e-5, atol=1e-7)
    assert len(first[0]) == 3 * 2 * (3 + 2)
    identity = first[0][first[0].intervention.eq("original")]
    assert not identity.flipped.any()
    assert not identity.attention_l1_change.any()
    assert not identity.probability_l1_change.any()
    assert first[-1]["state_unchanged"]
    assert len(first[-2]) == 3
    assert all(base.shape == (224, 224, 3) for base in first[-2].values())


def test_attention_statistics_keep_raw_amplitude():
    cases = [SimpleNamespace(case_id="a", patient_id="001", path="/a.png", label="lt")]
    row = diagnostic.attention_statistics(cases, 42, "eq_on", torch.full((1, 1, 7, 7), .98))[0]
    assert row["mean"] == pytest.approx(.98)
    assert row["min"] == row["max"]
    assert row["std"] == 0
    assert row["normalized_entropy"] == pytest.approx(1)


def test_provenance_accepts_seed_list_output_root_differences(manifest, tmp_path):
    first, split = write_controlled_run(tmp_path, manifest, 42)
    second, _ = write_controlled_run(tmp_path, manifest, 2026)
    runs = diagnostic.load_run_provenance([second, first], split / "val.csv", manifest)
    assert [run["seed"] for run in runs] == [42, 2026]


@pytest.mark.parametrize("missing", ["pair_metadata.json", "eq_off/run_config.json", "eq_on/last_model.pt"])
def test_provenance_requires_complete_pair(manifest, tmp_path, missing):
    directory, split = write_controlled_run(tmp_path, manifest)
    (directory / missing).unlink()
    with pytest.raises(ValueError, match="requires both"):
        diagnostic.load_run_provenance([directory], split / "val.csv", manifest)


@pytest.mark.parametrize("fault", ["checkpoint", "val_sha", "member", "duplicate_seed", "pair_seed",
                                  "pair_protocol", "source_changed"])
def test_provenance_rejects_mismatch(manifest, tmp_path, fault):
    directory, split = write_controlled_run(tmp_path, manifest)
    directories = [directory]
    if fault == "checkpoint":
        with (directory / "eq_off/last_model.pt").open("ab") as handle:
            handle.write(b"modified")
    elif fault == "val_sha":
        with (split / "val.csv").open("a") as handle:
            handle.write("\n")
    elif fault == "member":
        manifest = manifest.copy()
        manifest.loc[0, "patient_id"] = "changed"
    elif fault == "duplicate_seed":
        directories.append(directory)
    elif fault == "source_changed":
        path = directory / "eq_off/run_config.json"
        content = json.loads(path.read_text())
        content["source_sha256"]["src/data.py"] = "changed"
        diagnostic._write_json(path, content)
    else:
        path = directory / "pair_metadata.json"
        content = json.loads(path.read_text())
        content["seed" if fault == "pair_seed" else "protocol_version"] = 99
        diagnostic._write_json(path, content)
    with pytest.raises(ValueError):
        diagnostic.load_run_provenance(directories, split / "val.csv", manifest)


def test_cli_rejects_test_split_and_invalid_counts():
    base = ["--run-dirs", "run", "--output-dir", "out"]
    for args in (["--split", "test"], ["--shuffles", "0"], ["--batch-size", "0"],
                 ["--bootstrap", "-1"], ["--render-per-class", "-1"]):
        with pytest.raises(SystemExit):
            diagnostic.parse_args(base + args)


def test_complete_tiny_cli_outputs_and_refuses_overwrite(manifest, tmp_path, monkeypatch):
    directory, split = write_controlled_run(tmp_path, manifest)
    monkeypatch.setattr(diagnostic, "build_attention_resnet50", TinyAttention)
    output = tmp_path / "diagnostic"
    args = ["--run-dirs", str(directory), "--splits-dir", str(split),
            "--output-dir", str(output), "--device", "cpu", "--batch-size", "2",
            "--shuffles", "2", "--bootstrap", "20", "--render-per-class", "0"]
    diagnostic.main(args)
    assert (output / "summary.md").is_file()
    assert (output / "intervention_summary.csv").is_file()
    assert (output / "intervention_combined.csv").is_file()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["completed"] and metadata["split"] == "val"
    assert metadata["test_images_loaded"] is False
    assert metadata["n_prediction_rows"] == 30
    assert "scripts/diagnose_attention.py" in metadata["source_sha256"]
    with np.load(output / "seed_42/raw_maps/eq_off.npz", allow_pickle=False) as raw:
        assert raw["maps"].shape == (3, 7, 7)
        assert raw["maps"].dtype == np.float32
        assert raw["case_ids"].dtype.kind == "U"
    with pytest.raises(ValueError, match="not overwritten"):
        diagnostic.main(args)


def test_plot_uses_fixed_scale_and_processed_base(manifest, tmp_path, monkeypatch):
    import matplotlib.axes
    actual = matplotlib.axes.Axes.imshow
    calls = []
    def record(self, data, *args, **kwargs):
        calls.append((np.asarray(data), kwargs.copy()))
        return actual(self, data, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", record)
    case_id = manifest.iloc[0].case_id
    base = np.full((224, 224, 3), .3, dtype=np.float32)
    diagnostic.render_pair_figures(
        tmp_path / "figures", {"eq_off": np.full((1, 7, 7), .98, dtype=np.float32),
                              "eq_on": np.full((1, 7, 7), .99, dtype=np.float32)},
        [case_id], {case_id: base}, manifest, 42)
    images = [call for call in calls if call[0].ndim == 2]
    assert len(images) == 4
    assert all(options["vmin"] == 0 and options["vmax"] == 1 for _, options in images)
    assert images[0][0].min() == pytest.approx(.98)
    assert images[2][0].min() == pytest.approx(.99)
    assert np.array_equal(calls[0][0], base)
    assert (tmp_path / "figures" / f"{case_id}.png").is_file()
