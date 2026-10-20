"""Validation-only, fixed-weight attention maps and spatial-intervention diagnostic.

No training, image rotations, threshold fitting, or test-image access is performed.
The internal attention is not a class-specific Grad-CAM explanation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airway_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "airway_cache"))

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision

from scripts.evaluate_glare_pairs import sha256_file, stable_seed, validate_manifest
from scripts.evaluate_rotation_pairs import _attention_distribution, _training_provenance
from src.attention import build_attention_resnet50
from src.data import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("eq_off", "eq_on")
CLASSES = ("lt", "yz", "zz")
INTERVENTIONS = ("original", "mean_constant", "spatial_shuffle", "ones_constant")
SCRIPT_VERSION = "1.0.0"
FORWARD_RTOL = 1e-5
FORWARD_ATOL = 1e-6


def state_fingerprint(model):
    """Content hash for parameters and buffers, using the training hash format."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits_patent")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--render-per-class", type=int, default=3)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.shuffles < 1 or args.bootstrap < 0:
        parser.error("batch-size and shuffles must be positive; bootstrap must be nonnegative")
    if args.seed < 0 or args.render_per_class < 0:
        parser.error("seed and render-per-class must be nonnegative")
    return args


def load_run_provenance(run_dirs, split_path, manifest):
    """Require the completed controlled experiment, not an unverified state dict."""
    actual_members = manifest.assign(image_name=manifest.path.map(lambda p: Path(p).name))[
        ["patient_id", "label", "label_id", "image_name"]
    ].sort_values(["patient_id", "label", "image_name"]).to_dict("records")
    runs, seen_seeds = [], set()
    for directory in run_dirs:
        directory = directory.expanduser().resolve()
        required = [directory / "pair_metadata.json"]
        required += [directory / model / name for model in MODELS
                     for name in ("run_config.json", "last_model.pt")]
        if any(not path.is_file() for path in required):
            raise ValueError(f"Controlled run requires both run configs, last checkpoints and pair metadata: {directory}")
        configs = {model: json.loads((directory / model / "run_config.json").read_text())
                   for model in MODELS}
        pair_content = json.loads((directory / "pair_metadata.json").read_text())
        if not isinstance(pair_content, dict) or pair_content.get("protocol_version") != "eq_sham_control_v1":
            raise ValueError("Pair metadata must use eq_sham_control_v1")
        for model, config in configs.items():
            if not isinstance(config, dict) or config.get("protocol_version") != "eq_sham_control_v1":
                raise ValueError("Only eq_sham_control_v1 training provenance is accepted")
            if config.get("arm") != model or config.get("primary_checkpoint") != "last_model.pt":
                raise ValueError("Run arm or primary checkpoint is inconsistent")
            if config.get("fixed_architecture") != "AnatomyAttentionResNet50_layer4_7x7":
                raise ValueError("Only the fixed layer4 7x7 architecture is supported")
            if config.get("crop_border") is not True or config.get("evaluation_split") != "val":
                raise ValueError("Expected cropped validation preprocessing")
            if config.get("splits", {}).get("val", {}).get("members") != actual_members:
                raise ValueError("Validation membership does not match training provenance")
            if not config.get("source_sha256") or not config.get("final_state_sha256"):
                raise ValueError("Training provenance must include source and final-state hashes")
            if any(not config.get(field) for field in
                   ("initial_state_sha256", "training_input_trace_sha256", "training_input_traces",
                    "completed_epochs")):
                raise ValueError("Controlled provenance requires initialization, input traces and completed epoch count")
            for filename in ("src/attention.py", "src/data.py", "src/viz.py"):
                if config["source_sha256"].get(filename) != sha256_file(PROJECT_ROOT / filename):
                    raise ValueError(f"Inference source differs from training provenance: {filename}")
        seed = configs["eq_off"].get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed != configs["eq_on"].get("seed"):
            raise ValueError("Both arms must have the same integer training seed")
        if seed in seen_seeds:
            raise ValueError(f"Duplicate training seed: {seed}")
        seen_seeds.add(seed)
        dropout = configs["eq_off"].get("arguments", {}).get("dropout")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("A valid training dropout value is required")
        checkpoints = {model: directory / model / "last_model.pt" for model in MODELS}
        provenance = _training_provenance(checkpoints, dropout, split_path)
        pair = provenance["pair_metadata"][0]["content"]
        if pair.get("seed") != seed or pair.get("primary_checkpoint") != "last_model.pt":
            raise ValueError("Pair metadata seed or primary checkpoint is inconsistent")
        runs.append(dict(seed=seed, directory=directory, checkpoints=checkpoints,
                         dropout=dropout, provenance=provenance))
    # The earlier two invocations deliberately differed only in seed list/output root.
    shared_fields = ("protocol_version", "splits", "source_sha256", "fixed_architecture",
                     "crop_border", "pf_mask", "mask_size", "eq_auxiliary_forward",
                     "primary_checkpoint", "completed_epochs")
    reference = runs[0]["provenance"]["eq_off"]["config"]
    reference_args = {key: value for key, value in reference["arguments"].items()
                      if key not in ("seeds", "output_root")}
    for run in runs[1:]:
        config = run["provenance"]["eq_off"]["config"]
        for field in shared_fields:
            if config.get(field) != reference.get(field):
                raise ValueError(f"Training runs differ in shared field: {field}")
        arguments = {key: value for key, value in config["arguments"].items()
                     if key not in ("seeds", "output_root")}
        if arguments != reference_args:
            raise ValueError("Training runs differ in arguments beyond seeds/output_root")
    return sorted(runs, key=lambda run: run["seed"])


def load_models(run, device):
    models = {}
    for name in MODELS:
        model = build_attention_resnet50(pretrained=False, dropout=run["dropout"], hires=False)
        model.load_state_dict(torch.load(run["checkpoints"][name], map_location="cpu",
                                        weights_only=True), strict=True)
        expected = run["provenance"][name]["config"]["final_state_sha256"]
        if state_fingerprint(model) != expected:
            raise ValueError(f"Loaded state content differs from training provenance: {name}")
        models[name] = model.to(device).eval()
    return models


def shuffled_attention(attention, case_ids, repeat, seed):
    """Same native-cell permutation for every training seed/arm for each case."""
    flat = attention.flatten(1)
    if len(case_ids) != len(flat):
        raise ValueError("Case identities must match the attention batch")
    indices = np.stack([np.random.default_rng(stable_seed(seed, case, repeat)).permutation(flat.size(1))
                        for case in case_ids])
    index_tensor = torch.as_tensor(indices, device=flat.device, dtype=torch.long)
    return flat.gather(1, index_tensor).reshape_as(attention)


def intervention_maps(attention, case_ids, shuffles, seed):
    yield "original", -1, attention
    yield "mean_constant", -1, attention.mean(dim=(-2, -1), keepdim=True).expand_as(attention)
    for repeat in range(shuffles):
        yield "spatial_shuffle", repeat, shuffled_attention(attention, case_ids, repeat, seed)
    yield "ones_constant", -1, torch.ones_like(attention)


def cached_forward(model, inputs):
    """Validate the decomposed forward against the unmodified model each batch."""
    if model.training or getattr(model, "hires", False):
        raise ValueError("A low-resolution model in eval mode is required")
    features = model.backbone(inputs)
    attention = model.attention(features)
    if attention.shape != (len(inputs), 1, 7, 7):
        raise ValueError("Expected the native (B,1,7,7) attention map")
    if not torch.isfinite(attention).all() or (attention < 0).any() or (attention > 1).any():
        raise ValueError("Attention values must be finite within [0,1]")
    global_features = model.pool(features).flatten(1)
    logits = intervention_logits(model, features, global_features, attention)
    reference_logits, reference_attention = model(inputs, return_attn=True)
    torch.testing.assert_close(logits, reference_logits, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    torch.testing.assert_close(attention, reference_attention, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    error = dict(logits_max_abs_error=float((logits - reference_logits).abs().max()),
                 attention_max_abs_error=float((attention - reference_attention).abs().max()))
    return features, global_features, attention, logits, error


def intervention_logits(model, features, global_features, attention):
    attended = model.pool(features * attention).flatten(1)
    logits = model.fc(torch.cat((global_features, attended), dim=1))
    if logits.shape != (len(features), 3) or not torch.isfinite(logits).all():
        raise ValueError("Expected three finite output logits")
    return logits


def prediction_rows(cases, training_seed, model, intervention, repeat, original, changed, attention_change):
    rows = []
    for i, case in enumerate(cases):
        truth = int(case.label_id)
        old_pred, new_pred = int(original[i].argmax()), int(changed[i].argmax())
        old_ok, new_ok = old_pred == truth, new_pred == truth
        row = dict(seed=int(training_seed), model=model, case_id=case.case_id,
                   patient_id=case.patient_id, image_name=Path(case.path).name,
                   true_label=case.label, true_label_id=truth, intervention=intervention, repeat=repeat,
                   original_pred_id=old_pred, intervened_pred_id=new_pred,
                   original_correct=int(old_ok), intervened_correct=int(new_ok),
                   flipped=int(old_pred != new_pred), correct_to_wrong=int(old_ok and not new_ok),
                   wrong_to_correct=int(not old_ok and new_ok),
                   true_probability_delta=float(changed[i, truth] - original[i, truth]),
                   original_pred_probability_drop=float(original[i, old_pred] - changed[i, old_pred]),
                   probability_l1_change=float(np.abs(changed[i] - original[i]).sum()),
                   attention_l1_change=float(attention_change[i]))
        for index, label in enumerate(CLASSES):
            row[f"original_prob_{label}"] = float(original[i, index])
            row[f"prob_{label}"] = float(changed[i, index])
        rows.append(row)
    return rows


def attention_statistics(cases, training_seed, model, attention):
    values = attention.detach().cpu().flatten(1).double()
    mean, std, entropy = _attention_distribution(attention.detach().cpu())
    rows = []
    for i, case in enumerate(cases):
        rows.append(dict(seed=int(training_seed), model=model, case_id=case.case_id,
                         patient_id=case.patient_id, image_name=Path(case.path).name,
                         true_label=case.label, mean=float(mean[i]), std=float(std[i]),
                         min=float(values[i].min()), max=float(values[i].max()),
                         p05=float(torch.quantile(values[i], .05)), p95=float(torch.quantile(values[i], .95)),
                         normalized_entropy=float(entropy[i])))
    return rows


def selected_case_ids(manifest, per_class):
    chosen = []
    for label in CLASSES:
        chosen.extend(manifest.loc[manifest.label.eq(label)].sort_values("case_id")
                      .head(per_class).case_id.tolist())
    return chosen


def collect_pair(models, manifest, training_seed, shuffles=20, seed=20261020, batch_size=16,
                 device="cpu", render_ids=()):
    if set(models) != set(MODELS) or shuffles < 1 or batch_size < 1:
        raise ValueError("Provide both models and positive shuffles/batch size")
    for model in models.values():
        model.to(device).eval()
    original_states = {name: state_fingerprint(model) for name, model in models.items()}
    transform = get_eval_transforms(crop_border=True)
    ordered = manifest.sort_values("case_id").reset_index(drop=True)
    rows, statistics, render_inputs = [], [], {}
    native = {name: [] for name in MODELS}
    errors = {name: dict(logits_max_abs_error=0., attention_max_abs_error=0.) for name in MODELS}
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            cases = list(ordered.iloc[start:start + batch_size].itertuples(index=False))
            tensors = []
            for case in cases:
                with Image.open(case.path) as opened:
                    tensor = transform(opened.convert("RGB"))
                tensors.append(tensor)
                if case.case_id in render_ids:
                    render_inputs[case.case_id] = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            inputs = torch.stack(tensors).to(device)
            for name in MODELS:
                features, global_features, attention, logits, error = cached_forward(models[name], inputs)
                for key, value in error.items():
                    errors[name][key] = max(errors[name][key], value)
                native[name].append(attention[:, 0].cpu().numpy())
                statistics.extend(attention_statistics(cases, training_seed, name, attention))
                original = logits.softmax(1).cpu().numpy()
                for kind, repeat, changed_attention in intervention_maps(
                        attention, [case.case_id for case in cases], shuffles, seed):
                    changed = original if kind == "original" else intervention_logits(
                        models[name], features, global_features, changed_attention).softmax(1).cpu().numpy()
                    difference = (changed_attention - attention).abs().flatten(1).mean(1).cpu().numpy()
                    rows.extend(prediction_rows(cases, training_seed, name, kind, repeat,
                                                original, changed, difference))
            print(f"Attention diagnostic seed={training_seed}: "
                  f"{min(start + batch_size, len(ordered))}/{len(ordered)} images", flush=True)
    final_states = {name: state_fingerprint(model) for name, model in models.items()}
    if final_states != original_states:
        raise RuntimeError("Model parameters or buffers changed during inference")
    verification = dict(state_sha256_before=original_states, state_sha256_after=final_states,
                        state_unchanged=True, forward_equivalence=errors,
                        forward_tolerance=dict(rtol=FORWARD_RTOL, atol=FORWARD_ATOL))
    return (pd.DataFrame(rows).sort_values(["seed", "model", "case_id", "intervention", "repeat"])
            .reset_index(drop=True), pd.DataFrame(statistics),
            {name: np.concatenate(values) for name, values in native.items()},
            ordered.case_id.to_numpy(dtype=str), render_inputs, verification)


def render_pair_figures(output_dir, maps, case_ids, render_inputs, manifest, training_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {case_id: i for i, case_id in enumerate(case_ids)}
    labels = manifest.set_index("case_id").label.to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id, base in render_inputs.items():
        index = lookup[case_id]
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), constrained_layout=True)
        axes[0].imshow(base)
        axes[0].set_title("Actual cropped input")
        heat = None
        for offset, name in ((1, "eq_off"), (3, "eq_on")):
            raw = maps[name][index]
            heat = axes[offset].imshow(raw, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            axes[offset].set_title(f"{name}: native 7 x 7\nmean={raw.mean():.4f}, std={raw.std():.4f}")
            up = F.interpolate(torch.from_numpy(raw)[None, None], size=(224, 224),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
            axes[offset + 1].imshow(base)
            axes[offset + 1].imshow(up, cmap="viridis", vmin=0, vmax=1, alpha=.45,
                                    interpolation="nearest")
            axes[offset + 1].set_title(f"{name}: overlay")
        for axis in axes:
            axis.axis("off")
        fig.colorbar(heat, ax=axes.tolist(), shrink=.75, label="Raw attention [0, 1]")
        fig.suptitle(f"seed={training_seed}, label={labels[case_id]}, case={case_id}\n"
                     "Fixed color scale; overlay is bilinear enlargement, not higher spatial resolution",
                     fontsize=10)
        fig.savefig(output_dir / f"{case_id}.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output-dir must be absent or empty; existing results are not overwritten")
    split_path = args.splits_dir / "val.csv"
    manifest = validate_manifest(pd.read_csv(split_path, dtype=str), args.image_root)
    runs = load_run_provenance(args.run_dirs, split_path, manifest)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; activate the server environment or use --device cpu")
        torch.cuda.get_device_properties(device)
    render_ids = selected_case_ids(manifest, args.render_per_class)
    output.mkdir(parents=True, exist_ok=True)
    all_predictions, all_statistics, run_metadata = [], [], []
    for run in runs:
        directory = output / f"seed_{run['seed']}"
        directory.mkdir()
        models = load_models(run, device)
        predictions, stats, maps, case_ids, render_inputs, verification = collect_pair(
            models, manifest, run["seed"], args.shuffles, args.seed, args.batch_size, device, render_ids)
        predictions.to_csv(directory / "per_image_predictions.csv", index=False)
        stats.to_csv(directory / "attention_stats.csv", index=False)
        raw_dir = directory / "raw_maps"
        raw_dir.mkdir()
        for name in MODELS:
            np.savez_compressed(raw_dir / f"{name}.npz", maps=maps[name].astype(np.float32),
                                case_ids=case_ids.astype(str))
        if render_inputs:
            render_pair_figures(directory / "figures", maps, case_ids, render_inputs, manifest, run["seed"])
        details = dict(seed=run["seed"], training_directory=str(run["directory"]),
                       checkpoint_sha256={name: sha256_file(path) for name, path in run["checkpoints"].items()},
                       training_provenance=run["provenance"], inference_verification=verification)
        _write_json(directory / "metadata.json", details)
        run_metadata.append(details)
        all_predictions.append(predictions)
        all_statistics.append(stats)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / "per_image_predictions.csv", index=False)
    pd.concat(all_statistics, ignore_index=True).to_csv(output / "attention_stats.csv", index=False)
    from scripts.attention_intervention_stats import write_outputs
    write_outputs(predictions, output, n_bootstrap=args.bootstrap, seed=args.seed)
    source_files = ("scripts/diagnose_attention.py", "scripts/attention_intervention_stats.py",
                    "scripts/evaluate_rotation_pairs.py", "scripts/evaluate_glare_pairs.py",
                    "src/attention.py", "src/data.py", "src/viz.py")
    metadata = dict(
        completed=True, script_version=SCRIPT_VERSION, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: [str(item) for item in value] if isinstance(value, list) else
                   str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source_sha256={name: sha256_file(PROJECT_ROOT / name) for name in source_files},
        split_sha256=sha256_file(split_path), split="val", test_images_loaded=False,
        n_images=len(manifest), n_patients=int(manifest.patient_id.nunique()),
        n_prediction_rows=len(predictions), training_seeds=[run["seed"] for run in runs],
        class_order=CLASSES, runs=run_metadata, device=str(device),
        preprocessing="CropBlackBorder, Resize(256), CenterCrop(224), ImageNet normalization; same tensors for both arms",
        intervention=dict(
            modes=INTERVENTIONS, original="Exact cached predictions from reconstructed verified forward",
            mean_constant="Per-image native attention mean broadcast over all 49 cells; removes spatial selection",
            spatial_shuffle="Permute all 49 native cells, preserving exact histogram; same permutation across seeds/arms",
            ones_constant="All ones; attended feature equals global feature, changing amplitude as well as location",
            shuffles=args.shuffles, random_seed=args.seed,
            permutation_seed="SHA256([evaluation_seed, case_id, repeat]) first 64 bits; numpy default_rng permutation",
            fixed="Backbone features, global branch, head, weights and BatchNorm buffers; eval + inference_mode"),
        attention=dict(native_shape=[7, 7], raw_dtype="float32", npz_case_ids="Unicode; allow_pickle=False",
                       normalized_entropy="Shannon entropy after dividing by total, divided by log(49); zero-total undefined",
                       attention_l1_change="Mean absolute native-cell change, not probability-vector L1"),
        visualization=dict(selected_case_ids=render_ids, per_class=args.render_per_class,
                           selection="First case_ids sorted within each class; identical across seeds/arms, independent of predictions",
                           colormap="viridis", vmin=0, vmax=1, native_interpolation="nearest",
                           overlay_interpolation="bilinear to 224, align_corners=False",
                           background="The actual normalized input after inverse ImageNet normalization",
                           minmax_normalization=False),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__),
        limitations=[
            "Fixed-weight validation diagnostic only; no new training or clinical effectiveness result.",
            "The global branch remains active. Unchanged labels alone do not prove the attention branch is unused.",
            "Interventions can leave the training distribution; sensitivity does not prove anatomically correct evidence.",
            "Raw internal attention is class-independent and not Grad-CAM.",
            "Repeated shuffles and training seeds do not increase independent patient count.",
            "Selected PNGs are illustrative only; all images have saved raw maps, predictions and statistics.",
        ])
    _write_json(output / "metadata.json", metadata)
    print(f"Completed: {output} ({len(predictions)} prediction rows)", flush=True)


if __name__ == "__main__":
    main()
