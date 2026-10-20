"""Fixed-weight attention interventions at the previously specified validation rotations.

The seven angles and preprocessing match evaluate_rotation_pairs. Every attention
intervention is compared with the same model's unmodified output at the SAME angle.
No training, test-image access, or search for more favorable rotation angles occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision

from scripts import diagnose_attention as attention_diagnostic
from scripts.evaluate_glare_pairs import sha256_file, validate_manifest
from scripts.evaluate_rotation_pairs import DEFAULT_ANGLES, rotate_inputs
from src.data import IMAGENET_MEAN, get_eval_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = attention_diagnostic.MODELS
CLASSES = attention_diagnostic.CLASSES
INTERVENTIONS = attention_diagnostic.INTERVENTIONS
ANGLES = tuple(DEFAULT_ANGLES)
SCRIPT_VERSION = "1.0.0"


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
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.shuffles < 1 or args.bootstrap < 0 or args.seed < 0:
        parser.error("batch-size/shuffles must be positive; bootstrap/seed must be nonnegative")
    return args


def load_run_provenance(run_dirs, split_path, manifest):
    """Keep the strict controlled-pair checks and also lock the rotation operator."""
    runs = attention_diagnostic.load_run_provenance(run_dirs, split_path, manifest)
    source = "src/regularization.py"
    actual = sha256_file(PROJECT_ROOT / source)
    for run in runs:
        for name in MODELS:
            if run["provenance"][name]["config"]["source_sha256"].get(source) != actual:
                raise ValueError(f"Rotation source differs from training provenance: {source}")
    return runs


def collect_pair(models, manifest, training_seed, shuffles=20, seed=20261020,
                 batch_size=16, device="cpu"):
    if set(models) != set(MODELS) or shuffles < 1 or batch_size < 1:
        raise ValueError("Provide both models and positive shuffles/batch size")
    for model in models.values():
        model.to(device).eval()
    before = {name: attention_diagnostic.state_fingerprint(model)
              for name, model in models.items()}
    transform = get_eval_transforms(crop_border=True)
    ordered = manifest.sort_values("case_id").reset_index(drop=True)
    predictions, statistics = [], []
    native = {name: [] for name in MODELS}
    errors = {name: dict(logits_max_abs_error=0., attention_max_abs_error=0.) for name in MODELS}
    forward_batches = {name: {str(angle): 0 for angle in ANGLES} for name in MODELS}
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            cases = list(ordered.iloc[start:start + batch_size].itertuples(index=False))
            tensors = []
            for case in cases:
                with Image.open(case.path) as opened:
                    tensors.append(transform(opened.convert("RGB")))
            inputs = torch.stack(tensors).to(device)
            batch_native = {name: [] for name in MODELS}
            for angle in ANGLES:
                # The existing helper returns the SAME tensor for zero degrees.
                # That angle is evaluated once and reused for every intervention.
                rotated = rotate_inputs(inputs, angle)
                for name in MODELS:
                    features, global_features, attention, logits, error = (
                        attention_diagnostic.cached_forward(models[name], rotated))
                    for key, value in error.items():
                        errors[name][key] = max(errors[name][key], value)
                    forward_batches[name][str(angle)] += 1
                    batch_native[name].append(attention[:, 0].cpu().numpy())
                    angle_statistics = attention_diagnostic.attention_statistics(
                        cases, training_seed, name, attention)
                    statistics.extend(dict(row, angle_deg=float(angle)) for row in angle_statistics)
                    original = logits.softmax(1).cpu().numpy()
                    for kind, repeat, changed_attention in attention_diagnostic.intervention_maps(
                            attention, [case.case_id for case in cases], shuffles, seed):
                        changed = original if kind == "original" else (
                            attention_diagnostic.intervention_logits(
                                models[name], features, global_features, changed_attention)
                            .softmax(1).cpu().numpy())
                        difference = (changed_attention - attention).abs().flatten(1).mean(1).cpu().numpy()
                        rows = attention_diagnostic.prediction_rows(
                            cases, training_seed, name, kind, repeat, original, changed, difference)
                        predictions.extend(dict(row, angle_deg=float(angle)) for row in rows)
            for name in MODELS:
                # [case, angle, native_y, native_x], with no enlargement or normalization.
                native[name].append(np.stack(batch_native[name], axis=1))
            print(f"Rotation attention seed={training_seed}: "
                  f"{min(start + batch_size, len(ordered))}/{len(ordered)} images, "
                  f"{len(ANGLES)} fixed angles", flush=True)
    after = {name: attention_diagnostic.state_fingerprint(model)
             for name, model in models.items()}
    if before != after:
        raise RuntimeError("Model parameters or buffers changed during inference")
    verification = dict(
        state_sha256_before=before, state_sha256_after=after, state_unchanged=True,
        forward_equivalence=errors, verified_forward_batches_per_angle=forward_batches,
        forward_tolerance=dict(rtol=attention_diagnostic.FORWARD_RTOL,
                               atol=attention_diagnostic.FORWARD_ATOL))
    predictions = pd.DataFrame(predictions).sort_values(
        ["seed", "model", "case_id", "angle_deg", "intervention", "repeat"]).reset_index(drop=True)
    statistics = pd.DataFrame(statistics).sort_values(
        ["seed", "model", "case_id", "angle_deg"]).reset_index(drop=True)
    return (predictions, statistics, {name: np.concatenate(values) for name, values in native.items()},
            ordered.case_id.to_numpy(dtype=str), verification)


def compare_historical_original(predictions, run_directory):
    """Audit optional prior outputs; never substitute them for this run's predictions."""
    path = run_directory / "rotation_val" / "per_image_predictions.csv"
    if not path.is_file():
        return dict(status="missing", path=str(path),
                    explanation="This diagnostic recomputed all original predictions.")
    keys = ["model", "case_id", "angle_deg"]
    identity = ["patient_id", "image_name", "true_label", "true_label_id"]
    probs = [f"prob_{label}" for label in CLASSES]
    historical = pd.read_csv(path, dtype={"patient_id": str, "case_id": str, "image_name": str})
    missing = set(keys + identity + probs) - set(historical.columns)
    if missing:
        raise ValueError(f"Historical rotation CSV lacks columns: {sorted(missing)}")
    original = predictions.loc[predictions.intervention.eq("original")].copy()
    if historical.duplicated(keys).any() or original.duplicated(keys).any():
        raise ValueError("Historical comparison requires exactly one row per model/case/angle")
    historical = historical.set_index(keys).sort_index()
    original = original.set_index(keys).sort_index()
    if not original.index.equals(historical.index):
        raise ValueError("Historical rotation CSV has a different model/case/angle grid")
    for column in identity:
        if not original[column].equals(historical[column]):
            raise ValueError(f"Historical rotation CSV has unmatched identity: {column}")
    old, new = historical[probs].to_numpy(float), original[probs].to_numpy(float)
    if (not np.isfinite(old).all() or (old < 0).any() or (old > 1).any()
            or not np.allclose(old.sum(1), 1., rtol=0, atol=1e-5)):
        raise ValueError("Historical rotation CSV has invalid probabilities")
    old_labels, new_labels = old.argmax(1), new.argmax(1)
    disagreement = old_labels != new_labels
    if disagreement.any():
        print(f"WARNING: {int(disagreement.sum())} original predictions differ from {path}; "
              "all summaries use this run's recomputed original predictions.", flush=True)
    by_model = {}
    nonzero = original.index.get_level_values("angle_deg") != 0
    truth = original.true_label_id.to_numpy(int)
    for name in MODELS:
        mask = nonzero & (original.index.get_level_values("model") == name)
        by_model[name] = dict(
            historical_nonzero_accuracy=float((old_labels[mask] == truth[mask]).mean()),
            current_nonzero_accuracy=float((new_labels[mask] == truth[mask]).mean()))
    return dict(status="compared", path=str(path), sha256=sha256_file(path), n_rows=len(original),
                n_label_disagreements=int(disagreement.sum()), labels_identical=not bool(disagreement.any()),
                probability_max_abs_difference=float(np.abs(old - new).max()),
                probability_mean_abs_difference=float(np.abs(old - new).mean()), by_model=by_model,
                explanation="GPU kernels/batch size can change probabilities; historical values are not substituted.")


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
    output.mkdir(parents=True, exist_ok=True)
    all_predictions, all_statistics, run_metadata = [], [], []
    for run in runs:
        directory = output / f"seed_{run['seed']}"
        directory.mkdir()
        models = attention_diagnostic.load_models(run, device)
        predictions, statistics, maps, case_ids, verification = collect_pair(
            models, manifest, run["seed"], args.shuffles, args.seed, args.batch_size, device)
        historical = compare_historical_original(predictions, run["directory"])
        predictions.to_csv(directory / "per_image_predictions.csv", index=False)
        statistics.to_csv(directory / "attention_stats.csv", index=False)
        raw_dir = directory / "raw_maps"
        raw_dir.mkdir()
        for name in MODELS:
            np.savez_compressed(raw_dir / f"{name}.npz", maps=maps[name].astype(np.float32),
                                case_ids=case_ids.astype(str), angles_deg=np.asarray(ANGLES))
        details = dict(
            seed=run["seed"], training_directory=str(run["directory"]),
            checkpoint_sha256={name: sha256_file(path) for name, path in run["checkpoints"].items()},
            training_provenance=run["provenance"], inference_verification=verification,
            historical_original_comparison=historical)
        attention_diagnostic._write_json(directory / "metadata.json", details)
        run_metadata.append(details)
        all_predictions.append(predictions)
        all_statistics.append(statistics)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / "per_image_predictions.csv", index=False)
    pd.concat(all_statistics, ignore_index=True).to_csv(output / "attention_stats.csv", index=False)
    from scripts.rotation_attention_stats import write_outputs
    write_outputs(predictions, output, n_bootstrap=args.bootstrap, seed=args.seed)
    sources = (
        "scripts/diagnose_rotation_attention.py", "scripts/rotation_attention_stats.py",
        "scripts/attention_intervention_stats.py",
        "scripts/diagnose_attention.py", "scripts/evaluate_rotation_pairs.py",
        "scripts/evaluate_glare_pairs.py", "src/attention.py", "src/data.py",
        "src/regularization.py", "src/viz.py")
    metadata = dict(
        completed=True, script_version=SCRIPT_VERSION,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments={key: [str(item) for item in value] if isinstance(value, list) else
                   str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source_sha256={name: sha256_file(PROJECT_ROOT / name) for name in sources},
        split_sha256=sha256_file(split_path), split="val", test_images_loaded=False,
        n_images=len(manifest), n_patients=int(manifest.patient_id.nunique()),
        n_prediction_rows=len(predictions), training_seeds=[run["seed"] for run in runs],
        class_order=CLASSES, runs=run_metadata, device=str(device),
        preprocessing="CropBlackBorder, Resize(256), CenterCrop(224), ImageNet normalization; same tensors for both arms",
        rotation=dict(
            angles_deg=ANGLES, primary_angles_deg=[angle for angle in ANGLES if angle],
            operator="evaluate_rotation_pairs.rotate_inputs -> src.regularization._rotate",
            stage="After ImageNet normalization", padding="Normalized zero, equivalent to ImageNet mean RGB",
            padding_rgb=IMAGENET_MEAN, interpolation="Bilinear grid_sample, align_corners=False",
            zero="Exact input identity; one cached output per model/batch reused by all interventions"),
        intervention=dict(
            modes=INTERVENTIONS,
            original="This run's original attention and prediction at the SAME angle",
            mean_constant="The current rotated image's native attention mean broadcast over all 49 cells",
            spatial_shuffle="Permute all 49 native cells, preserving histogram; same case/repeat permutation across angles/seeds/arms",
            ones_constant="All ones; attended feature equals global feature, changing amplitude as well as location",
            shuffles=args.shuffles, random_seed=args.seed,
            permutation_seed="SHA256([evaluation_seed, case_id, repeat]) first 64 bits; numpy default_rng",
            fixed="Each rotated input's backbone features, global branch, head, weights and BN buffers; eval + inference_mode"),
        attention=dict(native_shape=[7, 7], raw_dtype="float32",
                       npz_axes=["case", "angle", "native_y", "native_x"],
                       npz_keys=["maps", "case_ids", "angles_deg"], npz_case_ids="Unicode; allow_pickle=False"),
        statistics=dict(
            primary="Unconditional accuracy at six nonzero angles; EQ-on minus EQ-off for each intervention",
            gain_change="Intervention EQ gain minus original-attention EQ gain, using paired observations",
            aggregation="Average shuffle repeats within case/angle, then the six nonzero angles within case; "
                        "compute each seed's image-weighted metric, then average seeds equally",
            bootstrap=dict(method="Paired patient-cluster percentile bootstrap", replicates=args.bootstrap,
                           seed=args.seed, confidence_level=.95,
                           shared_draws="Same patient multiplicities across all models, modes, angles and training seeds",
                           conditioning="Intervals condition on these fitted model pairs; seeds and angles are not new patients")),
        original_reference="Original probability/flip columns refer to the same angle, never the zero-degree image",
        historical_reference="Optional rotation_val CSV audited for identity/grid/probabilities; never used in current statistics",
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__),
        limitations=[
            "Fixed-weight validation diagnostic; no retraining, test images, or new independent patients.",
            "The primary outcome is EQ-on minus EQ-off accuracy over the six nonzero angles, within each intervention.",
            "Gain-change is an inference-time intervention interaction, not causal mediation of the training loss.",
            "Both arms retain their separately trained weights and active global branch.",
            "The absence of gain change does not exclude training-time attention effects.",
            "Sensitivity to attention interventions does not demonstrate anatomically correct localization.",
            "Shuffles, angles and training seeds are paired repetitions, not independent patient samples.",
        ])
    attention_diagnostic._write_json(output / "metadata.json", metadata)
    print(f"Completed: {output} ({len(predictions)} prediction rows)", flush=True)


if __name__ == "__main__":
    main()
