"""Fixed ImageNet features and support-only classifiers for the renew screen.

This is a descriptive baseline, not an adaptive patient-selection algorithm.
No classifier function accepts an outer evaluation set during fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
import warnings

import numpy as np

CLASSIFIERS = ("prototype", "linear")
FEATURE_DIM = 512
MODEL_VERSION = "renew_frozen_head_v1"
TUNING_FIELDS = ("C", "heldout_patient", "train_patients", "n_train", "n_val",
                 "val_log_loss", "n_iter", "mean_log_loss", "selected")


def classifier_config():
    """Return a fresh copy of the fixed protocol, never an adjustable grid."""
    return {"feature_dim": FEATURE_DIM, "normalization": "per-image L2; no fitted transform",
            "prototype": {"distance": "cosine", "normalize_class_mean": True,
                          "probability": "softmax cosine", "temperature": 1.0},
            "linear": {"C": [0.1, 1.0, 10.0], "solver": "lbfgs", "penalty": "l2",
                       "fit_intercept": True, "tol": 1e-8, "max_iter": 10000,
                       "class_weight": None, "warm_start": False,
                       "selection": "leave-one-support-patient-out mean log loss",
                       "tie_absolute_tolerance": 1e-12, "tie_break": "smaller C",
                       "convergence": "fail on ConvergenceWarning or iteration limit"}}


def l2_normalize(raw):
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] != FEATURE_DIM:
        raise ValueError("Features must be a nonempty N x 512 matrix")
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Feature norms must be finite and nonzero")
    return np.ascontiguousarray(values / norms)


def _features(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or not len(values) or values.shape[1] != FEATURE_DIM
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-8, rtol=0)):
        raise ValueError("Expected finite, per-image L2-normalized N x 512 features")
    return np.ascontiguousarray(values)


def _support(features, labels, patient_keys):
    features = _features(features)
    labels = np.asarray(labels)
    patients = np.asarray(patient_keys, dtype=str)
    if (len(features) != 15 or labels.shape != (15,) or patients.shape != (15,)
            or labels.dtype.kind not in "iu" or set(labels.tolist()) != {0, 1, 2}
            or len(set(patients)) != 5 or any(not p for p in patients)):
        raise ValueError("Support requires five patients and fifteen integer-labelled images")
    for patient in sorted(set(patients)):
        if sorted(labels[patients == patient].tolist()) != [0, 1, 2]:
            raise ValueError("Every support patient must have one image per class")
    return features, labels.astype(np.int64), patients


def _softmax(scores):
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite classifier scores")
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _log_loss(labels, probabilities):
    return float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels],
                                   np.finfo(np.float64).tiny)).mean())


def _fit_linear(features, labels, c):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    config = classifier_config()["linear"]
    # With three classes and lbfgs, sklearn 1.3's default multi_class='auto'
    # and sklearn 1.8 (which removed multi_class) both fit multinomial loss.
    # Default penalty is L2 in both; avoid the penalty keyword deprecated in 1.8.
    estimator = LogisticRegression(C=c, solver=config["solver"],
        tol=config["tol"], max_iter=config["max_iter"], fit_intercept=True,
        class_weight=None, warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(features, labels)
        except ConvergenceWarning as error:
            raise ValueError("Linear classifier failed to converge; no result recorded") from error
    if (not np.array_equal(estimator.classes_, [0, 1, 2])
            or np.any(estimator.n_iter_ >= config["max_iter"])
            or not np.isfinite(estimator.coef_).all()
            or not np.isfinite(estimator.intercept_).all()):
        raise ValueError("Linear classifier has invalid classes, parameters or convergence")
    return estimator


def _select_c(scores):
    minimum = min(row["mean_log_loss"] for row in scores)
    tolerance = classifier_config()["linear"]["tie_absolute_tolerance"]
    return min(row["C"] for row in scores if row["mean_log_loss"] <= minimum + tolerance)


def fit_classifier(method, support_features, labels, patient_keys):
    """Fit on 15 selected images only, returning a JSON model and inner-CV rows."""
    if method not in CLASSIFIERS:
        raise ValueError("Unknown classifier")
    features, labels, patients = _support(support_features, labels, patient_keys)
    model = {"version": MODEL_VERSION, "method": method, "classes": [0, 1, 2],
             "feature_dim": FEATURE_DIM, "config": classifier_config()}
    if method == "prototype":
        means = np.stack([features[labels == label].mean(axis=0) for label in range(3)])
        model["prototypes"] = l2_normalize(means).tolist()
        return model, []

    tuning, scores = [], []
    for c in classifier_config()["linear"]["C"]:
        candidate = []
        for patient in sorted(set(patients)):
            heldout = patients == patient
            estimator = _fit_linear(features[~heldout], labels[~heldout], c)
            loss = _log_loss(labels[heldout], estimator.predict_proba(features[heldout]))
            row = {"C": c, "heldout_patient": str(patient),
                   "train_patients": json.dumps(sorted(set(patients[~heldout]))),
                   "n_train": int((~heldout).sum()), "n_val": int(heldout.sum()),
                   "val_log_loss": loss, "n_iter": int(estimator.n_iter_.max())}
            candidate.append(row)
        mean = sum(row["val_log_loss"] * row["n_val"] for row in candidate) / len(features)
        scores.append({"C": c, "mean_log_loss": float(mean)})
        for row in candidate:
            row["mean_log_loss"] = float(mean)
        tuning.extend(candidate)
    chosen = _select_c(scores)
    for row in tuning:
        row["selected"] = row["C"] == chosen
    estimator = _fit_linear(features, labels, chosen)
    model.update(coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist(),
                 selected_C=chosen, cv_scores=scores, n_iter=int(estimator.n_iter_.max()))
    if not np.allclose(predict_classifier(model, features), estimator.predict_proba(features),
                       rtol=0, atol=1e-12):
        raise ValueError("Exported linear classifier does not reproduce fitted probabilities")
    return model, tuning


def predict_classifier(model, features):
    features = _features(features)
    if (model.get("version") != MODEL_VERSION or model.get("classes") != [0, 1, 2]
            or model.get("feature_dim") != FEATURE_DIM
            or model.get("config") != classifier_config()):
        raise ValueError("Classifier identity/configuration changed")
    if model.get("method") == "prototype":
        prototypes = _features(model["prototypes"])
        if prototypes.shape != (3, FEATURE_DIM):
            raise ValueError("Expected exactly three prototypes")
        scores = features @ prototypes.T
    elif model.get("method") == "linear":
        coef, intercept = np.asarray(model["coef"]), np.asarray(model["intercept"])
        if (coef.shape != (3, FEATURE_DIM) or intercept.shape != (3,)
                or model.get("selected_C") not in classifier_config()["linear"]["C"]):
            raise ValueError("Invalid linear classifier export")
        scores = features @ coef.T + intercept
    else:
        raise ValueError("Unknown exported classifier")
    return _softmax(scores)


def _extract_with_model(rows, model, device):
    """Internal extraction kernel; testable with a synthetic model and images."""
    import torch
    from PIL import Image
    from src.fewshot import sha256_file, state_fingerprint
    from src.fewshot_finetune import _digest, bn_fingerprint
    from src.renew_data import get_transforms
    from src.renew_experiment import configure_torch

    if not rows or len({r["image_name"] for r in rows}) != len(rows):
        raise ValueError("Extraction requires nonempty unique image rows")
    configure_torch()
    model.to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    state_before, bn_before = state_fingerprint(model), bn_fingerprint(model)
    transform = get_transforms("full_frame", train=False)
    chunks = []
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            tensors = []
            for row in rows[offset:offset + 16]:
                if sha256_file(row["path"]) != row["image_sha256"]:
                    raise ValueError("Image changed before feature extraction: " + row["image_name"])
                with Image.open(row["path"]) as image:
                    tensors.append(transform(image.convert("RGB")))
                if sha256_file(row["path"]) != row["image_sha256"]:
                    raise ValueError("Image changed during feature extraction: " + row["image_name"])
            chunks.append(model(torch.stack(tensors).to(device)).detach().cpu().numpy())
    raw = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if raw.shape != (len(rows), FEATURE_DIM):
        raise ValueError("Encoder did not return one 512-dimensional feature per image")
    l2_normalize(raw)  # Also rejects nonfinite / zero encodings before caching.
    state_after, bn_after = state_fingerprint(model), bn_fingerprint(model)
    frozen = all(not p.requires_grad for p in model.parameters())
    is_eval = all(not module.training for module in model.modules())
    if state_before != state_after or bn_before != bn_after or not frozen or not is_eval:
        raise ValueError("Frozen encoder state or evaluation mode changed during extraction")
    evidence = {"architecture": "resnet18", "preprocessing": "full_frame_eval",
        "feature_layer": "avgpool_flatten", "feature_dim": FEATURE_DIM,
        "batch_size": 16, "num_workers": 0, "device": str(device), "precision": "float32",
        "encoder_eval": is_eval, "all_parameters_frozen": frozen,
        "bn_before_sha256": bn_before, "bn_after_sha256": bn_after,
        "state_before_sha256": state_before, "state_after_sha256": state_after,
        "row_order_sha256": _digest([r["image_name"] for r in rows]), "images": len(rows),
        "raw_feature_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started}
    return raw, evidence


def _verified_weight_hash(path, url):
    from src.fewshot import sha256_file
    expected = re.search(r"-([a-f0-9]{8,64})\.pth$", urlparse(url).path)
    actual = sha256_file(path)
    if expected is None or not actual.startswith(expected.group(1)):
        raise ValueError("Cached ImageNet weights do not match the official URL hash")
    return actual


def extract_features(rows, device="cuda"):
    """Load fixed public ImageNet weights and independently encode each image."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    from src.renew_experiment import configure_torch

    if device != "cpu" and not str(device).startswith("cuda"):
        raise ValueError("Only explicitly selected CPU or CUDA extraction is supported")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent CPU fallback")
    configure_torch()
    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    # torchvision checks newly downloaded files, but a pre-existing cache also
    # needs verification before its contents are deserialized into the encoder.
    if weight_file.exists():
        _verified_weight_hash(weight_file, weights.url)
    model = resnet18(weights=weights)
    weight_sha = _verified_weight_hash(weight_file, weights.url)
    model.fc = torch.nn.Identity()
    raw, evidence = _extract_with_model(rows, model, device)
    evidence.update(weights="ResNet18_Weights.IMAGENET1K_V1", weights_url=weights.url,
                    weights_file_sha256=weight_sha)
    return raw, evidence
