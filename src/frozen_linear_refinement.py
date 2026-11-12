"""One fixed, wider regularization grid on the unchanged v1 frozen features.

The support-only selection rule and numerical solver are reused from v1.
There is no encoder, feature extraction, or outer-evaluation input here.
"""
from __future__ import annotations

import json
import math

import numpy as np

from src.frozen_screen import (
    FEATURE_DIM, TUNING_FIELDS, _features, _fit_linear, _log_loss, _select_c,
    _softmax, _support, classifier_config as v1_classifier_config,
)

MODEL_VERSION = "renew_frozen_linear_refinement_v2"
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def classifier_config():
    """Return a fresh configuration; the sole v1 classifier change is C."""
    config = v1_classifier_config()
    config["linear"]["C"] = list(C_GRID)
    return config


def validate_model(model):
    """Validate a JSON export even when real feature arrays are unavailable."""
    if (model.get("version") != MODEL_VERSION or model.get("method") != "linear"
            or model.get("classes") != [0, 1, 2]
            or model.get("feature_dim") != FEATURE_DIM
            or model.get("config") != classifier_config()):
        raise ValueError("Refinement classifier identity/configuration changed")
    coef = np.asarray(model.get("coef"), dtype=np.float64)
    intercept = np.asarray(model.get("intercept"), dtype=np.float64)
    if (coef.shape != (3, FEATURE_DIM) or intercept.shape != (3,)
            or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
        raise ValueError("Invalid refinement classifier parameters")
    selected = model.get("selected_C")
    scores = model.get("cv_scores")
    if (type(selected) not in (int, float) or selected not in C_GRID
            or not isinstance(scores, list) or len(scores) != len(C_GRID)
            or [row.get("C") for row in scores] != list(C_GRID)):
        raise ValueError("Invalid refinement candidate or CV evidence")
    if any(type(row.get("mean_log_loss")) not in (int, float)
           or not math.isfinite(row["mean_log_loss"]) or row["mean_log_loss"] < 0
           for row in scores):
        raise ValueError("Invalid refinement validation loss")
    if selected != _select_c(scores):
        raise ValueError("Refinement C differs from support-only selection")
    if type(model.get("n_iter")) is not int or not 0 <= model["n_iter"] < classifier_config()["linear"]["max_iter"]:
        raise ValueError("Invalid refinement convergence evidence")


def predict_classifier(model, features):
    features = _features(features)
    validate_model(model)
    return _softmax(features @ np.asarray(model["coef"], dtype=np.float64).T
                    + np.asarray(model["intercept"], dtype=np.float64))


def fit_classifier(support_features, labels, patient_keys):
    """Fit 25 inner models, select on support patients, then refit all 15 rows."""
    features, labels, patients = _support(support_features, labels, patient_keys)
    tuning, scores = [], []
    for c in C_GRID:
        candidate = []
        for patient in sorted(set(patients)):
            heldout = patients == patient
            estimator = _fit_linear(features[~heldout], labels[~heldout], c)
            candidate.append({"C": c, "heldout_patient": str(patient),
                "train_patients": json.dumps(sorted(set(patients[~heldout]))),
                "n_train": int((~heldout).sum()), "n_val": int(heldout.sum()),
                "val_log_loss": _log_loss(labels[heldout], estimator.predict_proba(features[heldout])),
                "n_iter": int(estimator.n_iter_.max())})
        mean = sum(row["val_log_loss"] * row["n_val"] for row in candidate) / len(features)
        scores.append({"C": c, "mean_log_loss": float(mean)})
        for row in candidate:
            row["mean_log_loss"] = float(mean)
        tuning.extend(candidate)
    chosen = _select_c(scores)
    for row in tuning:
        row["selected"] = row["C"] == chosen
    estimator = _fit_linear(features, labels, chosen)
    model = {"version": MODEL_VERSION, "method": "linear", "classes": [0, 1, 2],
        "feature_dim": FEATURE_DIM, "config": classifier_config(),
        "coef": estimator.coef_.tolist(), "intercept": estimator.intercept_.tolist(),
        "selected_C": chosen, "cv_scores": scores, "n_iter": int(estimator.n_iter_.max())}
    if not np.allclose(predict_classifier(model, features), estimator.predict_proba(features),
                       rtol=0, atol=1e-12):
        raise ValueError("Exported refinement classifier does not reproduce fitted probabilities")
    return model, tuning
