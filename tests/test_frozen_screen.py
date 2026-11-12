"""Synthetic-only checks for support isolation, exported heads and frozen state."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from src import frozen_screen as f


def support():
    rng = np.random.default_rng(19)
    labels = np.tile(np.arange(3), 5)
    patients = np.repeat([str(p) for p in range(5)], 3)
    values = rng.normal(0, .01, (15, 512))
    values[np.arange(15), labels] += 1
    # A unique coordinate encodes row identity for the fit/validation spy.
    values[np.arange(15), np.arange(15) + 10] = .2
    return f.l2_normalize(values), labels, patients


class FrozenHeadTests(unittest.TestCase):
    def test_normalization_and_invalid_features(self):
        values = np.arange(1, 1025).reshape(2, 512)
        result = f.l2_normalize(values)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-14)
        for invalid in (np.zeros((1, 512)), np.full((1, 512), np.nan),
                        np.ones((0, 512)), np.ones((2, 3)), np.ones(512)):
            with self.assertRaises(ValueError):
                f.l2_normalize(invalid)

    def test_prototype_matches_independent_formula_and_json_replay(self):
        features, labels, patients = support()
        model, tuning = f.fit_classifier("prototype", features, labels, patients)
        self.assertEqual(tuning, [])
        means = np.stack([features[labels == i].mean(axis=0) for i in range(3)])
        means /= np.linalg.norm(means, axis=1, keepdims=True)
        scores = features @ means.T
        expected = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        predicted = f.predict_classifier(json.loads(json.dumps(model)), features)
        np.testing.assert_allclose(predicted, expected, atol=1e-14)
        np.testing.assert_array_equal(predicted.argmax(axis=1), labels)

    def test_support_is_a_complete_five_patient_set(self):
        x, y, p = support()
        malformed = ((x[:-1], y[:-1], p[:-1]), (x, y.astype(float), p),
                     (x, np.zeros(15, dtype=int), p), (x, y, np.repeat("x", 15)),
                     (x * 2, y, p))
        wrong_labels = y.copy()
        wrong_labels[0] = 1
        malformed += ((x, wrong_labels, p),)
        for args in malformed:
            with self.assertRaises(ValueError):
                f.fit_classifier("prototype", *args)

    def test_linear_inner_folds_only_see_four_support_patients(self):
        from sklearn.linear_model import LogisticRegression
        x, y, patients = support()
        original = LogisticRegression.fit
        seen = []

        def fit_spy(estimator, fit_x, fit_y, *args, **kwargs):
            indices = [np.where(np.all(x == row, axis=1))[0].item() for row in fit_x]
            seen.append((estimator.C, indices))
            return original(estimator, fit_x, fit_y, *args, **kwargs)

        with patch.object(LogisticRegression, "fit", fit_spy):
            model, tuning = f.fit_classifier("linear", x, y, patients)
        self.assertEqual(len(seen), 16)
        self.assertEqual(len(tuning), 15)
        for (c, indices), row in zip(seen[:-1], tuning):
            self.assertEqual(c, row["C"])
            self.assertEqual(len(indices), 12)
            self.assertEqual(len(set(patients[indices])), 4)
            self.assertNotIn(row["heldout_patient"], set(patients[indices]))
            self.assertEqual(sorted(set(patients[indices])), json.loads(row["train_patients"]))
        self.assertEqual(len(seen[-1][1]), 15)
        self.assertEqual(seen[-1][0], model["selected_C"])
        self.assertEqual(sum(row["selected"] for row in tuning), 5)
        probabilities = f.predict_classifier(json.loads(json.dumps(model)), x)
        final = f._fit_linear(x, y, model["selected_C"])
        np.testing.assert_allclose(probabilities, final.predict_proba(x), atol=1e-12, rtol=0)
        for score in model["cv_scores"]:
            self.assertAlmostEqual(score["mean_log_loss"],
                np.mean([r["val_log_loss"] for r in tuning if r["C"] == score["C"]]))

    def test_ties_choose_stronger_regularization_with_fixed_tolerance(self):
        self.assertEqual(f._select_c([{"C": .1, "mean_log_loss": 1.0 + 5e-13},
                                     {"C": 1.0, "mean_log_loss": 1.0},
                                     {"C": 10.0, "mean_log_loss": 1.0}]), .1)
        self.assertEqual(f._select_c([{"C": .1, "mean_log_loss": 1.0 + 2e-12},
                                     {"C": 1.0, "mean_log_loss": 1.0}]), 1.)

    def test_convergence_warning_or_limit_fails(self):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        x, y, _ = support()

        def warn(*_args):
            warnings.warn("synthetic nonconvergence", ConvergenceWarning)

        with patch.object(LogisticRegression, "fit", warn):
            with self.assertRaisesRegex(ValueError, "converge"):
                f._fit_linear(x, y, 1.)

        def hit_limit(estimator, *_args):
            estimator.classes_ = np.arange(3)
            estimator.n_iter_ = np.array([10000])
            estimator.coef_ = np.zeros((3, 512))
            estimator.intercept_ = np.zeros(3)
            return estimator

        with patch.object(LogisticRegression, "fit", hit_limit):
            with self.assertRaisesRegex(ValueError, "convergence"):
                f._fit_linear(x, y, 1.)

    def test_export_configuration_and_input_norm_are_checked(self):
        x, y, p = support()
        model, _ = f.fit_classifier("prototype", x, y, p)
        with self.assertRaises(ValueError):
            f.predict_classifier(model, x * 2)
        model["config"]["prototype"]["temperature"] = .1
        with self.assertRaises(ValueError):
            f.predict_classifier(model, x)
        self.assertEqual(f.classifier_config()["prototype"]["temperature"], 1.)


class ExtractionTests(unittest.TestCase):
    def test_existing_weight_cache_requires_official_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            payload = b"synthetic weights; not a torch checkpoint"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            url = f"https://download.pytorch.org/models/resnet18-{digest[:8]}.pth"
            self.assertEqual(f._verified_weight_hash(path, url), digest)
            with self.assertRaisesRegex(ValueError, "official URL hash"):
                f._verified_weight_hash(path, "https://download.pytorch.org/models/resnet18-f37072fd.pth")

    def test_synthetic_encoder_is_frozen_and_repeatable(self):
        import torch
        from PIL import Image
        from torch import nn
        from src.fewshot import state_fingerprint

        torch.set_num_threads(1)
        torch.manual_seed(4)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.bn = nn.BatchNorm2d(4)
                self.dropout = nn.Dropout(.6)
                self.linear = nn.Linear(4, 512)

            def forward(self, tensor):
                value = self.bn(self.conv(tensor)).mean((2, 3))
                return self.linear(self.dropout(value))

        model = Tiny().train()
        before = state_fingerprint(model)
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for i in range(3):
                path = Path(directory) / f"{i}.png"
                Image.new("RGB", (43 + i, 31), (30 + i * 10, 90, 110)).save(path)
                rows.append({"image_name": path.name, "path": str(path),
                             "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            raw, evidence = f._extract_with_model(rows, model, "cpu")
            repeated, _ = f._extract_with_model(rows, model, "cpu")
            np.testing.assert_array_equal(raw, repeated)
            self.assertEqual(raw.shape, (3, 512))
            self.assertEqual(raw.dtype, np.float32)
            self.assertEqual(evidence["state_before_sha256"], before)
            self.assertEqual(evidence["state_after_sha256"], before)
            self.assertEqual(evidence["bn_before_sha256"], evidence["bn_after_sha256"])
            self.assertTrue(evidence["encoder_eval"])
            self.assertTrue(evidence["all_parameters_frozen"])
            rows[0]["image_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "changed before"):
                f._extract_with_model(rows, model, "cpu")


if __name__ == "__main__":
    unittest.main()
