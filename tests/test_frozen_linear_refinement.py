"""Synthetic-only tests of the one-time wider linear regularization grid."""
import copy
import json
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from src import frozen_linear_refinement as v2
from src import frozen_screen as v1


def support():
    rng = np.random.default_rng(29)
    labels = np.tile(np.arange(3), 5)
    patients = np.repeat([str(p) for p in range(5)], 3)
    features = rng.normal(0, .005, (15, 512))
    features[np.arange(15), labels] += 1.
    features[np.arange(15), np.arange(15) + 10] = .2
    return v1.l2_normalize(features), labels, patients


class LinearRefinementTests(unittest.TestCase):
    def test_grid_is_only_config_change_and_v1_remains_unchanged(self):
        before = v1.classifier_config()
        refined = v2.classifier_config()
        self.assertEqual(refined["linear"]["C"], [.1, 1., 10., 100., 1000.])
        refined["linear"]["C"] = before["linear"]["C"]
        self.assertEqual(refined, before)
        self.assertEqual(v1.classifier_config(), before)
        refined["linear"]["max_iter"] = 1
        self.assertEqual(v1.classifier_config()["linear"]["max_iter"], 10000)
        self.assertEqual(v2.classifier_config()["linear"]["max_iter"], 10000)

    def test_actual_five_candidate_lopo_fit_is_isolated_and_replays(self):
        from sklearn.linear_model import LogisticRegression
        x, y, patients = support()
        original = LogisticRegression.fit
        fits = []

        def spy(estimator, fit_x, fit_y, *args, **kwargs):
            indices = [np.where(np.all(x == row, axis=1))[0].item() for row in fit_x]
            fits.append((estimator.C, indices))
            return original(estimator, fit_x, fit_y, *args, **kwargs)

        with patch.object(LogisticRegression, "fit", spy):
            model, tuning = v2.fit_classifier(x, y, patients)
        self.assertEqual(len(fits), 26)
        self.assertEqual(len(tuning), 25)
        self.assertEqual([row["C"] for row in model["cv_scores"]], list(v2.C_GRID))
        for (c, indices), row in zip(fits[:-1], tuning):
            self.assertEqual(c, row["C"])
            self.assertEqual(len(indices), 12)
            self.assertEqual(len(set(patients[indices])), 4)
            self.assertNotIn(row["heldout_patient"], patients[indices])
            self.assertEqual(json.loads(row["train_patients"]), sorted(set(patients[indices])))
        self.assertEqual(len(fits[-1][1]), 15)
        self.assertEqual(fits[-1][0], model["selected_C"])
        self.assertEqual(sum(row["selected"] for row in tuning), 5)
        self.assertEqual(model["selected_C"], 1000.)
        exported = json.loads(json.dumps(model))
        replay = v2.predict_classifier(exported, x)
        refit = v1._fit_linear(x, y, model["selected_C"])
        np.testing.assert_allclose(replay, refit.predict_proba(x), rtol=0, atol=1e-12)
        old_model, old_tuning = v1.fit_classifier("linear", x, y, patients)
        for old, new in zip(old_tuning, tuning[:15]):
            self.assertEqual({k: v for k, v in old.items() if k != "selected"},
                             {k: v for k, v in new.items() if k != "selected"})
        self.assertEqual(old_model["cv_scores"], model["cv_scores"][:3])
        self.assertEqual(v1.classifier_config()["linear"]["C"], [.1, 1., 10.])
        with self.assertRaises(ValueError):
            v1.predict_classifier(model, x)

    def fake_fit(self, preferred):
        class Estimator:
            def __init__(self, c):
                self.coef_ = np.zeros((3, 512))
                self.coef_[np.arange(3), np.arange(3)] = 2. if c == preferred else .1
                self.intercept_ = np.zeros(3)
                self.n_iter_ = np.array([3])

            def predict_proba(self, x):
                scores = x @ self.coef_.T
                exp = np.exp(scores - scores.max(axis=1, keepdims=True))
                return exp / exp.sum(axis=1, keepdims=True)

        return lambda _x, _y, c: Estimator(c)

    def test_both_new_candidates_are_eligible_for_inner_selection(self):
        x, y, p = support()
        for selected in (100., 1000.):
            with self.subTest(selected=selected), patch.object(v2, "_fit_linear", self.fake_fit(selected)):
                model, tuning = v2.fit_classifier(x, y, p)
            self.assertEqual(model["selected_C"], selected)
            self.assertEqual(sum(row["selected"] for row in tuning), 5)
            np.testing.assert_array_equal(v2.predict_classifier(json.loads(json.dumps(model)), x).argmax(1), y)

    def test_ties_and_malformed_exports_are_rejected(self):
        x, y, p = support()
        with patch.object(v2, "_fit_linear", self.fake_fit(None)):
            model, _ = v2.fit_classifier(x, y, p)
        self.assertEqual(model["selected_C"], .1)
        self.assertEqual(v2._select_c([{"C": 100., "mean_log_loss": 1. + 5e-13},
                                      {"C": 1000., "mean_log_loss": 1.}]), 100.)
        invalid_models = []
        for key, value in (("selected_C", 1000.), ("n_iter", 10000),
                           ("classes", [2, 1, 0]), ("version", v1.MODEL_VERSION)):
            changed = copy.deepcopy(model)
            changed[key] = value
            invalid_models.append(changed)
        changed = copy.deepcopy(model)
        changed["coef"][0][1] = float("inf")
        invalid_models.append(changed)
        changed = copy.deepcopy(model)
        changed["cv_scores"][0]["mean_log_loss"] = float("nan")
        invalid_models.append(changed)
        for changed in invalid_models:
            with self.assertRaises(ValueError):
                v2.predict_classifier(changed, x)

    def test_incomplete_support_and_solver_nonconvergence_fail(self):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        x, y, p = support()
        with self.assertRaises(ValueError):
            v2.fit_classifier(x[:-1], y[:-1], p[:-1])

        def no_convergence(*_args):
            warnings.warn("synthetic nonconvergence", ConvergenceWarning)

        with patch.object(LogisticRegression, "fit", no_convergence):
            with self.assertRaisesRegex(ValueError, "converge"):
                v2.fit_classifier(x, y, p)


if __name__ == "__main__":
    unittest.main()
