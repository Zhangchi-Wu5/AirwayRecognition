"""Isolated synthetic renew protocol, real tiny training, and offline package tests."""
import copy
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.run_renew_experiment import parse_args
from src import renew_experiment as runner
from src.renew_data import audit_dataset


class RenewExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.images = self.base / "images"
        self.images.mkdir()
        for patient in range(1, 32):
            for label_id, label in enumerate(("lt", "yz", "zz")):
                Image.new("RGB", (32, 18), (patient * 5, label_id * 80, 100)).save(self.images / f"{patient:010d}{label}.png")
        self.output = self.base / "output"
        self.rows, self.suite = runner.prepare(self.output, self.images, loader=self.loader)
        self.suite["runtime_identity"] = {"synthetic": True}
        runner._atomic_json(self.output / "suite.json", self.suite)

    @staticmethod
    def loader(root, strict=True):
        return audit_dataset(root, strict=False)

    def tearDown(self):
        self.temp.cleanup()

    def full_jobs(self):
        return [j for j in self.suite["identity"]["jobs"] if j["budget"] == "full"]

    def mark(self, directory, job):
        runner._atomic_json(directory / "complete.json", {"identity_sha256": self.suite["identity_sha256"],
                            "run_id": job["run_id"], "artifact_sha256": {name: runner.sha256_file(directory / name)
                            for name in (*runner.RUN_FILES[:-1], "model.pt")}})

    def fixture_run(self, job):
        import numpy as np
        directory = self.output / "runs" / job["run_id"]
        directory.mkdir(parents=True)
        by_name = {r["image_name"]: r for r in self.rows}
        rows = [by_name[n] for n in job["eval"]]
        probabilities = []
        for index, row in enumerate(rows):
            prediction = (row["label_id"] + 1) % 3 if job["method"] == "original" and job["fold"] == 0 and index == 0 else row["label_id"]
            probabilities.append([.8 if label == prediction else .1 for label in range(3)])
        probabilities = np.asarray(probabilities, dtype=np.float32)
        predictions = [{"patient_id": r["patient_id"], "image_name": r["image_name"], "label_id": r["label_id"],
                        "predicted_id": int(p.argmax()), **{f"prob_{label}": float(p[i]) for label, i in runner.LABEL_TO_ID.items()}}
                       for r, p in zip(rows, probabilities)]
        config = self.suite["identity"]["config"]
        runner._atomic_json(directory / "config.json", {"identity_sha256": self.suite["identity_sha256"], "job": job,
                            "config": config, "initial_state_sha256": "same-initial", "execution_runtime": self.suite["runtime_identity"]})
        history = [{"stage": stage, "round": i + 1, "steps": 28, "image_occurrences": 448,
                    "lr": .001 if stage == "head" else .0001 * (1 + math.cos(math.pi * i / 20)) / 2,
                    "mean_train_loss": .5, "seconds": .01}
                   for stage, count in (("head", 5), ("finetune", 20)) for i in range(count)]
        runner.write_csv(directory / "history.csv", history)
        runner.write_csv(directory / "predictions.csv", predictions)
        result = {**{key: job[key] for key in ("run_id", "fold", "method", "budget", "repeat")},
                  "n_train_images": len(job["support"]), "n_train_patients": len(job["patient_order_prefix"]),
                  "initial_state_sha256": "same-initial", "head_stage_backbone_unchanged": True,
                  "finetune_backbone_changed": True, "initial_backbone_sha256": "before", "final_backbone_sha256": "after",
                  "bn_buffers_unchanged": True, "bn_before_sha256": "bn", "bn_after_sha256": "bn",
                  "final_state_sha256": "final", "reloaded_state_sha256": "final", "checkpoint_reload_verified": True,
                  "actual_optimizer_steps": 700, "actual_image_occurrences": 11200,
                  **runner.classification_metrics([r["label_id"] for r in rows], probabilities)}
        runner._atomic_json(directory / "result.json", result)
        (directory / "model.pt").write_bytes(b"synthetic fixture, never a scientific result")
        self.mark(directory, job)
        return directory

    def populate(self, all_jobs=False):
        for job in self.suite["identity"]["jobs"] if all_jobs else self.full_jobs():
            self.fixture_run(job)
        return runner.write_aggregates(self.output, self.suite)

    def test_plan_110_jobs_patient_grouping_nesting_and_equal_supports(self):
        jobs = self.suite["identity"]["jobs"]
        self.assertEqual(len(jobs), 110)
        self.assertEqual(len(self.full_jobs()), 10)
        self.assertEqual(len([j for j in jobs if j["budget"] != "full"]), 100)
        by_name = {r["image_name"]: r for r in self.rows}
        for fold in range(5):
            for repeat in range(5):
                a = next(j for j in jobs if (j["fold"], j["repeat"], j["budget"], j["method"]) == (fold, repeat, "5", "original"))
                b = next(j for j in jobs if (j["fold"], j["repeat"], j["budget"], j["method"]) == (fold, repeat, "10", "original"))
                c = next(j for j in jobs if (j["fold"], j["repeat"], j["budget"], j["method"]) == (fold, repeat, "5", "full_frame"))
                self.assertTrue(set(a["support"]) < set(b["support"]))
                self.assertEqual(a["support"], c["support"])
                self.assertEqual(a["sampling_seed"], 20261104 + fold * 100 + repeat)
                self.assertEqual(len({by_name[n]["patient_key"] for n in a["support"]}), 5)
                self.assertFalse({by_name[n]["patient_key"] for n in a["support"]} & {by_name[n]["patient_key"] for n in a["eval"]})
        args = parse_args(["run"])
        self.assertEqual(args.phase, "full")
        c = runner.fixed_config()
        self.assertEqual(c["head_rounds"] * c["steps_per_round"], 140)
        self.assertEqual(c["finetune_rounds"] * c["steps_per_round"], 560)

    def test_resume_is_phase_independent_but_source_config_and_data_changes_rejected(self):
        runner.prepare(self.output, self.images, loader=self.loader)
        changed = {**runner.source_hashes(), "src/models.py": "changed"}
        with patch.object(runner, "source_hashes", return_value=changed), self.assertRaisesRegex(ValueError, "Resume rejected"):
            runner.prepare(self.output, self.images, loader=self.loader)
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            runner.prepare(self.output, self.images, loader=self.loader, config={**runner.fixed_config(), "head_lr": .02})
        Image.new("RGB", (32, 18), (255, 255, 255)).save(self.images / self.rows[0]["image_name"])
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            runner.prepare(self.output, self.images, loader=self.loader)

    def test_changed_support_and_existing_unidentified_output_rejected(self):
        p = self.output / self.full_jobs()[0]["support_csv"]
        p.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Plan artifact"):
            runner.prepare(self.output, self.images, loader=self.loader)
        other = self.base / "other"
        other.mkdir()
        (other / "keep.txt").write_text("must not overwrite")
        with self.assertRaisesRegex(ValueError, "not empty"):
            runner.prepare(other, self.images, loader=self.loader)

    def test_completed_corruption_and_rehashed_false_metrics_rejected(self):
        job = self.full_jobs()[0]
        directory = self.fixture_run(job)
        self.assertIsNotNone(runner.validate_run(self.output, self.suite, job))
        result = runner.read_json(directory / "result.json")
        result["accuracy"] = .123
        runner._atomic_json(directory / "result.json", result)
        with self.assertRaisesRegex(ValueError, "corrupted"):
            runner.validate_run(self.output, self.suite, job)
        self.mark(directory, job)
        with self.assertRaisesRegex(ValueError, "Prediction metric"):
            runner.validate_run(self.output, self.suite, job)

    def test_oof_uses_93_images_and_not_unweighted_fold_means(self):
        data = self.populate()
        original = next(r for r in data["oof"] if r["method"] == "original")
        self.assertEqual(original["n_images"], 93)
        self.assertEqual(original["n_patients"], 31)
        self.assertAlmostEqual(original["accuracy"], 92 / 93)
        wrong_fold_average = sum(r["accuracy"] for r in data["results"] if r["method"] == "original") / 5
        self.assertNotAlmostEqual(original["accuracy"], wrong_fold_average)
        self.assertAlmostEqual(data["oof_pairs"][0]["accuracy_delta"], 1 / 93)
        self.assertTrue(all(r["sample_sd"] is None for r in data["summaries"]))

    def test_offline_full_pack_without_images_or_models_and_no_image_archive(self):
        self.populate()
        shutil.rmtree(self.images)
        for p in self.output.glob("runs/*/model.pt"):
            p.unlink()
        with patch.object(runner, "EXPECTED_FINGERPRINT", self.suite["identity"]["dataset_fingerprint"]), patch.object(runner, "environment", side_effect=AssertionError("CUDA inspected")):
            archive = runner.pack_suite(self.output)
        with tarfile.open(archive) as bundle:
            names = bundle.getnames()
            self.assertTrue(all(not n.endswith((".pt", ".png")) for n in names))
            self.assertFalse(any(n.startswith("preview/") for n in names))
            self.assertIn("suite.json", names)
            self.assertIn("oof_predictions.csv", names)
        self.assertEqual(Path(str(archive) + ".sha256").read_text().split()[0], runner.sha256_file(archive))

    def test_partial_pack_and_nonformal_config_rejected(self):
        self.fixture_run(self.full_jobs()[0])
        runner.write_aggregates(self.output, self.suite)
        with patch.object(runner, "EXPECTED_FINGERPRINT", self.suite["identity"]["dataset_fingerprint"]), self.assertRaisesRegex(ValueError, "complete full phase"):
            runner.pack_suite(self.output)
        altered = copy.deepcopy(self.suite["identity"])
        altered["config"]["steps_per_round"] = 1
        with self.assertRaisesRegex(ValueError, "fixed training protocol"):
            runner.validate_formal_identity(altered)

    def test_rehashed_aggregate_values_are_not_trusted(self):
        self.populate()
        path = self.output / "per_run_metrics.csv"
        rows = runner.read_csv(path)
        rows[0]["accuracy"] = "0.123"
        runner.write_csv(path, rows)
        self.suite["aggregate_sha256"][path.name] = runner.sha256_file(path)
        runner._atomic_json(self.output / "suite.json", self.suite)
        with patch.object(runner, "EXPECTED_FINGERPRINT", self.suite["identity"]["dataset_fingerprint"]), self.assertRaisesRegex(ValueError, "Aggregate values"):
            runner.pack_suite(self.output)

    def test_all_110_complete_has_22_oof_sets_not_465_independent_images(self):
        data = self.populate(all_jobs=True)
        self.assertEqual(len(data["results"]), 110)
        self.assertEqual(len(data["oof"]), 22)
        self.assertTrue(all(r["n_images"] == 93 and r["n_patients"] == 31 for r in data["oof"]))
        self.assertEqual(len(data["oof_pairs"]), 11)
        self.assertEqual({r["n_oof_repeats"] for r in data["summaries"] if r["budget"] != "full"}, {5})
        with patch.object(runner, "EXPECTED_FINGERPRINT", self.suite["identity"]["dataset_fingerprint"]):
            archive = runner.pack_suite(self.output)
        with tarfile.open(archive) as bundle:
            self.assertEqual(sum(n.endswith("/complete.json") for n in bundle.getnames()), 110)
            self.assertFalse(any(n.endswith(".pt") for n in bundle.getnames()))

    def test_fewshot_phase_requires_all_ten_full_fits(self):
        args = parse_args(["run", "--phase", "fewshot", "--image-root", str(self.images), "--output-root", str(self.output)])
        with patch.object(runner, "prepare", return_value=(self.rows, self.suite)), patch.object(runner, "environment", return_value={}):
            self.suite["runtime_identity"] = None
            with self.assertRaisesRegex(ValueError, "ten full-data"):
                runner.run_suite(args)

    def test_real_tiny_training_both_pipelines_reload_bn_and_completed_skip(self):
        import torch
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Linear(3, 8)
                self.bn = torch.nn.BatchNorm1d(8)
                self.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(8, 3))
            def forward(self, x):
                return self.fc(self.bn(self.stem(x.mean(dim=(2, 3)))).relu())
        config = {**runner.fixed_config(), "mode": "synthetic_test", "head_rounds": 1, "finetune_rounds": 1,
                  "steps_per_round": 1, "batch_size": 3, "num_workers": 0, "weights": "synthetic_untrained"}
        output = self.base / "tiny"
        rows, suite = runner.prepare(output, self.images, loader=self.loader, config=config)
        suite["runtime_identity"] = {"synthetic": True}
        runner._atomic_json(output / "suite.json", suite)
        mapping = {r["image_name"]: r for r in rows}
        hashes = []
        runner.configure_torch()
        for job in suite["identity"]["jobs"][:2]:
            result = runner.train_one(output, suite, job, mapping, torch.device("cpu"), model_factory=lambda pretrained: Tiny())
            hashes.append(result["initial_state_sha256"])
            self.assertTrue(result["checkpoint_reload_verified"])
            self.assertTrue(result["bn_buffers_unchanged"])
            self.assertTrue(result["head_stage_backbone_unchanged"])
            self.assertEqual(result["actual_optimizer_steps"], 2)
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                runner.train_one(output, suite, job, mapping, torch.device("cpu"), model_factory=lambda pretrained: Tiny())
        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
