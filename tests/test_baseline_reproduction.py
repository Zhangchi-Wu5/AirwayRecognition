"""Offline tests for identity, recovery, references and portable packaging."""
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_resnet_baseline_reproduction import parse_args
from src import baseline_reproduction as runner
from src.fewshot import classification_metrics, sha256_file, write_csv
from src.fewshot_finetune import _atomic_json, _digest


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "experiment"
        self.root.mkdir()
        self.args = parse_args(["run", "--output-root", str(self.root)])
        self.probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]], dtype=np.float32)
        self.metrics = classification_metrics([0, 1, 2], self.probabilities)
        self.predictions = [{"patient_id": f"p{i}", "image_name": f"{i}.png", "label_id": i,
                             "predicted_id": i, **{f"prob_{label}": float(prob[j]) for j, label in enumerate(("lt", "yz", "zz"))}}
                            for i, prob in enumerate(self.probabilities)]
        self.identity = {"version": runner.VERSION, "historical_csv_bytes_match": {"train": False, "val": False, "test": False}}
        self.suite = {"identity": self.identity, "identity_sha256": _digest(self.identity), "status": "completed",
                      "runtime_identity": runner.runtime_identity({}), "reference_artifact_sha256": {}}
        ref = self.root / "references/current_seed_42"
        ref.mkdir(parents=True)
        write_csv(ref / "predictions.csv", self.predictions)
        _atomic_json(ref / "result.json", self.metrics)
        self.suite["reference_artifact_sha256"] = {str(path.relative_to(self.root)): sha256_file(path) for path in ref.iterdir()}

    def tearDown(self):
        self.temporary.cleanup()

    def make_job(self, job):
        directory = runner.job_directory(self.root, job)
        directory.mkdir(parents=True)
        historical = job.startswith("historical_")
        seed = int(job.rsplit("_", 1)[1])
        result = {**self.metrics, "job": job, "method": "historical_epoch" if historical else "current_fixed_updates",
                  "seed": seed, "identity_sha256": self.suite["identity_sha256"], "test_images_accessed": False,
                  "checkpoint_reload_verified": True, "batchnorm_buffers_unchanged": True, "bn_buffers_unchanged": True,
                  "historical_accuracy": .99, "historical_accuracy_difference": .01,
                  "initial_state_matches_history": True, "training_input_stream_matches_history": False,
                  "final_state_matches_history": False,
                  "reproduction_scope": "same_membership_reproduction"}
        _atomic_json(directory / "result.json", result)
        write_csv(directory / "predictions.csv", self.predictions)
        if historical:
            files = ["runtime_flags.json", "training/protocol.json", f"training/seed_{seed}/pair_metadata.json"]
            files += [f"training/seed_{seed}/baseline/{name}" for name in
                      ("run_config.json", "history.csv", "training_input_trace.csv", "best_val.pt", "last_model.pt")]
        else:
            files = ["protocol.json"] + [f"full/{name}" for name in
                                          ("config.json", "history.csv", "model.pt", "predictions.csv", "result.json", "complete.json")]
        for name in files:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        runner.complete_job(directory, self.suite, job)
        return directory

    def make_completed_suite(self):
        for job in runner.JOBS:
            self.make_job(job)
        runner.aggregate(self.root, self.suite)
        _atomic_json(self.root / "suite.json", self.suite)

    def test_cli_full_stage_only_and_disjoint_job_list(self):
        self.assertEqual(runner.JOBS, ("historical_42", "historical_2026", "historical_3407", "current_2026", "current_3407"))
        self.assertEqual(self.args.historical_splits_dir, self.args.splits_dir)
        with self.assertRaises(SystemExit):
            parse_args(["run", "--job", "current_2026"])

    def test_historical_original_baseline_arguments(self):
        command = runner.historical_arguments(self.args, 2026, self.root)
        self.assertIn("--skip-evaluation", command)
        self.assertEqual(command[command.index("--arms") + 1], "baseline")
        self.assertEqual(command[command.index("--seeds") + 1], "2026")
        self.assertEqual(command[command.index("--stage1-epochs") + 1], "5")
        self.assertEqual(command[command.index("--stage2-epochs") + 1], "20")

    def test_current_seed_is_declared_before_identity(self):
        left = runner.make_current_config(self.args, 446, 42)
        right = runner.make_current_config(self.args, 446, 2026)
        self.assertEqual(left["steps_per_round"], 28)
        self.assertEqual(right["training_seed"], 2026)
        self.assertEqual({k: v for k, v in left.items() if k != "training_seed"},
                         {k: v for k, v in right.items() if k != "training_seed"})
        self.assertNotEqual(_digest(left), _digest(right))

    def test_worker_commands_are_separate_processes_and_explicit(self):
        commands = [runner.worker_command(self.args, job) for job in runner.JOBS]
        self.assertEqual(len(commands), 5)
        self.assertTrue(all("_worker" in command for command in commands))
        self.assertEqual([command[command.index("--job") + 1] for command in commands], list(runner.JOBS))
        self.assertTrue(all("--historical-splits-dir" in command for command in commands))

    def test_retained_seed_runtime_drift_rejected(self):
        old = {"python": "x", "packages": {"torch": "2"}, "gpu": "GPU"}
        identity = {"current_reference": {"environment": old}}
        runner.verify_reference_runtime(identity, old)
        with self.assertRaisesRegex(ValueError, "Environment differs"):
            runner.verify_reference_runtime(identity, {**old, "packages": {"torch": "3"}})
        with self.assertRaisesRegex(ValueError, "GPU model differs"):
            runner.verify_reference_runtime(identity, {**old, "gpu": "other"})

    def test_prediction_metric_and_identity_mismatch_rejected(self):
        path = self.root / "references/current_seed_42/predictions.csv"
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            runner.verify_prediction_metrics(path, {**self.metrics, "accuracy": .5})
        with self.assertRaisesRegex(ValueError, "identity/order"):
            runner.verify_prediction_metrics(path, self.metrics, list(reversed(self.predictions)))

    def test_float32_extreme_probabilities_preserve_original_log_loss(self):
        import numpy as np
        probabilities = np.eye(3, dtype=np.float32)
        metrics = classification_metrics([0, 1, 2], probabilities)
        rows = [{**row, **{f"prob_{label}": float(prob[j]) for j, label in enumerate(("lt", "yz", "zz"))}}
                for row, prob in zip(self.predictions, probabilities)]
        path = self.root / "extreme.csv"
        write_csv(path, rows)
        self.assertEqual(runner.verify_prediction_metrics(path, metrics)["loss"], metrics["loss"])
        runner.verify_prediction_metrics(path, {**metrics, "loss": metrics["loss"] + 1.2e-7})
        with self.assertRaisesRegex(ValueError, "metric mismatch: loss"):
            runner.verify_prediction_metrics(path, {**metrics, "loss": metrics["loss"] + 1e-4})

    def test_complete_job_requires_intact_weights_for_resume(self):
        directory = self.make_job("current_2026")
        self.assertIsNotNone(runner.validate_job(self.root, self.suite, "current_2026"))
        (directory / "full/model.pt").unlink()
        with self.assertRaisesRegex(ValueError, "corrupted/missing"):
            runner.validate_job(self.root, self.suite, "current_2026")
        self.assertIsNotNone(runner.validate_job(self.root, self.suite, "current_2026", require_weights=False))

    def test_pack_still_rejects_present_corrupt_weights(self):
        directory = self.make_job("current_2026")
        (directory / "full/model.pt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "corrupted/missing"):
            runner.validate_job(self.root, self.suite, "current_2026", require_weights=False)

    def test_incomplete_manifest_and_missing_bn_evidence_rejected(self):
        directory = self.make_job("current_2026")
        marker = runner.read_json(directory / "job_complete.json")
        marker["artifact_sha256"].pop("protocol.json")
        _atomic_json(directory / "job_complete.json", marker)
        with self.assertRaisesRegex(ValueError, "manifest"):
            runner.validate_job(self.root, self.suite, "current_2026")
        result = runner.read_json(directory / "result.json")
        result["bn_buffers_unchanged"] = False
        _atomic_json(directory / "result.json", result)
        runner.complete_job(directory, self.suite, "current_2026")
        with self.assertRaisesRegex(ValueError, "BatchNorm"):
            runner.validate_job(self.root, self.suite, "current_2026")

    def test_reference_tamper_and_path_traversal_rejected(self):
        (self.root / "references/current_seed_42/result.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "reference corrupted"):
            runner.verify_copied_reference(self.root, self.suite)
        outside = self.root.parent / "outside.json"
        outside.write_text("outside")
        self.suite["reference_artifact_sha256"] = {"../outside.json": sha256_file(outside)}
        with self.assertRaisesRegex(ValueError, "reference corrupted"):
            runner.verify_copied_reference(self.root, self.suite)

    def test_complete_pack_without_cuda_old_suite_or_weights(self):
        self.make_completed_suite()
        for path in self.root.rglob("*.pt"):
            path.unlink()
        abandoned = self.root / "incomplete_attempts/old"
        abandoned.mkdir(parents=True)
        (abandoned / "should_not_ship.txt").write_text("failure")
        with patch.object(runner, "check_environment", side_effect=AssertionError("pack must be offline")):
            archive = runner.pack_results(self.args)
        with tarfile.open(archive) as handle:
            names = handle.getnames()
        self.assertIn("references/current_seed_42/predictions.csv", names)
        self.assertIn("REPORT.md", names)
        self.assertFalse(any("incomplete_attempts" in name or name.endswith(".pt") for name in names))
        self.assertTrue(Path(str(archive) + ".sha256").read_text().startswith(sha256_file(archive)))
        with self.assertRaisesRegex(ValueError, "already exists"):
            runner.pack_results(self.args)

    def test_incomplete_suite_cannot_pack(self):
        self.suite["status"] = "failed"
        _atomic_json(self.root / "suite.json", self.suite)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            runner.pack_results(self.args)

    def test_worker_rejects_input_drift_before_training(self):
        _atomic_json(self.root / "suite.json", self.suite)
        args = SimpleNamespace(**vars(self.args))
        args.job = "current_2026"
        with patch.object(runner, "check_inputs", return_value=({}, {}, {"changed": True}, {})), \
             patch.object(runner, "train_current", side_effect=AssertionError("must not train")):
            with self.assertRaisesRegex(ValueError, "arguments/data/reference/source changed"):
                runner.run_worker(args)

    def test_resume_skips_verified_jobs_and_archives_incomplete_job(self):
        self.make_completed_suite()
        incomplete = self.root / "runs/current_3407"
        (incomplete / "job_complete.json").unlink()
        self.suite["status"] = "failed"
        _atomic_json(self.root / "suite.json", self.suite)
        calls = []
        def run(args, job):
            calls.append(job)
            self.assertFalse(runner.job_directory(self.root, job).exists())
            self.make_job(job)
        with patch.object(runner, "check_inputs", return_value=({}, {}, self.identity, {})), \
             patch.object(runner, "check_environment", return_value={}), \
             patch.object(runner, "verify_reference_runtime"), \
             patch.object(runner, "run_subprocess", side_effect=run):
            runner.run_suite(self.args)
        self.assertEqual(calls, ["current_3407"])
        self.assertEqual(len(list((self.root / "incomplete_attempts").iterdir())), 1)
        self.assertEqual(runner.read_json(self.root / "suite.json")["status"], "completed")


if __name__ == "__main__":
    unittest.main()
