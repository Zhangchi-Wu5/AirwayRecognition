"""Synthetic/offline integrity tests and tiny-network optimizer equivalence."""
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_resnet_layer4_suite import parse_args
from src import fewshot_partial as runner
from src.fewshot import classification_metrics, prepare_experiment, sha256_file, state_fingerprint, write_csv
from src.fewshot_finetune import (_atomic_json, _digest, _portable_rows, make_config, prepare,
                                 seed_training, train_fixed_updates as old_train)


class TinyTrainingTests(unittest.TestCase):
    @staticmethod
    def factory():
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU())
                self.layer4 = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU())
                self.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(8, 3))

            def forward(self, x):
                return self.fc(self.layer4(self.stem(x)))
        return Tiny()

    def inputs(self):
        import torch
        from torch.utils.data import TensorDataset
        data = TensorDataset(torch.arange(36, dtype=torch.float32).reshape(9, 4) / 7 - 2, torch.arange(9) % 3)
        config = make_config(SimpleNamespace(smoke_test=True, batch_size=4, num_workers=0, device="cpu"), 9)
        return data, config, torch.device("cpu")

    def test_full_mode_is_numerically_identical_to_unchanged_trainer(self):
        data, config, device = self.inputs()
        seed_training(42)
        old = self.factory()
        old_history, _ = old_train(old, data, config, device)
        seed_training(42)
        new = self.factory()
        new_history, _ = runner.train_fixed_updates(new, data, config, device, scope="full")
        self.assertEqual(state_fingerprint(old), state_fingerprint(new))
        self.assertEqual([{k: v for k, v in r.items() if k != "seconds"} for r in old_history],
                         [{k: v for k, v in r.items() if k != "seconds"} for r in new_history])

    def test_layer4_only_updates_last_stage_and_head_with_fixed_bn(self):
        data, config, device = self.inputs()
        seed_training(42)
        model = self.factory()
        runner.set_training_stage(model, "finetune")
        self.assertFalse(model.stem[0].weight.requires_grad)
        self.assertFalse(model.stem[1].weight.requires_grad)
        self.assertTrue(model.layer4[0].weight.requires_grad)
        self.assertTrue(model.layer4[1].weight.requires_grad)
        self.assertTrue(model.fc[1].weight.requires_grad)
        self.assertFalse(model.layer4[1].training)
        history, evidence = runner.train_fixed_updates(model, data, config, device)
        for key in ("bn_buffers_unchanged", "early_parameters_unchanged", "layer4_updated", "fc_updated_in_stage2", "head_stage_backbone_unchanged"):
            self.assertTrue(evidence[key])
        self.assertEqual(evidence["actual_optimizer_steps"], 4)
        seed_training(42)
        full = self.factory()
        full_history, _ = old_train(full, data, config, device)
        self.assertTrue(runner.compare_head_history(history, full_history, 1))
        self.assertNotEqual(state_fingerprint(model), state_fingerprint(full))

    def test_head_loss_drift_is_rejected(self):
        row = {"stage": "head", "round": 1, "steps": 28, "image_occurrences": 448, "lr": .001, "mean_train_loss": .5}
        with self.assertRaisesRegex(ValueError, "Head-stage"):
            runner.compare_head_history([row], [{**row, "mean_train_loss": .5001}], 1)


class PairedSuiteTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.images, self.splits, self.reference, self.root = [self.base / name for name in ("images", "splits", "reference", "output")]
        self.images.mkdir()
        self.splits.mkdir()
        for split, count in (("train", 446), ("val", 96), ("test", 99)):
            rows = []
            for i in range(count):
                label_id, patient = i % 3, i // 3
                label = ("lt", "yz", "zz")[label_id]
                name = f"{split}_{i}.png"
                if split != "test":
                    (self.images / name).write_bytes(name.encode())
                rows.append({"patient_id": f"{split}_{patient:03d}", "label": label, "label_id": label_id, "path": str(self.images / name)})
            write_csv(self.splits / f"{split}.csv", rows)
        base_args = SimpleNamespace(splits_dir=self.splits, image_root=self.images, shots=[1, 5, 10, 20], repeats=20,
                                    sampling_seed=20261022, batch_size=16, num_workers=4, device="cuda", output_root=self.reference / "frozen")
        _, _, frozen = prepare_experiment(base_args)
        frozen["status"] = "completed"
        _atomic_json(base_args.output_root / "protocol.json", frozen)
        ftargs = SimpleNamespace(**{**vars(base_args), "output_root": self.reference / "finetune", "reference_root": self.reference / "frozen",
                                    "resume": False, "run_ids": None, "smoke_test": False})
        self.frames, runs, protocol = prepare(ftargs)
        self.environment = {"python": "fixture", "packages": {}, "gpu": "fixture"}
        protocol.update(status="completed", expected_runs=81, completed_runs=81, environment_execution=self.environment)
        _atomic_json(ftargs.output_root / "protocol.json", protocol)
        self.old_protocol = protocol
        self.runs = [r for r in runs if r["run_id"] in runner.RUN_IDS]
        self.probabilities = np.array([[.8 if label == r["label_id"] else .1 for label in range(3)] for r in self.frames["val"]], dtype=np.float32)
        self.metrics = classification_metrics([r["label_id"] for r in self.frames["val"]], self.probabilities)
        self.predictions = [{"patient_id": row["patient_id"], "image_name": row["image_name"], "label_id": row["label_id"],
                             "predicted_id": int(prob.argmax()), **{f"prob_{label}": float(prob[i]) for label, i in runner.LABEL_TO_ID.items()}}
                            for row, prob in zip(self.frames["val"], self.probabilities)]
        self.history = [{"stage": stage, "round": i + 1, "steps": 28, "image_occurrences": 448,
                         "lr": .001 if stage == "head" else .0001 * (1 + math.cos(math.pi * i / 20)) / 2,
                         "mean_train_loss": .5, "seconds": .1}
                        for stage, count in (("head", 5), ("finetune", 20)) for i in range(count)]
        for run in self.runs:
            directory = self.reference / "finetune/runs" / run["run_id"]
            directory.mkdir()
            result = {"run_id": run["run_id"], "budget": run["budget"], "repeat": run["repeat"], "n_train_images": len(run["rows"]),
                      "n_train_patients": len({r["patient_id"] for r in run["rows"]}), "initial_state_sha256": "initial", **self.metrics,
                      "checkpoint_reload_verified": True, "bn_buffers_unchanged": True, "head_stage_backbone_unchanged": True,
                      "finetune_backbone_changed": True, "actual_optimizer_steps": 700, "actual_image_occurrences": 11200,
                      "bn_before_sha256": "bn", "bn_after_sha256": "bn", "final_state_sha256": "final", "reloaded_state_sha256": "final"}
            _atomic_json(directory / "result.json", result)
            _atomic_json(directory / "config.json", {"identity_sha256": protocol["identity_sha256"], "run_id": run["run_id"], "config": protocol["config"],
                         "support": _portable_rows(run["rows"]), "initial_state_sha256": "initial", "execution_environment": self.environment})
            write_csv(directory / "history.csv", self.history)
            write_csv(directory / "predictions.csv", self.predictions)
            (directory / "model.pt").write_bytes(b"fixture checkpoint")
            self.mark(directory, protocol["identity_sha256"], run["run_id"])
        self.args = parse_args(["run", "--reference-root", str(self.reference), "--splits-dir", str(self.splits), "--image-root", str(self.images), "--output-root", str(self.root)])
        self.frames, self.runs, self.identity = runner.check_inputs(self.args)
        with runner.output_lock(self.root):
            self.suite = runner.prepare_suite(self.args, self.identity, self.environment)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def mark(directory, identity, run_id):
        _atomic_json(directory / "complete.json", {"identity_sha256": identity, "run_id": run_id,
                     "artifact_sha256": {name: sha256_file(directory / name) for name in (*runner.RUN_FILES[:-1], "model.pt")}})

    def make_run(self, run_id):
        old = self.root / "references/finetune/runs" / run_id
        directory = self.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        for name in runner.RUN_FILES[:-1]:
            shutil.copyfile(old / name, directory / name)
        config = runner.read_json(directory / "config.json")
        config.update(identity_sha256=self.suite["identity_sha256"], config=self.identity["config"], execution_environment=self.suite["environment"])
        _atomic_json(directory / "config.json", config)
        result = runner.read_json(directory / "result.json")
        result.update(early_parameters_unchanged=True, layer4_updated=True, fc_updated_in_stage2=True,
                      initialization_matches_reference=True, head_history_matches_reference=True, test_images_accessed=False,
                      early_before_sha256="early", early_after_sha256="early", bn_before_sha256="bn", bn_after_sha256="bn",
                      final_state_sha256="final", reloaded_state_sha256="final",
                      stage2_before_sha256={"layer4": "l4-before", "fc": "fc-before"}, stage2_after_sha256={"layer4": "l4-after", "fc": "fc-after"})
        _atomic_json(directory / "result.json", result)
        (directory / "model.pt").write_bytes(b"fixture checkpoint")
        self.mark(directory, self.suite["identity_sha256"], run_id)
        return directory

    def make_completed(self):
        for run_id in runner.RUN_IDS:
            self.make_run(run_id)
        runner.write_aggregate(self.root, self.suite)

    def test_exact_40_reference_supports_and_no_test_pixels(self):
        self.assertEqual(len(self.runs), 40)
        self.assertEqual({r["budget"] for r in self.runs}, {"5", "10"})
        self.assertEqual(self.identity["config"]["num_workers"], 4)
        self.assertEqual(self.identity["config"]["steps_per_round"], 28)
        self.assertFalse(list(self.images.glob("test*")))
        runner.verify_copied_reference(self.root, self.suite)

    def test_changed_config_or_runtime_blocks_resume(self):
        with self.assertRaisesRegex(ValueError, "source/data/config/reference"):
            runner.prepare_suite(self.args, {**self.identity, "version": "changed"}, self.environment)
        with self.assertRaisesRegex(ValueError, "runtime"):
            runner.prepare_suite(self.args, self.identity, {**self.environment, "gpu": "other"})

    def test_source_image_and_support_corruption_rejected(self):
        (self.images / "train_0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image/order/hash"):
            runner.check_inputs(self.args)
        path = self.root / "references/finetune/supports" / (runner.RUN_IDS[0] + ".csv")
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Copied reference corrupted"):
            runner.verify_copied_reference(self.root, self.suite)

    def test_weights_required_for_resume_optional_for_offline_pack_validation(self):
        directory = self.make_run(runner.RUN_IDS[0])
        runner.validate_run(self.root, self.suite, runner.RUN_IDS[0])
        (directory / "model.pt").unlink()
        with self.assertRaisesRegex(ValueError, "corrupted"):
            runner.validate_run(self.root, self.suite, runner.RUN_IDS[0])
        runner.validate_run(self.root, self.suite, runner.RUN_IDS[0], False)
        (directory / "model.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupted"):
            runner.validate_run(self.root, self.suite, runner.RUN_IDS[0], False)

    def test_modified_invariant_with_rehashed_manifest_still_fails(self):
        run_id = runner.RUN_IDS[0]
        directory = self.make_run(run_id)
        result = runner.read_json(directory / "result.json")
        result["early_after_sha256"] = "changed"
        _atomic_json(directory / "result.json", result)
        self.mark(directory, self.suite["identity_sha256"], run_id)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            runner.validate_run(self.root, self.suite, run_id)

    def test_pack_offline_contains_80_paired_rows_without_weights(self):
        self.make_completed()
        shutil.rmtree(self.reference)
        shutil.rmtree(self.images)
        for path in (self.root / "runs").glob("*/model.pt"):
            path.unlink()
        with patch.object(runner, "check_environment", side_effect=AssertionError("CUDA used")), patch.object(runner, "check_inputs", side_effect=AssertionError("source data used")):
            archive = runner.pack_suite(self.args)
        self.assertEqual(len(runner.read_csv(self.root / "per_run_metrics.csv")), 80)
        self.assertEqual(len(runner.read_csv(self.root / "paired_comparison.csv")), 40)
        with tarfile.open(archive) as handle:
            names = handle.getnames()
            self.assertFalse(any(name.endswith(".pt") for name in names))
            self.assertIn("suite.json", names)
        self.assertEqual(Path(str(archive) + ".sha256").read_text().split()[0], sha256_file(archive))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            runner.pack_suite(self.args)

    def test_aggregate_corruption_and_partial_pack_rejected(self):
        self.make_run(runner.RUN_IDS[0])
        runner.write_aggregate(self.root, self.suite)
        with self.assertRaisesRegex(ValueError, "completed 40-run"):
            runner.pack_suite(self.args)
        self.suite["status"] = "completed"
        self.suite["completed_runs"] = 40
        _atomic_json(self.root / "suite.json", self.suite)
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            runner.pack_suite(self.args)

    def test_rehashed_aggregate_still_checked_against_run_evidence(self):
        self.make_completed()
        path = self.root / "paired_comparison.csv"
        rows = runner.read_csv(path)
        rows[0]["accuracy_difference"] = ".2"
        write_csv(path, rows)
        self.suite["aggregate_sha256"][path.name] = sha256_file(path)
        _atomic_json(self.root / "suite.json", self.suite)
        with self.assertRaisesRegex(ValueError, "differs from run"):
            runner.pack_suite(self.args)

    def test_safe_paths_and_runtime_model_guard(self):
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            runner.safe_path(self.root, "../outside")
        runner.verify_runtime(self.environment, self.environment)
        with self.assertRaisesRegex(ValueError, "GPU model"):
            runner.verify_runtime(self.environment, {**self.environment, "gpu": "other"})
        with self.assertRaisesRegex(ValueError, "Environment"):
            runner.verify_runtime(self.environment, {**self.environment, "packages": {"torch": "changed"}})

    def test_resume_skips_verified_and_archives_unfinished_runs(self):
        self.make_run(runner.RUN_IDS[0])
        unfinished = self.root / "runs" / runner.RUN_IDS[1]
        unfinished.mkdir(parents=True)
        (unfinished / "partial.txt").write_text("retain this failed attempt")
        def synthetic_fit(root, suite, frames, run, directory, device):
            self.make_run(run["run_id"])
        with patch.object(runner, "check_environment", return_value=self.environment), \
             patch.object(runner, "train_run", side_effect=synthetic_fit) as train, \
             patch("torch.cuda.empty_cache"):
            runner.run_suite(self.args)
        self.assertEqual(train.call_count, 39)
        self.assertEqual(len(list((self.root / "incomplete_attempts").glob("*/partial.txt"))), 1)
        self.assertEqual(runner.read_json(self.root / "suite.json")["status"], "completed")

    def test_online_aggregate_skips_already_verified_checkpoint_hashes(self):
        self.make_run(runner.RUN_IDS[0])
        import src.fewshot_finetune as old_module
        original = old_module.sha256_file
        def guarded(path):
            self.assertNotEqual(Path(path).name, "model.pt")
            return original(path)
        with patch.object(old_module, "sha256_file", side_effect=guarded):
            runner.write_aggregate(self.root, self.suite)

    def test_output_lock_is_exclusive(self):
        with runner.output_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another"):
                with runner.output_lock(self.root):
                    pass


if __name__ == "__main__":
    unittest.main()
