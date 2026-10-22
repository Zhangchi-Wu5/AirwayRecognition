"""Server orchestration tests; no images, weights or GPU training are used."""
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_resnet_fewshot_suite as suite
from src.fewshot import prepare_experiment, sha256_file, write_csv, write_json


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(splits_dir=self.root / "splits", image_root=self.root / "images",
                                    output_root=self.root / "results", batch_size=16, num_workers=4)

    def tearDown(self):
        self.temporary.cleanup()

    def test_commands_share_server_reference_and_defaults(self):
        frozen = suite.phase_command(self.args, "frozen")
        formal = suite.phase_command(self.args, "finetune")
        smoke = suite.phase_command(self.args, "smoke")
        self.assertEqual(formal[formal.index("--reference-root") + 1], str(self.args.output_root / "frozen"))
        self.assertEqual(formal[formal.index("--device") + 1], "cuda")
        self.assertEqual(frozen[frozen.index("--repeats") + 1], "20")
        self.assertIn("--smoke-test", smoke)
        self.assertNotIn("--smoke-test", formal)
        self.assertNotIn("--resume", formal)
        (self.args.output_root / "finetune").mkdir(parents=True)
        write_json(self.args.output_root / "finetune/protocol.json", {})
        self.assertIn("--resume", suite.phase_command(self.args, "finetune"))

    def test_run_is_sequential_and_uses_separate_smoke_output(self):
        with patch.object(suite, "run_phase") as run, patch.object(suite, "verify_frozen"), patch.object(suite, "verify_finetune"), patch.object(suite, "aggregate_hashes", return_value={}):
            suite.run_suite(self.args, {}, {})
        self.assertEqual([call.args[1] for call in run.call_args_list], ["frozen", "smoke", "finetune"])
        self.assertEqual(suite.read_json(self.args.output_root / "suite.json")["formal_fits"], 162)

    def test_resume_verifies_frozen_before_skip(self):
        with suite.suite_lock(self.args.output_root):
            suite.ensure_identity(self.args, {})
        (self.args.output_root / "frozen").mkdir()
        write_json(self.args.output_root / "frozen/protocol.json", {"status": "completed"})
        with patch.object(suite, "run_phase") as run, patch.object(suite, "verify_frozen", side_effect=ValueError("corrupted")):
            with self.assertRaisesRegex(ValueError, "corrupted"):
                suite.run_suite(self.args, {}, {})
        run.assert_not_called()
        self.assertEqual(suite.read_json(self.args.output_root / "suite.json")["status"], "failed")

    def test_incomplete_frozen_is_preserved_before_restart(self):
        with suite.suite_lock(self.args.output_root):
            suite.ensure_identity(self.args, {})
        frozen = self.args.output_root / "frozen"
        frozen.mkdir()
        (frozen / "partial.txt").write_text("preserve this")
        with patch.object(suite, "run_phase"), patch.object(suite, "verify_frozen"), patch.object(suite, "verify_finetune"), patch.object(suite, "aggregate_hashes", return_value={}):
            suite.run_suite(self.args, {}, {})
        archived = list((self.args.output_root / "incomplete_attempts").glob("frozen_*/partial.txt"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(), "preserve this")

    def test_changed_configuration_cannot_resume(self):
        with suite.suite_lock(self.args.output_root):
            suite.ensure_identity(self.args, {})
            self.args.batch_size = 8
            with self.assertRaisesRegex(ValueError, "changed"):
                suite.ensure_identity(self.args, {})

    def test_another_process_cannot_lock_same_output(self):
        with suite.suite_lock(self.args.output_root):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                with suite.suite_lock(self.args.output_root):
                    self.fail("Should not acquire second lock")

    def test_runtime_change_cannot_resume(self):
        self.args.runtime = {"packages": {"torch": "2.3"}, "python": "3.11", "cuda": "12.1"}
        with patch.object(suite, "run_phase"), patch.object(suite, "verify_frozen"), patch.object(suite, "verify_finetune"), patch.object(suite, "aggregate_hashes", return_value={}):
            suite.run_suite(self.args, {}, {})
        self.args.runtime["packages"]["torch"] = "2.4"
        with self.assertRaisesRegex(ValueError, "Runtime.*versions changed"):
            suite.run_suite(self.args, {}, {})

    def test_pack_requires_completed_suite_and_verified_phases(self):
        with suite.suite_lock(self.args.output_root):
            saved = suite.ensure_identity(self.args, {})
        with self.assertRaisesRegex(ValueError, "not completed"):
            suite.pack_results(self.args, {}, {})
        saved["status"] = "completed"
        write_json(self.args.output_root / "suite.json", saved)
        with patch.object(suite, "verify_frozen", side_effect=ValueError("bad frozen")), patch.object(suite, "verify_finetune") as verify_ft:
            with self.assertRaisesRegex(ValueError, "bad frozen"):
                suite.pack_results(self.args, {}, {})
        verify_ft.assert_not_called()

    def test_pack_excludes_weights_and_incomplete_attempts_preserves_logs(self):
        with suite.suite_lock(self.args.output_root):
            saved = suite.ensure_identity(self.args, {})
        for name in ("frozen/REPORT.md", "finetune/runs/full/model.pt", "finetune/runs/full/result.json",
                     "smoke/runs/one/history.csv", "incomplete_attempts/old/result.json", "frozen.log", "finetune.log", "smoke.log",
                     "finetune/incomplete_attempts/old/history.csv", *(f"finetune/{name}" for name in suite.AGGREGATES)):
            path = self.args.output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        saved.update(status="completed", finetune_aggregate_sha256=suite.aggregate_hashes(self.args))
        write_json(self.args.output_root / "suite.json", saved)
        with patch.object(suite, "verify_frozen"), patch.object(suite, "verify_finetune"):
            archive = suite.pack_results(self.args, {}, {})
            with self.assertRaisesRegex(ValueError, "already exists"):
                suite.pack_results(self.args, {}, {})
        with tarfile.open(archive) as bundle:
            names = bundle.getnames()
        self.assertIn("results/finetune/runs/full/result.json", names)
        self.assertIn("results/finetune.log", names)
        self.assertFalse(any(name.endswith(".pt") or "/smoke/" in name or "/incomplete_attempts/" in name for name in names))
        self.assertTrue((self.args.output_root / "finetune/runs/full/model.pt").exists())
        self.assertTrue(archive.with_suffix(archive.suffix + ".sha256").exists())

    def test_pack_rejects_changed_aggregate(self):
        with suite.suite_lock(self.args.output_root):
            saved = suite.ensure_identity(self.args, {})
        saved.update(status="completed", finetune_aggregate_sha256={"REPORT.md": "old"})
        write_json(self.args.output_root / "suite.json", saved)
        with patch.object(suite, "verify_frozen"), patch.object(suite, "verify_finetune"), patch.object(suite, "aggregate_hashes", return_value={"REPORT.md": "new"}):
            with self.assertRaisesRegex(ValueError, "hashes changed"):
                suite.pack_results(self.args, {}, {})

    def make_frozen_fixture(self):
        self.args.image_root.mkdir()
        self.args.splits_dir.mkdir()
        for split, count in (("train", 20), ("val", 2), ("test", 2)):
            rows = []
            for patient in range(count):
                for label_id, label in enumerate(("lt", "yz", "zz")):
                    name = f"{split}_{patient}_{label}.png"
                    if split != "test":
                        (self.args.image_root / name).write_bytes(name.encode())
                    rows.append({"patient_id": f"{split}_{patient}", "label": label, "label_id": label_id, "path": f"/old/{name}"})
            write_csv(self.args.splits_dir / f"{split}.csv", rows)
        args = SimpleNamespace(**vars(self.args))
        args.output_root = self.args.output_root / "frozen"
        args.shots, args.repeats, args.sampling_seed, args.device = [1, 5, 10, 20], 20, 20261022, "cuda"
        frames, runs, protocol = prepare_experiment(args)
        for run in runs:
            path = args.output_root / "heads" / f"{run['run_id']}.npz"
            path.parent.mkdir(exist_ok=True)
            path.write_text("unit fixture only")
        for name in ("learning_curve.pdf", "learning_curve.png", "per_run_metrics.csv", "predictions.csv", "summary_metrics.csv", "features/train.npz", "features/val.npz"):
            path = args.output_root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("unit fixture only")
        protocol.update(status="completed", artifact_sha256={str(p.relative_to(args.output_root)): sha256_file(p) for p in args.output_root.rglob("*") if p.is_file() and p.name != "protocol.json"})
        write_json(args.output_root / "protocol.json", protocol)
        return frames, protocol["splits"]

    def test_all_171_frozen_artifacts_verified_before_reuse(self):
        frames, metadata = self.make_frozen_fixture()
        protocol = suite.verify_frozen(self.args, frames, metadata)
        self.assertEqual(len(protocol["artifact_sha256"]), 171)
        (self.args.output_root / "frozen/features/train.npz").write_text("corruption")
        with self.assertRaisesRegex(ValueError, "artifact corrupted"):
            suite.verify_frozen(self.args, frames, metadata)

    def test_missing_frozen_manifest_entry_is_rejected(self):
        frames, metadata = self.make_frozen_fixture()
        path = self.args.output_root / "frozen/protocol.json"
        protocol = suite.read_json(path)
        del protocol["artifact_sha256"]["features/train.npz"]
        write_json(path, protocol)
        with self.assertRaisesRegex(ValueError, "manifest is incomplete"):
            suite.verify_frozen(self.args, frames, metadata)


if __name__ == "__main__":
    unittest.main()
