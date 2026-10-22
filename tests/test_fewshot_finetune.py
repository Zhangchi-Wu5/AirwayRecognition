"""Offline protocol/integrity tests and a small real two-stage optimizer test."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from src.fewshot import prepare_experiment, sha256_file, write_csv, write_json
from src.fewshot_finetune import (RepeatedBatchSampler, make_config, prepare, save_reload_final,
                                 seed_training, set_training_stage, train_fixed_updates,
                                 validate_completed, write_aggregate)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.splits = self.root / "splits"
        self.splits.mkdir()
        for split, count in (("train", 20), ("val", 2), ("test", 2)):
            rows = []
            for patient in range(count):
                for label_id, label in enumerate(("lt", "yz", "zz")):
                    name = f"{split}_{patient}_{label}.png"
                    if split != "test":
                        (self.images / name).write_bytes(name.encode())
                    rows.append({"patient_id": f"{split}_{patient:03d}", "label": label, "label_id": label_id,
                                 "path": f"/prior/machine/{name}"})
            write_csv(self.splits / f"{split}.csv", rows)
        self.reference = self.root / "reference"
        ref_args = SimpleNamespace(splits_dir=self.splits, image_root=self.images, output_root=self.reference,
                                   shots=[1, 5, 10, 20], repeats=20, sampling_seed=20261022,
                                   batch_size=16, num_workers=0, device="cpu")
        _, _, protocol = prepare_experiment(ref_args)
        protocol["status"] = "completed"
        write_json(self.reference / "protocol.json", protocol)
        self.args = SimpleNamespace(reference_root=self.reference, splits_dir=self.splits, image_root=self.images,
                                    output_root=self.root / "formal", device="cpu", batch_size=16, num_workers=0,
                                    resume=False, run_ids=None, smoke_test=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_plan_matches_81_supports_without_test_images(self):
        frames, runs, protocol = prepare(self.args)
        self.assertEqual(len(runs), 81)
        self.assertEqual(protocol["config"]["steps_per_round"], 4)
        self.assertFalse(protocol["test_images_accessed"])
        self.assertEqual(len(frames["test"]), 6)
        self.assertFalse(any((self.images / r["image_name"]).exists() for r in frames["test"]))

    def test_no_overwrite_resume_and_config_change(self):
        prepare(self.args)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare(self.args)
        self.args.resume = True
        self.assertEqual(len(prepare(self.args)[1]), 81)
        self.args.batch_size = 8
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            prepare(self.args)

    def test_changed_image_rejected(self):
        (self.images / "train_0_lt.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image/order/hash"):
            prepare(self.args)

    def test_changed_reference_support_rejected(self):
        path = self.reference / "supports/shot_001_repeat_000.csv"
        with path.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare(self.args)

    def test_reordered_split_rejected(self):
        path = self.splits / "train.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        write_csv(path, list(reversed(rows)))
        with self.assertRaisesRegex(ValueError, "image/order/hash"):
            prepare(self.args)

    def test_saved_support_mutation_rejected_on_resume(self):
        prepare(self.args)
        path = self.args.output_root / "supports/full.csv"
        path.write_text("modified")
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "support artifact"):
            prepare(self.args)

    def test_smoke_cannot_mix_with_formal(self):
        self.args.smoke_test = True
        _, runs, protocol = prepare(self.args)
        self.assertEqual(len(runs), 1)
        self.assertEqual(protocol["config"]["steps_per_round"], 2)
        self.args.smoke_test = False
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            prepare(self.args)

    def test_partial_aggregate_has_no_learning_curve(self):
        _, runs, protocol = prepare(self.args)
        results = write_aggregate(self.args, runs, protocol)
        self.assertEqual(results, [])
        self.assertEqual(protocol["status"], "partial")
        self.assertFalse((self.args.output_root / "learning_curve.png").exists())

    def test_completed_marker_requires_intact_checkpoint(self):
        path = self.root / "completed"
        path.mkdir()
        for name in ("model.pt", "config.json", "history.csv", "predictions.csv"):
            (path / name).write_text("example")
        write_json(path / "result.json", {"run_id": "full", "checkpoint_reload_verified": True, "bn_buffers_unchanged": True})
        artifacts = {p.name: sha256_file(p) for p in path.iterdir()}
        write_json(path / "complete.json", {"identity_sha256": "identity", "run_id": "full", "artifact_sha256": artifacts})
        self.assertIsNotNone(validate_completed(path, "identity", "full"))
        (path / "model.pt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "corrupted"):
            validate_completed(path, "identity", "full")

    def test_incomplete_directory_is_not_complete(self):
        path = self.root / "incomplete"
        path.mkdir()
        (path / "model.pt").write_text("partial")
        self.assertIsNone(validate_completed(path, "identity", "full"))


class SamplerTests(unittest.TestCase):
    def test_one_shot_fills_all_batches_and_is_deterministic(self):
        batches = list(RepeatedBatchSampler(3, 16, 28, 42))
        self.assertEqual(len(batches), 28)
        self.assertTrue(all(len(batch) == 16 for batch in batches))
        flattened = sum(batches, [])
        self.assertEqual(len(flattened), 448)
        self.assertEqual(set(flattened), {0, 1, 2})
        self.assertEqual(batches, list(RepeatedBatchSampler(3, 16, 28, 42)))
        self.assertNotEqual(batches, list(RepeatedBatchSampler(3, 16, 28, 43)))

    def test_every_cycle_is_a_permutation(self):
        flat = sum(list(RepeatedBatchSampler(60, 16, 28, 42)), [])
        for start in range(0, len(flat) - 60 + 1, 60):
            self.assertEqual(sorted(flat[start:start + 60]), list(range(60)))


class NumericalTests(unittest.TestCase):
    @staticmethod
    def factory():
        import torch
        class TinyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(4, 4)
                self.bn = torch.nn.BatchNorm1d(4)
                self.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(4, 3))

            def forward(self, value):
                return self.fc(self.bn(self.backbone(value)).relu())
        return TinyNet()

    def test_two_stage_bn_affine_and_final_checkpoint(self):
        import torch
        from torch.utils.data import TensorDataset
        seed_training(42)
        model = self.factory()
        set_training_stage(model, "head")
        self.assertFalse(model.bn.training)
        self.assertFalse(model.bn.weight.requires_grad)
        self.assertTrue(model.fc[1].weight.requires_grad)
        set_training_stage(model, "finetune")
        self.assertFalse(model.bn.training)
        self.assertTrue(model.bn.weight.requires_grad)
        before = model.bn.weight.detach().clone()
        data = TensorDataset(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [-1., 2., -3., 4.]]), torch.tensor([0, 1, 2]))
        args = SimpleNamespace(smoke_test=True, batch_size=4, num_workers=0, device="cpu")
        config = make_config(args, 3)
        history, evidence = train_fixed_updates(model, data, config, torch.device("cpu"))
        self.assertEqual(evidence["actual_optimizer_steps"], 4)
        self.assertEqual(evidence["actual_image_occurrences"], 16)
        self.assertTrue(evidence["bn_buffers_unchanged"])
        self.assertTrue(evidence["head_stage_backbone_unchanged"])
        self.assertTrue(evidence["finetune_backbone_changed"])
        self.assertNotEqual(evidence["initial_backbone_sha256"], evidence["final_backbone_sha256"])
        self.assertFalse(torch.equal(before, model.bn.weight))
        self.assertEqual([r["stage"] for r in history], ["head", "finetune"])
        with tempfile.TemporaryDirectory() as root:
            restored, evidence = save_reload_final(model, Path(root) / "final.pt", self.factory, torch.device("cpu"))
            self.assertTrue(evidence["checkpoint_reload_verified"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
