"""Synthetic input, patient isolation, and full-frame geometry checks."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageFile

from src.renew_data import (Letterbox224, audit_dataset, get_geometry_transforms,
                            get_transforms, make_patient_folds)


class RenewDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_cohort(self, patients=31):
        for patient in range(1, patients + 1):
            for label_id, label in enumerate(("lt", "yz", "zz")):
                # Unique exact byte and pixel content for all 93 fixture images.
                Image.new("RGB", (32, 18), (patient, label_id * 80, 120)).save(
                    self.root / f"{patient:010d}{label}.png")

    def test_first_level_audit_preserves_ids_ignores_failed_extraction_directory(self):
        self.make_cohort(2)
        nested = self.root / "old_failed_extraction"
        nested.mkdir()
        (nested / "broken.png").write_bytes(b"not a PNG")
        (self.root / "archive.rar").write_bytes(b"not opened")
        rows, meta = audit_dataset(self.root, strict=False)
        self.assertEqual((meta["image_count"], meta["patient_count"]), (6, 2))
        self.assertEqual(rows[0]["patient_id"], "0000000001")
        self.assertEqual(rows[0]["patient_key"], "1")
        self.assertEqual(meta["class_counts"], {"lt": 2, "yz": 2, "zz": 2})
        self.assertEqual(meta["ignored_directories"], 1)
        self.assertEqual(meta["ignored_non_png_files"], 1)
        self.assertTrue(all((r["width"], r["height"]) == (32, 18) for r in rows))

    def test_formal_fingerprint_and_size_are_locked(self):
        self.make_cohort(31)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            audit_dataset(self.root)
        (self.root / "0000000031zz.png").unlink()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            audit_dataset(self.root, strict=False)

    def test_normalized_patient_class_collision_is_rejected(self):
        self.make_cohort(2)
        Image.new("RGB", (32, 18), (240, 3, 9)).save(self.root / "1lt.png")
        with self.assertRaisesRegex(ValueError, "Duplicate normalized patient/class"):
            audit_dataset(self.root, strict=False)

    def test_inconsistent_id_spellings_cannot_split_one_patient(self):
        self.make_cohort(2)
        (self.root / "0000000001zz.png").rename(self.root / "1zz.png")
        with self.assertRaisesRegex(ValueError, "Inconsistent patient ID"):
            audit_dataset(self.root, strict=False)

    def test_duplicate_bytes_and_invalid_names_are_rejected(self):
        self.make_cohort(2)
        source = self.root / "0000000001lt.png"
        (self.root / "0000000002lt.png").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate image bytes"):
            audit_dataset(self.root, strict=False)
        (self.root / "bad.png").write_bytes(source.read_bytes())
        (self.root / "0000000002lt.png").unlink()
        with self.assertRaisesRegex(ValueError, "Invalid PNG image name"):
            audit_dataset(self.root, strict=False)

    def test_decode_rejects_corruption_and_restores_pillow_setting(self):
        self.make_cohort(2)
        bad = self.root / "0000000001lt.png"
        bad.write_bytes(bad.read_bytes()[:45])
        with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            with self.assertRaisesRegex(ValueError, "Image decode failed"):
                audit_dataset(self.root, strict=False)
            self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)

    def test_png_extension_with_other_format_is_rejected(self):
        self.make_cohort(2)
        Image.new("RGB", (32, 18), (240, 5, 1)).save(self.root / "0000000001lt.png", format="JPEG")
        with self.assertRaisesRegex(ValueError, "does not contain PNG"):
            audit_dataset(self.root, strict=False)

    def test_five_folds_have_no_patient_leakage_and_one_eval_per_image(self):
        self.make_cohort()
        rows, _ = audit_dataset(self.root, strict=False)
        folds = make_patient_folds(rows)
        self.assertEqual(folds, make_patient_folds(rows))
        self.assertEqual([len(f["eval_patient_keys"]) for f in folds], [7, 6, 6, 6, 6])
        self.assertEqual([len(f["train_patient_keys"]) for f in folds], [24, 25, 25, 25, 25])
        counts = Counter()
        for fold in folds:
            self.assertFalse(set(fold["train_patient_keys"]) & set(fold["eval_patient_keys"]))
            self.assertEqual(len(fold["train"]) + len(fold["eval"]), 93)
            self.assertEqual(len(fold["eval"]), 3 * len(fold["eval_patient_keys"]))
            counts.update(row["image_name"] for row in fold["eval"])
        self.assertEqual(counts, Counter({r["image_name"]: 1 for r in rows}))
        bad_rows = [dict(r) for r in rows]
        bad_rows[0]["patient_key"] = "99999"
        with self.assertRaisesRegex(ValueError, "grouping key"):
            make_patient_folds(bad_rows)


class GeometryTests(unittest.TestCase):
    @staticmethod
    def edge_image(width=1600, height=900):
        image = Image.new("RGB", (width, height), (80, 80, 80))
        image.paste((255, 0, 0), (0, 0, 80, height))
        image.paste((0, 255, 0), (width - 80, 0, width, height))
        image.paste((0, 0, 255), (80, 0, width - 80, 80))
        image.paste((255, 255, 0), (80, height - 80, width - 80, height))
        return image

    def test_letterbox_preserves_all_four_edges_and_aspect(self):
        image = Letterbox224()(self.edge_image())
        self.assertEqual(image.size, (224, 224))
        # A 16:9 image occupies 224x126, centered with 49 padding rows each side.
        self.assertEqual(image.getpixel((0, 112)), (255, 0, 0))
        self.assertEqual(image.getpixel((223, 112)), (0, 255, 0))
        self.assertEqual(image.getpixel((112, 49)), (0, 0, 255))
        self.assertEqual(image.getpixel((112, 174)), (255, 255, 0))
        self.assertEqual(image.getpixel((112, 48)), (0, 0, 0))
        self.assertEqual(image.getpixel((112, 175)), (0, 0, 0))
        self.assertEqual(Letterbox224()(Image.new("RGB", (1, 400))).size, (224, 224))

    def test_rotation_expands_and_preserves_edge_markers(self):
        import numpy as np
        from torchvision import transforms
        operations = get_geometry_transforms("full_frame", train=True).transforms
        rotation = next(op for op in operations if isinstance(op, transforms.RandomRotation))
        self.assertTrue(rotation.expand)
        image = get_geometry_transforms("full_frame", train=True, angle=15)(self.edge_image())
        data = np.asarray(image)
        self.assertEqual(image.size, (224, 224))
        for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)):
            self.assertGreater(int(np.all(data == color, axis=2).sum()), 10)

    def test_original_pipeline_is_unchanged_for_both_train_and_eval(self):
        import torch
        from src.data import get_train_transforms, get_eval_transforms
        image = self.edge_image()
        for train, factory in ((True, get_train_transforms), (False, get_eval_transforms)):
            torch.manual_seed(42)
            expected = factory(False)(image)
            torch.manual_seed(42)
            actual = get_transforms("original", train=train)(image)
            self.assertTrue(torch.equal(expected, actual))

    def test_full_frame_tensor_and_invalid_parameters(self):
        import torch
        for train in (True, False):
            result = get_transforms("full_frame", train=train)(self.edge_image())
            self.assertEqual(tuple(result.shape), (3, 224, 224))
            self.assertTrue(bool(torch.isfinite(result).all()))
        with self.assertRaises(ValueError):
            get_transforms("unknown")
        with self.assertRaises(ValueError):
            get_geometry_transforms("full_frame", train=False, angle=15)
        with self.assertRaises(ValueError):
            Letterbox224(0)


if __name__ == "__main__":
    unittest.main()
