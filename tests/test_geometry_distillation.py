import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torchvision import models

from src.geometry_distillation import (VARIANTS, GeometryImageDataset,
    build_resnet18_student, cache_predictions, file_hash, joint_geometry,
    relation_loss, validate_cache)
from src.renew_data import get_geometry_transforms, get_transforms


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def image_row(tmp_path, name="1lt.png"):
    path = tmp_path / name
    image = np.random.default_rng(int(name[0])).integers(0, 255, (60, 100, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)
    return {"path": str(path), "image_name": name, "image_sha256": file_hash(path), "label_id": 0}


def teacher(image):
    return np.tile(np.linspace(.1, .9, 32, dtype=np.float32), (24, 1))


def test_exact_rgb_geometry_and_padding_mask():
    image = Image.fromarray(np.random.default_rng(2).integers(0, 255, (80, 160, 3), dtype=np.uint8))
    gradient = np.tile(np.linspace(0, 1, 160, dtype=np.float32), (80, 1))
    for angle in (-15, 0, 15):
        rgb, maps, mask = joint_geometry(image, np.stack([gradient] * 5), train=True, angle=angle)
        expected = get_geometry_transforms("full_frame", train=True, angle=angle)(image)
        assert np.array_equal(np.asarray(rgb), np.asarray(expected))
        assert mask.shape == (14, 14) and 0 < mask.sum() < 196
        assert not mask[0].any() and not mask[-1].any()
        assert torch.isfinite(maps).all()
    _, maps, mask = joint_geometry(image, np.stack([gradient] * 5), train=False)
    assert maps[0, 7, 2] < maps[0, 7, 11]


def test_dataset_rgb_rng_identical_with_and_without_cache(tmp_path):
    row = image_row(tmp_path)
    cache = tmp_path / "cache"
    cache_predictions([row], cache, teacher, {"stub": 1})
    for is_train in (False, True):
        torch.manual_seed(42)
        a = GeometryImageDataset([row], train=is_train)[0]
        torch.manual_seed(42)
        b = GeometryImageDataset([row], cache, train=is_train)[0]
        torch.manual_seed(42)
        expected = get_transforms("full_frame", train=is_train)(Image.open(row["path"]).convert("RGB"))
        assert torch.equal(a["image"], b["image"]) and torch.equal(a["image"], expected)
        assert not a["valid"].any() and b["valid"].any()
        assert b["geometry"].shape == (5, 14, 14)


def test_cache_is_immutable_per_source_and_expansion_keeps_subset_identity(tmp_path):
    a, b = image_row(tmp_path), image_row(tmp_path, "2lt.png")
    cache = tmp_path / "cache"
    initial = cache_predictions([a], cache, teacher, {"stub": 1})
    cache_predictions([a, b], cache, teacher, {"stub": 1})
    assert validate_cache(cache, [a]) == initial
    with pytest.raises(ValueError, match="different teacher"):
        cache_predictions([a], cache, teacher, {"stub": 2})
    file = cache / f"{a['image_sha256']}.npz"
    file.write_bytes(file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        validate_cache(cache, [a])


def test_constant_teacher_rejected(tmp_path):
    row = image_row(tmp_path)
    with pytest.raises(ValueError, match="constant"):
        cache_predictions([row], tmp_path / "cache", lambda image: np.zeros((20, 20)), {"stub": 1})


def test_private_qa_contact_and_metrics(tmp_path):
    from scripts.prepare_endodac_teacher import write_qa
    row = image_row(tmp_path)
    cache = tmp_path / "cache"
    cache_predictions([row], cache, teacher, {"stub": 1})
    path = Path(write_qa([row], cache))
    assert path.is_file() and Image.open(path).size == (990, 174)
    stats = json.loads((cache / "QA_STATS.json").read_text())
    assert stats["records"][0]["selection_rate"] == 1
    assert stats["contact_sha256"] == file_hash(path)


def test_loss_order_stability_and_deterministic_shuffling():
    base = torch.arange(196, dtype=torch.float32).reshape(1, 14, 14) / 195
    geometry = base[:, None].expand(1, 5, 14, 14).clone()
    valid = torch.ones(1, 14, 14, dtype=torch.bool)
    correct, stats = relation_loss(base * 5, geometry, valid)
    incorrect, _ = relation_loss(-base * 5, geometry, valid)
    assert correct < incorrect and stats["selected_pairs"] > 0
    # Reverse one perturbation: all ordinary pairs survive, none are stable.
    geometry[:, 1] = -geometry[:, 1]
    plain, pstats = relation_loss(base, geometry, valid)
    stable, sstats = relation_loss(base.requires_grad_(), geometry, valid, "stable")
    assert pstats["selected_pairs"] > 0 and sstats["selected_pairs"] == 0
    assert stable.item() == 0 and stable.requires_grad
    stable.backward()
    assert torch.isfinite(base.grad).all()
    # Random target maps contain both signs, so permutation can destroy relations.
    geometry = torch.rand(1, 1, 14, 14, generator=torch.Generator().manual_seed(2)).expand(1, 5, 14, 14)
    c, cs = relation_loss(base, geometry, valid, "stable")
    d, ds = relation_loss(base, geometry, valid, "shuffled", [123])
    again, again_stats = relation_loss(base, geometry, valid, "shuffled", [123])
    assert ds["selected_pairs"] == cs["selected_pairs"] and ds["shuffled_changed_pairs"] > 0
    assert torch.equal(d, again) and ds == again_stats


def test_empty_mask_is_differentiable_zero():
    prediction = torch.zeros(2, 1, 14, 14, requires_grad=True)
    loss, stats = relation_loss(prediction, torch.zeros(2, 5, 14, 14), torch.zeros(2, 14, 14, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and stats["selected_pairs"] == 0 and torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_student_classification_initialization_and_inference_match_standard():
    torch.manual_seed(42)
    expected = models.resnet18(weights=None)
    expected.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(512, 3))
    expected_rng = torch.rand(3)
    torch.manual_seed(42)
    actual = build_resnet18_student(pretrained=False)
    assert torch.equal(expected_rng, torch.rand(3))
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    expected.eval(); actual.eval()
    image = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        logits, grid = actual(image, return_geometry=True)
        assert torch.equal(logits, expected(image))
        assert grid.shape == (1, 1, 14, 14)


def test_teacher_loader_requires_complete_state(monkeypatch, tmp_path):
    from scripts import prepare_endodac_teacher as script
    repo = tmp_path / "teacher"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "stub.py").write_text("# fixture")
    monkeypatch.setattr(script.subprocess, "check_output", lambda args, **kwargs:
        script.PINNED_COMMIT if "rev-parse" in args else "")
    class TinyTeacher(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs["pretrained_path"] is None
            self.scale = nn.Parameter(torch.ones(1))
        def forward(self, x):
            return {("disp", 0): x[:, :1] * self.scale}
    monkeypatch.setattr(script.importlib, "import_module", lambda name: SimpleNamespace(endodac=TinyTeacher))
    weights = tmp_path / "weights.pt"
    torch.save({"scale": torch.ones(1), "height": 32, "width": 48}, weights)
    predict, identity = script.load_teacher(repo, weights, None, "cpu")
    assert identity["strict_complete_state"]
    assert predict(Image.new("RGB", (40, 20), "white")).shape == (32, 48)
    torch.save({"height": 32, "width": 48}, weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        script.load_teacher(repo, weights, None, "cpu")
