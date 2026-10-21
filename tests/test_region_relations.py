"""Numerical mechanism checks; no pretrained download or model training."""
import torch
import pytest

from src.models import build_resnet50
from src.region_relations import RegionRelationResNet50


def proposals(n=3):
    inner = torch.zeros(1, 3, 16, 16)
    ring = torch.zeros_like(inner)
    for j, (y, x) in enumerate(((2, 2), (7, 9), (11, 3))):
        inner[0, j, y:y + 2, x:x + 2] = 1
        ring[0, j, y - 1:y + 3, x - 1:x + 3] = 1
        ring[0, j] -= inner[0, j]
    return dict(inner_masks=inner, ring_masks=ring,
                valid=torch.tensor([[j < n for j in range(3)]]),
                quality=torch.tensor([[.8, .6, .9]]),
                geometry=torch.tensor([[[.15, .2, .05, .2, .1, .3],
                                        [.7, .5, .03, .1, .2, .7],
                                        [.3, .8, .08, .3, .3, .4]]]))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(44)
    return RegionRelationResNet50(variant="relations", pretrained=False).eval()


def test_initial_logits_match_ordinary_resnet_and_all_variants(model):
    torch.manual_seed(44)
    baseline = build_resnet50(pretrained=False).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = baseline(images)
        for variant in model.VARIANTS:
            model.variant = variant
            torch.testing.assert_close(model(images, proposals()), expected, rtol=0, atol=0)
    model.variant = "relations"


def test_all_variants_construct_identical_state():
    torch.manual_seed(91)
    first = RegionRelationResNet50(variant="baseline", pretrained=False)
    torch.manual_seed(91)
    second = RegionRelationResNet50(variant="relations", pretrained=False)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())


def test_no_candidate_exact_fallback_and_single_candidate_no_pairs(model):
    features = torch.randn(1, 1024, 4, 4)
    with torch.no_grad():
        model.region_out.bias.fill_(2)
        model.relation_out.bias.fill_(3)
        region, relation, count, pairs = model._regional(features, proposals(0))
        assert not region.any() and not relation.any() and count.item() == pairs.item() == 0
        region, relation, count, pairs = model._regional(features, proposals(1))
        assert region.any() and not relation.any() and count.item() == 1 and pairs.item() == 0
        model.region_out.bias.zero_()
        model.relation_out.bias.zero_()


def test_padding_and_tiny_masks_do_not_create_fake_candidates(model):
    features = torch.randn(1, 1024, 4, 4)
    clean = proposals(1)
    dirty = {key: value.clone() for key, value in clean.items()}
    for key in ("inner_masks", "ring_masks", "quality", "geometry"):
        dirty[key][:, 1:] = float("nan")
    with torch.no_grad():
        for a, b in zip(model._regional(features, clean), model._regional(features, dirty)):
            torch.testing.assert_close(a, b)
        clean["inner_masks"][0, 0] = 0
        clean["inner_masks"][0, 0, 1, 1] = 1  # Area pooling retains a single pixel.
        assert model._regional(features, clean)[2].item() == 1


def test_geometry_control_ignores_descriptors_but_relations_uses_them(model):
    torch.manual_seed(32)
    features = torch.randn(1, 1024, 4, 4)
    with torch.no_grad():
        model.relation_out.weight.normal_()
        model.variant = "relations"
        ordinary = model._regional(features, proposals())
        changed = proposals()
        changed["geometry"][..., :5] *= .25
        altered = model._regional(features, changed)
        torch.testing.assert_close(ordinary[0], altered[0], rtol=0, atol=0)
        assert not torch.allclose(ordinary[1], altered[1])
        model.variant = "no_geometry_relations"
        without_geometry = model._regional(features, proposals())
        altered_without_geometry = model._regional(features, changed)
        torch.testing.assert_close(without_geometry[1], altered_without_geometry[1], rtol=0, atol=0)
        changed_contrast = proposals()
        changed_contrast["geometry"][..., 5] *= .25
        altered_contrast = model._regional(features, changed_contrast)
        assert not torch.allclose(without_geometry[1], altered_contrast[1])
        model.relation_out.weight.zero_()
        model.variant = "relations"


def test_heads_receive_gradients_and_bn_stats_stay_frozen(model):
    model.train()
    model.freeze_backbone()
    assert not model.backbone.conv1.weight.requires_grad
    assert model.backbone.fc[1].weight.requires_grad and model.region_out.weight.requires_grad
    model.unfreeze_all()
    assert model.backbone.bn1.weight.requires_grad
    stats = {name: value.clone() for name, value in model.named_buffers()}
    model(torch.randn(2, 3, 64, 64), {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in proposals().items()}).sum().backward()
    assert model.region_out.weight.grad.abs().sum() > 0
    assert model.relation_out.weight.grad.abs().sum() > 0
    assert all(torch.equal(value, stats[name]) for name, value in model.named_buffers())
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    model.zero_grad(set_to_none=True)
    model.eval()


def test_half_precision_empty_pool_has_finite_output():
    features = torch.randn(1, 1024, 4, 4, dtype=torch.float16)
    pooled, exists = RegionRelationResNet50._pool(features, torch.zeros(1, 3, 16, 16, dtype=torch.float16))
    assert torch.isfinite(pooled).all() and not pooled.any() and not exists.any()
