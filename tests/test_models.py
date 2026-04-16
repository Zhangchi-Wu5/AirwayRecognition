"""Tests for model construction and freeze/unfreeze utilities."""
import pytest
import torch
from src.models import build_resnet50, freeze_backbone, unfreeze_all, count_trainable_params


class TestBuildResnet50:
    def test_returns_module(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_output_shape(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3)

    def test_classifier_head_has_dropout_and_linear(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        # model.fc should be Sequential(Dropout, Linear(2048, 3))
        head = model.fc
        assert isinstance(head, torch.nn.Sequential)
        modules = list(head.modules())
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in modules)
        linears = [m for m in modules if isinstance(m, torch.nn.Linear)]
        assert has_dropout
        assert len(linears) == 1
        assert linears[0].out_features == 3


class TestFreezeUnfreeze:
    def test_freeze_backbone_only_trains_head(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        freeze_backbone(model)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        # Only fc layer parameters should be trainable
        for name in trainable:
            assert name.startswith("fc."), f"Unexpected trainable param: {name}"

    def test_unfreeze_all_makes_everything_trainable(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        freeze_backbone(model)
        unfreeze_all(model)
        for name, p in model.named_parameters():
            assert p.requires_grad, f"{name} is still frozen"

    def test_count_trainable_params_decreases_after_freeze(self):
        model = build_resnet50(num_classes=3, pretrained=False)
        before = count_trainable_params(model)
        freeze_backbone(model)
        after = count_trainable_params(model)
        assert after < before
        # fc is (2048 → 3) + bias = 6147; check after is small
        assert after < 10_000
