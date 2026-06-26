"""Tests for the anatomy-attention model and the training-time regularizers.

All tests run on CPU with synthetic tensors (no dataset / no pretrained download).
"""
import torch

from src.attention import AnatomyAttentionResNet50, build_attention_resnet50
from src.regularization import (
    equivariance_loss,
    pseudo_mask_from_tensor,
    pseudo_suppression_loss,
)
from src.data import IMAGENET_MEAN, IMAGENET_STD


def _model():
    return build_attention_resnet50(num_classes=3, pretrained=False, dropout=0.3)


def test_forward_returns_logits():
    model = _model().eval()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_forward_return_attn_shape():
    model = _model().eval()
    x = torch.randn(2, 3, 224, 224)
    logits, attn = model(x, return_attn=True)
    assert logits.shape == (2, 3)
    assert attn.shape == (2, 1, 7, 7)
    assert float(attn.min()) >= 0.0 and float(attn.max()) <= 1.0  # sigmoid output


def test_freeze_backbone_keeps_attention_and_head_trainable():
    model = _model()
    model.freeze_backbone()
    backbone_grad = [p.requires_grad for p in model.backbone.parameters()]
    assert not any(backbone_grad)
    assert all(p.requires_grad for p in model.attention.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    model.unfreeze_all()
    assert all(p.requires_grad for p in model.parameters())


def test_gradcam_target_layer_exists():
    model = _model()
    layer = model.gradcam_target_layer()
    assert isinstance(layer, torch.nn.Module)


def test_pseudo_suppression_loss_bounds():
    attn = torch.rand(4, 1, 7, 7)
    all_ones = torch.ones(4, 1, 28, 28)
    all_zeros = torch.zeros(4, 1, 28, 28)
    # All energy inside mask -> score 1; none inside -> score 0.
    assert abs(float(pseudo_suppression_loss(attn, all_ones)) - 1.0) < 1e-4
    assert float(pseudo_suppression_loss(attn, all_zeros)) < 1e-4


def test_pseudo_suppression_loss_is_differentiable():
    attn = torch.rand(2, 1, 7, 7, requires_grad=True)
    mask = (torch.rand(2, 1, 28, 28) > 0.5).float()
    loss = pseudo_suppression_loss(attn, mask)
    loss.backward()
    assert attn.grad is not None


def test_equivariance_loss_runs_and_backprops():
    model = _model().train()
    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    _, attn = model(x, return_attn=True)
    loss = equivariance_loss(model, x, attn, max_angle=15.0)
    assert loss.ndim == 0 and float(loss) >= 0.0
    loss.backward()  # gradients should flow through the attention/backbone
    grads = [p.grad is not None for p in model.attention.parameters()]
    assert any(grads)


def test_pseudo_mask_from_tensor_shape_and_values():
    img = torch.randn(3, 224, 224)
    mask = pseudo_mask_from_tensor(img, IMAGENET_MEAN, IMAGENET_STD, size=56)
    assert mask.shape == (1, 56, 56)
    uniq = set(torch.unique(mask).tolist())
    assert uniq.issubset({0.0, 1.0})


def test_hires_attention_is_14x14_and_keys_match():
    lo = build_attention_resnet50(num_classes=3, pretrained=False, hires=False).eval()
    hi = build_attention_resnet50(num_classes=3, pretrained=False, hires=True).eval()
    x = torch.randn(2, 3, 224, 224)
    logits_lo, attn_lo = lo(x, return_attn=True)
    logits_hi, attn_hi = hi(x, return_attn=True)
    assert attn_lo.shape == (2, 1, 7, 7)
    assert attn_hi.shape == (2, 1, 14, 14)      # finer attention map
    assert logits_hi.shape == (2, 3)
    # module structure identical -> checkpoints interchangeable (must use matching --attn-hires)
    assert set(lo.state_dict().keys()) == set(hi.state_dict().keys())


def test_hires_reg_training_step():
    model = build_attention_resnet50(num_classes=3, pretrained=False, hires=True).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 2])
    masks = (torch.rand(2, 1, 56, 56) > 0.7).float()
    crit = torch.nn.CrossEntropyLoss()
    opt.zero_grad()
    logits, attn = model(images, return_attn=True)
    loss = crit(logits, labels)
    loss = loss + 0.1 * pseudo_suppression_loss(attn, masks)
    loss = loss + 0.1 * equivariance_loss(model, images, attn, max_angle=15.0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_crop_black_border_removes_frame():
    import numpy as np
    from PIL import Image
    from src.data import CropBlackBorder

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[20:80, 30:70] = 200  # bright content surrounded by an edge-connected black frame
    cropped = CropBlackBorder()(Image.fromarray(arr))
    w, h = cropped.size
    assert w < 95 and h < 95          # frame removed
    assert w > 15 and h > 30          # content preserved


def test_pseudo_mask_kind_highlight_subset_of_combined():
    img = torch.randn(3, 224, 224)
    m_comb = pseudo_mask_from_tensor(img, IMAGENET_MEAN, IMAGENET_STD, size=56, kind="combined")
    m_hi = pseudo_mask_from_tensor(img, IMAGENET_MEAN, IMAGENET_STD, size=56, kind="specular_highlight")
    assert m_comb.shape == (1, 56, 56) and m_hi.shape == (1, 56, 56)
    # highlight mask must be a subset of the combined mask (combined = dark ∪ highlight)
    assert bool(torch.all(m_hi.bool() <= m_comb.bool()))


def test_reg_training_step():
    """A full CE + pseudo + equivariance step updates parameters without error."""
    model = _model().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 2])
    masks = (torch.rand(2, 1, 56, 56) > 0.7).float()
    criterion = torch.nn.CrossEntropyLoss()

    opt.zero_grad()
    logits, attn = model(images, return_attn=True)
    loss = criterion(logits, labels)
    loss = loss + 0.1 * pseudo_suppression_loss(attn, masks)
    loss = loss + 0.1 * equivariance_loss(model, images, attn, max_angle=15.0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
