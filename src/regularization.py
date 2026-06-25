"""Training-time regularizers for the anatomy-attention model.

Two losses, both requiring **no manual annotation**:

* ``pseudo_suppression_loss`` — penalizes attention energy that falls on
  rule-detected pseudo-feature regions (scope black border, specular highlight).
  This is the trainable counterpart of the post-hoc pseudo-feature audit.
* ``equivariance_loss`` — enforces that the attention map rotates consistently
  with the input under direction-preserving rotations (SEAM-style self-supervision),
  encouraging the model to lock onto anatomical structure rather than view angle.
"""
import torch
import torch.nn.functional as F


def _rotate(x: torch.Tensor, theta_deg: torch.Tensor) -> torch.Tensor:
    """Differentiably rotate a (B,C,H,W) tensor by a single angle (0-dim tensor, degrees)."""
    n = x.size(0)
    theta = torch.deg2rad(theta_deg).to(x.device)
    cos, sin = torch.cos(theta), torch.sin(theta)
    mat = x.new_zeros(n, 2, 3)
    mat[:, 0, 0] = cos
    mat[:, 0, 1] = -sin
    mat[:, 1, 0] = sin
    mat[:, 1, 1] = cos
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def pseudo_suppression_loss(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fraction of attention energy landing inside the pseudo-feature mask (lower is better).

    attn: (B,1,h,w) in (0,1); mask: (B,1,H,W) float in {0,1}. Differentiable.
    """
    a = F.interpolate(attn, size=mask.shape[-2:], mode="bilinear", align_corners=False).clamp(min=0)
    num = (a * mask).flatten(1).sum(dim=1)
    den = a.flatten(1).sum(dim=1).clamp(min=1e-6)
    return (num / den).mean()


def equivariance_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    attn: torch.Tensor,
    max_angle: float = 15.0,
    work: int = 56,
) -> torch.Tensor:
    """Attention rotation-consistency: attn(rot(x)) should equal rot(attn(x)).

    ``attn`` is the attention map already computed for ``images`` (reused to avoid an
    extra forward pass). One random angle in [-max_angle, max_angle] is applied per batch.
    """
    device = images.device
    theta = (torch.rand((), device=device) * 2 - 1) * max_angle  # 0-dim tensor, degrees
    x_rot = _rotate(images, theta)
    _, attn_rot = model(x_rot, return_attn=True)

    a = F.interpolate(attn, size=(work, work), mode="bilinear", align_corners=False)
    a_rot = F.interpolate(attn_rot, size=(work, work), mode="bilinear", align_corners=False)
    a_target = _rotate(a, theta)
    valid = (_rotate(torch.ones_like(a), theta) > 0.99).float()  # ignore corners rotated out
    diff = (a_rot - a_target) ** 2 * valid
    denom = valid.flatten(1).sum(dim=1).clamp(min=1e-6)
    return (diff.flatten(1).sum(dim=1) / denom).mean()


def pseudo_mask_from_tensor(
    img_tensor: torch.Tensor, mean, std, size: int = 56, kind: str = "combined"
) -> torch.Tensor:
    """Build a (1,size,size) pseudo-feature mask from a *normalized* (3,H,W) image tensor.

    ``kind`` selects which mask to return: ``"combined"`` (black border + highlight),
    ``"specular_highlight"`` (highlights only — use this when the black border is
    already removed by ``CropBlackBorder`` in preprocessing), or ``"dark_border"``.

    The mask is computed on the de-normalized image at low resolution so it aligns with
    exactly what the model sees (post-augmentation) and stays cheap inside the dataloader.
    """
    import numpy as np
    from PIL import Image
    from src.viz import build_pseudo_feature_masks

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    denorm = (img_tensor.detach().cpu() * std_t + mean_t).clamp(0, 1)
    arr = (denorm.permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr).resize((size, size))
    masks = build_pseudo_feature_masks(pil)
    m = masks[kind].astype("float32")
    return torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0)
