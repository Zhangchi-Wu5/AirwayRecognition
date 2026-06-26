"""Anatomy-attention ResNet-50: in-model spatial attention + global/attention feature fusion.

Unlike the plain ResNet-50 (which only supports *post-hoc* Grad-CAM), this model
produces a spatial attention map as an internal representation that participates in
classification. The attention map can be regularized during training (see
``src.regularization``) for rotation equivariance and pseudo-feature suppression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention producing a single (B,1,H,W) map in (0,1)."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        a = torch.cat([avg, mx], dim=1)  # B,2,H,W
        return torch.sigmoid(self.conv(a))  # B,1,H,W


class AnatomyAttentionResNet50(nn.Module):
    """ResNet-50 backbone + spatial attention + (global, attention) feature fusion head.

    forward(x) -> logits
    forward(x, return_attn=True) -> (logits, attn_map)  where attn_map is (B,1,7,7)
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3,
                 hires: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        # hires: compute the attention map at layer3 resolution (14x14 for 224 input)
        # instead of layer4 (7x7), for a finer attention map. Module structure is
        # unchanged, so checkpoints remain loadable across both modes (use the same flag).
        self.hires = hires
        self.attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = base.fc.in_features  # 2048
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features * 2, num_classes),  # fused: [global_feat | attention_feat]
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if self.hires:
            feat3 = None
            feat = x
            for i, module in enumerate(self.backbone):
                feat = module(feat)
                if i == 6:           # output of layer3 -> B,1024,14,14
                    feat3 = feat
            feat4 = feat             # B,2048,7,7
            attn = self.attention(feat3)                          # B,1,14,14 (finer)
            feat4_up = F.interpolate(feat4, size=attn.shape[-2:], mode="bilinear", align_corners=False)
            global_feat = self.pool(feat4).flatten(1)             # B,2048
            attn_feat = self.pool(feat4_up * attn).flatten(1)     # B,2048
        else:
            feat4 = self.backbone(x)                              # B,2048,7,7
            attn = self.attention(feat4)                          # B,1,7,7
            global_feat = self.pool(feat4).flatten(1)             # B,2048
            attn_feat = self.pool(feat4 * attn).flatten(1)        # B,2048
        fused = torch.cat([global_feat, attn_feat], dim=1)        # B,4096
        logits = self.fc(fused)
        if return_attn:
            return logits, attn
        return logits

    def gradcam_target_layer(self) -> nn.Module:
        """Last conv block of the backbone (layer4's last Bottleneck) for Grad-CAM."""
        return self.backbone[-1][-1]

    def freeze_backbone(self) -> None:
        """Stage 1: freeze conv backbone; keep attention + fusion head trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.attention.parameters():
            p.requires_grad = True
        for p in self.fc.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def build_attention_resnet50(
    num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3, hires: bool = False
) -> AnatomyAttentionResNet50:
    """Factory mirroring ``build_resnet50`` for the attention variant."""
    return AnatomyAttentionResNet50(
        num_classes=num_classes, pretrained=pretrained, dropout=dropout, hires=hires
    )
