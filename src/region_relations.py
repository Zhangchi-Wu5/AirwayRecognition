"""ResNet plus candidate/interior/surround and directed relation residuals.

This experimental architecture makes no assertion that heuristic dark regions
are anatomical lumens. All arms construct identical tensors and share ordinary
global ResNet initialization. BatchNorm running statistics stay frozen in every
arm; affine parameters are trainable when the backbone is unfrozen.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .models import build_resnet50


class RegionRelationResNet50(nn.Module):
    VARIANTS = ("baseline", "regions", "relations", "no_geometry_relations")

    def __init__(self, variant="baseline", pretrained=True, num_classes=3, dropout=.3, max_candidates=3):
        super().__init__()
        if variant not in self.VARIANTS or max_candidates < 1:
            raise ValueError("invalid variant or max_candidates")
        self.variant, self.max_candidates = variant, max_candidates
        self.backbone = build_resnet50(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
        self.region_encoder = nn.Sequential(nn.Linear(1024 * 3, 64), nn.ReLU())
        self.region_out = nn.Linear(64, num_classes)
        self.relation_encoder = nn.Sequential(nn.Linear(64 * 3 + 7, 64), nn.ReLU())
        self.relation_out = nn.Linear(64, num_classes)
        for head in (self.region_out, self.relation_out):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def freeze_backbone(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = not name.startswith("backbone.") or name.startswith("backbone.fc.")

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    @staticmethod
    def _pool(features, masks):
        b, k, h, w = masks.shape
        weights = F.interpolate(masks.reshape(b * k, 1, h, w), size=features.shape[-2:], mode="area")
        weights = weights.reshape(b, k, -1)
        mass = weights.sum(-1)
        epsilon = max(1e-8, torch.finfo(features.dtype).tiny)
        pooled = torch.einsum("bcs,bks->bkc", features.flatten(2), weights) / mass.clamp_min(epsilon)[..., None]
        return pooled, mass > 0

    def _regional(self, features, candidates):
        b, k = features.shape[0], self.max_candidates
        epsilon = max(1e-8, torch.finfo(features.dtype).tiny)
        def tensor(key, dtype=None):
            return torch.as_tensor(candidates[key], device=features.device, dtype=dtype)
        valid = tensor("valid", torch.bool)
        inner, ring = tensor("inner_masks", features.dtype), tensor("ring_masks", features.dtype)
        quality, geometry = tensor("quality", features.dtype), tensor("geometry", features.dtype)
        if valid.shape != (b, k) or quality.shape != (b, k) or geometry.shape != (b, k, 6):
            raise ValueError("candidate valid/quality/geometry batch or candidate dimensions mismatch")
        if inner.ndim != 4 or inner.shape[:2] != (b, k) or ring.shape != inner.shape:
            raise ValueError("inner/ring masks must have matching shape [batch, max_candidates, H, W]")
        valid = (valid & torch.isfinite(inner).flatten(2).all(-1)
                 & torch.isfinite(ring).flatten(2).all(-1)
                 & torch.isfinite(geometry).all(-1)
                 & torch.isfinite(quality) & (quality > 0))
        # Ignore padded fields even if callers leave arbitrary values in them.
        inner = torch.where(valid[..., None, None], inner, 0).clamp(0, 1)
        ring = torch.where(valid[..., None, None], ring, 0).clamp(0, 1)
        inner_features, inner_exists = self._pool(features, inner)
        ring_features, ring_exists = self._pool(features, ring)
        valid = valid & inner_exists & ring_exists & torch.isfinite(quality) & (quality > 0)
        quality = torch.where(valid, quality, 0).clamp(0, 1)
        geometry = torch.where(valid[..., None], geometry, 0)
        token = self.region_encoder(torch.cat((inner_features, ring_features, inner_features - ring_features), dim=-1))
        token = torch.where(valid[..., None], token, 0)
        regional = (self.region_out(token) * quality[..., None]).sum(1) / quality.sum(1).clamp_min(epsilon)[..., None]
        relation = torch.zeros_like(regional)
        pair_count = valid.sum(1) * (valid.sum(1) - 1)
        if self.variant in ("relations", "no_geometry_relations"):
            source = token[:, :, None].expand(-1, -1, k, -1)
            target = token[:, None, :].expand(-1, k, -1, -1)
            relative = geometry[:, None, :, :] - geometry[:, :, None, :]
            distance = torch.linalg.vector_norm(relative[..., :2], dim=-1, keepdim=True)
            if self.variant == "no_geometry_relations":
                # Same parameter count and feature/contrast relationships, but no
                # spatial geometry descriptors in training OR inference. A deterministic
                # two-candidate geometry swap is learnable and is not this control.
                relative = torch.cat((torch.zeros_like(relative[..., :5]), relative[..., 5:]), dim=-1)
                distance = torch.zeros_like(distance)
            edge = self.relation_encoder(torch.cat((source, target, target - source, relative, distance), -1))
            pairs = valid[:, :, None] & valid[:, None, :] & ~torch.eye(k, device=features.device, dtype=torch.bool)[None]
            pair_weight = quality[:, :, None] * quality[:, None, :] * pairs
            relation = (self.relation_out(edge) * pair_weight[..., None]).sum((1, 2)) / pair_weight.sum((1, 2)).clamp_min(epsilon)[:, None]
        return regional, relation, valid.sum(1), pair_count

    def forward(self, images, candidates, return_details=False):
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        x = net.layer2(net.layer1(x))
        local_features = net.layer3(x)
        global_features = net.avgpool(net.layer4(local_features)).flatten(1)
        global_logits = net.fc(global_features)
        regional = torch.zeros_like(global_logits)
        relation = torch.zeros_like(global_logits)
        count = torch.zeros(images.shape[0], device=images.device, dtype=torch.long)
        pair_count = torch.zeros_like(count)
        if self.variant != "baseline":
            regional, relation, count, pair_count = self._regional(local_features, candidates)
        elif candidates is not None:
            count = torch.as_tensor(candidates["valid"], device=images.device, dtype=torch.bool).sum(1)
        logits = global_logits + regional + relation
        details = dict(global_logits=global_logits, regional_delta=regional, relation_delta=relation,
                       candidate_count=count, pair_count=pair_count)
        return (logits, details) if return_details else logits


def build_region_relation_resnet50(**kwargs):
    return RegionRelationResNet50(**kwargs)
