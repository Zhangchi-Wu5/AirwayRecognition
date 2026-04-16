"""Model construction and parameter freezing utilities."""
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Build a ResNet-50 with a replaced classifier head.

    Original head: Linear(2048, 1000)
    New head: Sequential(Dropout(p=dropout), Linear(2048, num_classes))
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head (model.fc)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    """Mark all parameters as trainable."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
