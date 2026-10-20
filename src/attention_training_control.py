"""Opt-in training control; the legacy attention model keeps its default behavior."""
from __future__ import annotations

import torch

from src.attention import AnatomyAttentionResNet50


class ControlledAttentionResNet50(AnatomyAttentionResNet50):
    """Start both arms at A=0.5; either learn or permanently freeze spatial weights.

    The fixed arm retains both feature branches and their original fusion head.
    Its attention branch is 0.5 times the global feature branch, so it is not a
    plain ResNet baseline. The contrast includes learned spatial and mean effects.
    """

    def __init__(self, attention_policy: str, num_classes: int = 3,
                 pretrained: bool = True, dropout: float = .3):
        if attention_policy not in {"fixed", "learned"}:
            raise ValueError("attention_policy must be 'fixed' or 'learned'")
        super().__init__(num_classes=num_classes, pretrained=pretrained, dropout=dropout, hires=False)
        self.attention_policy = attention_policy
        # Both arms consume the same ordinary initialization RNG before zeroing.
        with torch.no_grad():
            self.attention.conv.weight.zero_()
        self._apply_attention_policy()

    def _apply_attention_policy(self) -> None:
        for parameter in self.attention.parameters():
            parameter.requires_grad_(self.attention_policy == "learned")

    def freeze_backbone(self) -> None:
        super().freeze_backbone()
        self._apply_attention_policy()

    def unfreeze_all(self) -> None:
        super().unfreeze_all()
        self._apply_attention_policy()

    def assert_attention_policy(self) -> dict:
        parameters = list(self.attention.parameters())
        nonzero = sum(int(torch.count_nonzero(parameter.detach()).item()) for parameter in parameters)
        trainable = any(parameter.requires_grad for parameter in parameters)
        if self.attention_policy == "fixed" and (nonzero or trainable):
            raise RuntimeError("Fixed attention must retain zero weights and remain frozen in both stages")
        if self.attention_policy == "learned" and not all(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("Learned attention must remain trainable")
        return {"policy": self.attention_policy, "count_nonzero": nonzero, "requires_grad": trainable}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._apply_attention_policy()
        self.assert_attention_policy()
        return result


def build_controlled_attention_resnet50(attention_policy: str, num_classes: int = 3,
                                       pretrained: bool = True, dropout: float = .3):
    return ControlledAttentionResNet50(
        attention_policy=attention_policy, num_classes=num_classes, pretrained=pretrained, dropout=dropout,
    )
