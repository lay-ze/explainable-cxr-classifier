"""Loss functions for imbalanced multi-label classification.

All losses take raw logits (B, C) and multi-hot targets (B, C), and treat each class as an
independent binary problem. Three are compared: plain BCE, weighted BCE and focal loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017): down-weights easy examples, which are mostly the
    abundant negatives, so the rare positives contribute more to the gradient."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def build_loss(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        assert pos_weight is not None, "weighted_bce needs pos_weight"
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    raise ValueError(f"unknown loss '{name}' (use bce, weighted_bce or focal)")
