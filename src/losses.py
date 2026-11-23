"""Loss functions for imbalanced multi-label classification.

All losses take raw logits (B, C) and multi-hot targets (B, C), and treat each class as an
independent binary problem.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017): down-weights easy examples, which are mostly the
    abundant negatives, so the rare positives contribute more to the gradient."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss (Ridnik et al. 2021): separate focusing for positives and negatives,
    plus a probability margin that discards very easy negatives entirely."""

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos, self.gamma_neg, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = targets * (1 - p) ** self.gamma_pos * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * p_neg ** self.gamma_neg * torch.log((1 - p_neg).clamp(min=self.eps))
        return -(loss_pos + loss_neg).mean()


def build_loss(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        assert pos_weight is not None, "weighted_bce needs pos_weight"
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    if name == "asl":
        return AsymmetricLoss()
    raise ValueError(f"unknown loss '{name}' (use bce, weighted_bce, focal or asl)")
