"""DenseNet-121 backbone + 14-way multi-label head.

The model is split into `features` (conv backbone producing a spatial feature map) and
`head` (pool + linear). Keeping them separate lets Grad-CAM take gradients w.r.t. the
feature map directly, without forward/backward hooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .labels import NUM_CLASSES


class CXRModel(nn.Module):
    def __init__(self, features: nn.Module, num_features: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(num_features, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> last conv feature map (B, 1024, 7, 7). This is the Grad-CAM target layer."""
        # torchvision's DenseNet applies ReLU after its final BatchNorm inside forward();
        # we do it here (not in-place) so the feature map stays differentiable.
        return F.relu(self.features(x))

    def head(self, fmap: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 14). Apply sigmoid (not softmax) to get independent probabilities."""
        return self.head(self.extract(x))


def build_model(pretrained: bool = True) -> CXRModel:
    """DenseNet-121 (the CheXNet backbone), starting from ImageNet weights when `pretrained`."""
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    return CXRModel(m.features, m.classifier.in_features)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[CXRModel, dict]:
    """Load a checkpoint written by train.py. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
