"""Backbone + 14-way multi-label head.

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
    def __init__(self, features: nn.Module, num_features: int, num_classes: int = NUM_CLASSES,
                 relu_after_features: bool = False, dropout: float = 0.0):
        super().__init__()
        self.features = features
        self.relu_after_features = relu_after_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_features, num_classes)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> last conv feature map (B, C, h, w). This is the Grad-CAM target layer."""
        f = self.features(x)
        # torchvision DenseNet applies ReLU after its final BatchNorm inside forward();
        # we do it here (not in-place) so the feature map stays differentiable.
        return F.relu(f) if self.relu_after_features else f

    def head(self, fmap: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return self.classifier(self.dropout(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 14). Apply sigmoid (not softmax) to get independent probabilities."""
        return self.head(self.extract(x))


def build_model(backbone: str = "densenet121", pretrained: bool = True, dropout: float = 0.0) -> CXRModel:
    backbone = backbone.lower()
    if backbone == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        return CXRModel(m.features, m.classifier.in_features, relu_after_features=True, dropout=dropout)
    if backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        return CXRModel(features, m.fc.in_features, dropout=dropout)
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        return CXRModel(m.features, m.classifier[1].in_features, dropout=dropout)
    raise ValueError(f"unknown backbone '{backbone}' (use densenet121, resnet50 or efficientnet_b0)")


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[CXRModel, dict]:
    """Load a checkpoint written by train.py. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["backbone"], pretrained=False, dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
