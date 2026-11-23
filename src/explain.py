"""Grad-CAM and Grad-CAM++ for the multi-label model, plus overlay rendering.

Implemented directly (no hooks) using CXRModel.extract() / head():
    A      = feature map from the last conv block          (C, h, w)
    y_c    = logit of class c
    dy/dA  = gradient of that logit w.r.t. A
Grad-CAM:    w_k = mean over (h, w) of dy/dA_k;  CAM = ReLU(sum_k w_k * A_k)
Grad-CAM++:  w_k = sum over (h, w) of alpha_k * ReLU(dy/dA_k), with alpha giving
             more weight to pixels that matter, which helps when a class appears in
             several places (e.g. bilateral effusion, multiple nodules).

Because the task is multi-label, each class gets its own heatmap from its own logit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_transforms, load_image
from .model import CXRModel

METHODS = ("gradcam", "gradcam++")


class CAMExplainer:
    def __init__(self, model: CXRModel, img_size: int = 224, method: str = "gradcam", device="cpu"):
        assert method in METHODS, f"method must be one of {METHODS}"
        self.model = model.eval()
        self.img_size = img_size
        self.method = method
        self.device = torch.device(device)
        self.transform = build_transforms(img_size, train=False)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

    def explain(self, x: torch.Tensor, class_indices) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """x: (1, 3, H, W). Returns (probabilities (14,), {class_idx: CAM (H, W) in [0, 1]})."""
        with torch.no_grad():
            fmap = self.model.extract(x)
        fmap = fmap.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model.head(fmap)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
            cams = {}
            for c in class_indices:
                grads = torch.autograd.grad(logits[0, c], fmap, retain_graph=True)[0]
                cams[int(c)] = self._cam(fmap.detach(), grads, x.shape[-2:])
        return probs, cams

    def explain_image(self, img: Image.Image, class_indices) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        return self.explain(self.preprocess(img), class_indices)

    def _cam(self, A: torch.Tensor, g: torch.Tensor, size) -> np.ndarray:
        if self.method == "gradcam":
            weights = g.mean(dim=(2, 3), keepdim=True)
        else:  # Grad-CAM++ (Chattopadhay et al. 2018), closed form for an exp-free score
            g2, g3 = g ** 2, g ** 3
            sum_A = A.sum(dim=(2, 3), keepdim=True)
            alpha = g2 / (2 * g2 + sum_A * g3 + 1e-7)
            alpha = torch.where(g != 0, alpha, torch.zeros_like(alpha))
            weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()


# --------------------------------------------------------------------------- rendering
def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a [0,1] CAM to (width, height) of the original image."""
    im = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> Image.Image:
    """Blend a heatmap onto the original X-ray (CAM is resized to the image's resolution)."""
    from matplotlib import colormaps

    base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    cam = resize_cam(cam, img.size)
    heat = colormaps[colormap](cam)[..., :3]
    # Scale opacity by CAM strength so cold regions show the X-ray clearly.
    a = alpha * cam[..., None]
    out = (1 - a) * base + a * heat
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))


def explain_path(model: CXRModel, path: str, class_indices, img_size=224, method="gradcam", device="cpu"):
    """Convenience: path -> (PIL image, probs, {class_idx: overlay image})."""
    img = load_image(path)
    exp = CAMExplainer(model, img_size, method, device)
    probs, cams = exp.explain_image(img, class_indices)
    return img, probs, {c: overlay(img, cam) for c, cam in cams.items()}
