"""Grad-CAM (Gradient-weighted Class Activation Mapping) via hooks.

Highlights which image regions drive a prediction. Hooks are removable
(:meth:`GradCAM.remove`) so a model can still be exported (TorchScript/ONNX)
afterwards — fixing the export failure in the original notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Compute Grad-CAM heatmaps for a single target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx=None, device="cpu") -> Tuple[np.ndarray, int, float]:
        """Return ``(heatmap[0..1], predicted_class_idx, confidence)``."""
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # GAP over spatial dims
        cam = (weights * self.activations).sum(dim=1, keepdim=True)      # weighted sum of maps
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)         # normalize 0..1
        prob = float(torch.softmax(logits, dim=1)[0, class_idx].item())
        return cam, class_idx, prob

    def remove(self) -> None:
        """Detach hooks (call before exporting the model)."""
        self._fwd.remove()
        self._bwd.remove()


def denormalize(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Undo ImageNet normalization for display."""
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


def overlay_heatmap(img_tensor: torch.Tensor, cam: np.ndarray, img_size: int = 224,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over the (denormalized) image. Returns ``HxWx3`` [0..1]."""
    import matplotlib.cm as cm
    from PIL import Image

    img = denormalize(img_tensor).permute(1, 2, 0).numpy()
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    cam_resized = np.array(cam_img) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * heatmap, 0, 1)


def save_gradcam_grid(model, target_layer, samples: List[Tuple[torch.Tensor, int]],
                      class_names: List[str], path, device="cpu", img_size: int = 224) -> Path:
    """Save a 2-row grid (original vs Grad-CAM) for one sample per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam = GradCAM(model, target_layer)
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.6))
    for col, (img_t, true_idx) in enumerate(samples):
        heatmap, pred_idx, conf = cam.generate(img_t, device=device)
        axes[0, col].imshow(denormalize(img_t).permute(1, 2, 0).numpy())
        axes[0, col].set_title(f"True: {class_names[true_idx]}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(overlay_heatmap(img_t, heatmap, img_size))
        axes[1, col].set_title(f"Pred: {class_names[pred_idx]}\n({conf*100:.0f}%)", fontsize=9)
        axes[1, col].axis("off")
    cam.remove()
    fig.suptitle("Grad-CAM — one sample per class", fontsize=13)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
