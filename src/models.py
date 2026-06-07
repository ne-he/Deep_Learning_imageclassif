"""Model factory with selective fine-tuning.

Three ImageNet-pretrained backbones are adapted for 6-class trash
classification. Only the deepest block(s) + a fresh classifier head are
unfrozen, so general low-level features stay intact while task-specific
features adapt — appropriate for a small (~2.5k image) dataset.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torchvision.models as models

from src.config import ModelConfig


def _set_trainable(model: nn.Module, prefixes: Tuple[str, ...]) -> None:
    """Freeze everything except parameters whose name starts with a prefix."""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(p) for p in prefixes)


def build_resnet50(cfg: ModelConfig) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
    model = models.resnet50(weights=weights)
    _set_trainable(model, ("layer4", "fc"))
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(cfg.dropout),
        nn.Linear(in_feats, 512),
        nn.ReLU(),
        nn.Dropout(cfg.dropout * 0.75),
        nn.Linear(512, cfg.num_classes),
    )
    return model


def build_efficientnet_b0(cfg: ModelConfig) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.efficientnet_b0(weights=weights)
    _set_trainable(model, ("features.7", "features.8", "classifier"))
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_feats, cfg.num_classes))
    return model


def build_efficientnet_b1(cfg: ModelConfig) -> nn.Module:
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.efficientnet_b1(weights=weights)
    _set_trainable(model, ("features.7", "features.8", "classifier"))
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_feats, cfg.num_classes))
    return model


def build_mobilenet_v2(cfg: ModelConfig) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.mobilenet_v2(weights=weights)
    _set_trainable(model, ("features.17", "features.18", "classifier"))
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_feats, cfg.num_classes))
    return model


_BUILDERS = {
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b1": build_efficientnet_b1,
    "mobilenet_v2": build_mobilenet_v2,
}


def build_model(cfg: ModelConfig) -> nn.Module:
    """Dispatch on ``cfg.architecture`` and return an uncompiled model."""
    builder = _BUILDERS.get(cfg.architecture)
    if builder is None:
        raise ValueError(f"Unknown architecture '{cfg.architecture}'. Supported: {sorted(_BUILDERS)}")
    return builder(cfg)


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_gradcam_target_layer(model: nn.Module, architecture: str):
    """Return the last conv layer used for Grad-CAM (Phase 2)."""
    if architecture == "resnet50":
        return model.layer4[-1].conv3
    # efficientnet_* and mobilenet_v2 share this structure
    return model.features[-1][0]
