"""Typed configuration loaded from layered YAML files.

A per-model YAML may declare ``defaults: base.yaml`` to inherit shared
settings; the merged dict is materialized into dataclasses so the rest of the
codebase never touches raw dicts or hardcoded paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml

PathLike = Union[str, Path]

SUPPORTED_ARCHITECTURES = ("resnet50", "efficientnet_b0", "mobilenet_v2", "efficientnet_b1")


@dataclass
class AugConfig:
    h_flip: float = 0.5
    v_flip: float = 0.3
    rotation: int = 20
    color_jitter: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 0.1])
    grayscale: float = 0.1


@dataclass
class DataConfig:
    dataset_path: str
    img_size: int = 224
    resize_size: int = 256
    batch_size: int = 32
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 0
    augmentation: AugConfig = field(default_factory=AugConfig)


@dataclass
class ModelConfig:
    architecture: str = "resnet50"
    num_classes: int = 6
    pretrained: bool = True
    dropout: float = 0.4


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.1
    early_stopping_patience: int = 4
    use_weighted_sampler: bool = True


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    output_dir: str = "./outputs"
    classes: List[str] = field(
        default_factory=lambda: ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    )

    # ------------------------------------------------------------------ load
    @classmethod
    def from_yaml(cls, path: PathLike) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        # single-level inheritance via `defaults:`
        if "defaults" in raw:
            base_path = path.parent / raw.pop("defaults")
            base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
            raw = cls._deep_merge(base_raw, raw)

        data_raw = dict(raw["data"])
        aug_raw = data_raw.pop("augmentation", {})
        cfg = cls(
            data=DataConfig(augmentation=AugConfig(**aug_raw), **data_raw),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainConfig(**raw.get("training", {})),
            output_dir=raw.get("output_dir", "./outputs"),
            classes=raw.get("classes", None) or Config.__dataclass_fields__["classes"].default_factory(),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.model.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{self.model.architecture}'. "
                f"Expected one of {SUPPORTED_ARCHITECTURES}."
            )
        if not 0.0 < self.data.val_split < 1.0:
            raise ValueError(f"val_split must be in (0,1), got {self.data.val_split}")
        if self.data.val_split + self.data.test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")
        if self.model.num_classes <= 1:
            raise ValueError("num_classes must be > 1")

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = Config._deep_merge(out[k], v)
            else:
                out[k] = v
        return out
