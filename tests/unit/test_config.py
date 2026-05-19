"""Unit tests for :mod:`src.config` (pure Python — always runs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.exceptions import ConfigError

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_from_yaml_resolves_inheritance() -> None:
    """A config with `defaults:` inherits and overrides base values."""
    config = Config.from_yaml(_CONFIG_DIR / "train_mobilenet.yaml")
    assert config.model.architecture == "mobilenet_v2"
    assert config.data.img_size == 224  # inherited from base.yaml
    assert config.data.augmentation.rotation_range == 20


def test_baseline_config_loads() -> None:
    """The baseline config loads and selects the baseline architecture."""
    config = Config.from_yaml(_CONFIG_DIR / "train_baseline.yaml")
    assert config.model.architecture == "baseline_cnn"
    assert config.training.epochs == 30


def test_from_yaml_missing_file_raises() -> None:
    """A missing config file raises ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(_CONFIG_DIR / "does_not_exist.yaml")


def test_validate_rejects_unknown_architecture(tmp_path: Path) -> None:
    """An unsupported architecture name raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data:\n  dataset_path: ./d\nmodel:\n  architecture: resnet999\n"
        "training:\n  epochs: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="architecture"):
        Config.from_yaml(bad)


def test_validate_rejects_bad_val_split(tmp_path: Path) -> None:
    """A val_split outside (0, 1) raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data:\n  dataset_path: ./d\n  val_split: 1.5\n"
        "model:\n  architecture: baseline_cnn\ntraining:\n  epochs: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="val_split"):
        Config.from_yaml(bad)


def test_to_dict_and_save_round_trip(tmp_path: Path) -> None:
    """to_dict() is nested and save() writes a reloadable YAML file."""
    config = Config.from_yaml(_CONFIG_DIR / "train_mobilenet.yaml")
    as_dict = config.to_dict()
    assert as_dict["model"]["architecture"] == "mobilenet_v2"

    out = tmp_path / "saved.yaml"
    config.save(out)
    reloaded = Config.from_yaml(out)
    assert reloaded.model.architecture == config.model.architecture
    assert reloaded.data.batch_size == config.data.batch_size
