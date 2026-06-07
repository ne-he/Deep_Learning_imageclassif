"""Cross-cutting helpers: seeding, device, JSON, timestamps."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, Path]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return CUDA device if available, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
