"""Generate Grad-CAM heatmaps from a trained checkpoint.

Usage:
    python scripts/gradcam.py --config configs/resnet50.yaml
    python scripts/gradcam.py --config configs/resnet50.yaml \
        --checkpoint outputs/resnet50/best_model.pth --output outputs/resnet50/gradcam.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data import get_dataloaders
from src.gradcam import save_gradcam_grid
from src.models import build_model, get_gradcam_target_layer
from src.utils import get_device, set_seed


def collect_one_per_class(dataset, num_classes: int):
    """Return ``[(img_tensor, label), ...]`` — first sample found per class."""
    found = {}
    for img, label in dataset:
        label = int(label)
        if label not in found:
            found[label] = (img, label)
        if len(found) == num_classes:
            break
    return [found[i] for i in sorted(found)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Defaults to outputs/<arch>/best_model.pth")
    ap.add_argument("--output", default=None, help="Defaults to outputs/<arch>/gradcam.png")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.data.seed)
    device = get_device()

    arch = cfg.model.architecture
    ckpt = Path(args.checkpoint or Path(cfg.output_dir) / arch / "best_model.pth")
    out = Path(args.output or Path(cfg.output_dir) / arch / "gradcam.png")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train first (scripts/train.py).")

    # Build model + load weights
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {arch} from {ckpt}")

    # Sample one image per class from the test split
    _, _, test_loader, class_names = get_dataloaders(cfg)
    samples = collect_one_per_class(test_loader.dataset, cfg.model.num_classes)

    target_layer = get_gradcam_target_layer(model, arch)
    path = save_gradcam_grid(model, target_layer, samples, class_names, out,
                             device=device, img_size=cfg.data.img_size)
    print(f"Saved Grad-CAM grid to {path}")


if __name__ == "__main__":
    main()
