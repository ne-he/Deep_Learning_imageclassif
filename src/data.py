"""Data loading: transforms, 3-way split, and an imbalance-aware sampler.

Mirrors the notebook pipeline but is config-driven (no hardcoded paths):
- Train: Resize -> RandomCrop -> flips/rotation/jitter/grayscale -> ImageNet norm.
- Val/Test: Resize -> ImageNet norm (no augmentation — honest evaluation).
- WeightedRandomSampler oversamples minority classes (e.g. ``trash``).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from src.config import Config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(cfg: Config) -> Tuple[T.Compose, T.Compose]:
    """Return ``(train_transform, eval_transform)``."""
    a = cfg.data.augmentation
    s = cfg.data.img_size
    train_tf = T.Compose(
        [
            T.Resize((cfg.data.resize_size, cfg.data.resize_size)),
            T.RandomCrop(s),
            T.RandomHorizontalFlip(p=a.h_flip),
            T.RandomVerticalFlip(p=a.v_flip),
            T.RandomRotation(degrees=a.rotation),
            T.ColorJitter(*a.color_jitter),
            T.RandomGrayscale(p=a.grayscale),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((s, s)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


class TransformedSubset(Dataset):
    """Apply a transform lazily to a ``random_split`` subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test loaders with a deterministic split.

    Returns ``(train_loader, val_loader, test_loader, class_names)``.
    """
    root = Path(cfg.data.dataset_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}. Put 6 class folders there "
            "(cardboard/ glass/ metal/ paper/ plastic/ trash/)."
        )

    full = ImageFolder(root=str(root), transform=None)
    class_names = list(full.class_to_idx.keys())

    n_total = len(full)
    n_test = int(cfg.data.test_split * n_total)
    n_val = int(cfg.data.val_split * n_total)
    n_train = n_total - n_val - n_test

    gen = torch.Generator().manual_seed(cfg.data.seed)
    train_raw, val_raw, test_raw = random_split(full, [n_train, n_val, n_test], generator=gen)

    train_tf, eval_tf = build_transforms(cfg)
    train_ds = TransformedSubset(train_raw, train_tf)
    val_ds = TransformedSubset(val_raw, eval_tf)
    test_ds = TransformedSubset(test_raw, eval_tf)

    # Imbalance handling: oversample minority classes in the training set.
    if cfg.training.use_weighted_sampler:
        targets = [full.targets[i] for i in train_raw.indices]
        counts = Counter(targets)
        class_w = {c: 1.0 / counts[c] for c in counts}
        sample_w = [class_w[t] for t in targets]
        sampler = WeightedRandomSampler(
            sample_w, len(sample_w), replacement=True,
            generator=torch.Generator().manual_seed(cfg.data.seed),
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.data.batch_size, sampler=sampler,
            num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, class_names
