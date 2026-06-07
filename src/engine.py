"""Training engine: loops, early stopping, and a Trainer orchestrator.

Recipe (from the PyTorch notebook, refactored): AdamW + weight decay,
CosineAnnealingLR, CrossEntropy with label smoothing, early stopping on
val_loss, best checkpoint kept by val accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.utils import ensure_dir


class EarlyStopping:
    def __init__(self, patience: int = 4, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return ``(loss, acc, preds, labels, probs)`` over a loader."""
    model.eval()
    total_loss = correct = 0
    preds, labels_all, probs_all = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        p = torch.softmax(out, dim=1)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        preds.extend(pred.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(p.cpu().numpy())
    n = len(loader.dataset)
    return total_loss / n, correct / n, np.array(preds), np.array(labels_all), np.array(probs_all)


class Trainer:
    """Coordinates a full training run for one model."""

    def __init__(self, model: nn.Module, cfg: Config, device, run_dir: Optional[Path] = None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_dir = Path(run_dir) if run_dir else Path(cfg.output_dir) / cfg.model.architecture
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    def _make_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        name = self.cfg.training.optimizer.lower()
        if name == "adamw":
            return optim.AdamW(params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        if name == "adam":
            return optim.Adam(params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        if name == "sgd":
            return optim.SGD(params, lr=self.cfg.training.lr, momentum=0.9,
                             weight_decay=self.cfg.training.weight_decay)
        raise ValueError(f"Unknown optimizer '{name}'")

    def train(self, train_loader, val_loader) -> Dict[str, list]:
        ensure_dir(self.run_dir)
        optimizer = self._make_optimizer()
        scheduler = None
        if self.cfg.training.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs)
        es = EarlyStopping(patience=self.cfg.training.early_stopping_patience)
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_acc = 0.0
        ckpt = self.run_dir / "best_model.pth"

        name = self.cfg.model.architecture
        print(f"\n{'='*60}\n  Training {name} | {self.cfg.training.epochs} epochs | lr={self.cfg.training.lr}\n{'='*60}")
        for epoch in range(1, self.cfg.training.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(self.model, train_loader, self.criterion, optimizer, self.device)
            vl_loss, vl_acc, *_ = evaluate(self.model, val_loader, self.criterion, self.device)
            if scheduler:
                scheduler.step()
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)
            flag = ""
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(self.model.state_dict(), ckpt)
                flag = " [BEST]"
            print(f"  Epoch {epoch:02d}/{self.cfg.training.epochs}  "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}{flag}")
            es.step(vl_loss)
            if es.stop:
                print(f"  Early stopping at epoch {epoch}")
                break

        # restore best weights
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"  Best Val Acc: {best_acc:.4f}")
        return history
