"""Train a single model from a YAML config.

Usage:
    python scripts/train.py --config configs/resnet50.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data import get_dataloaders
from src.engine import Trainer, evaluate
from src.metrics import compute_metrics, print_report, save_confusion_matrix
from src.models import build_model, count_params
from src.utils import get_device, save_json, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a model YAML config")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.data.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    print(f"Classes: {class_names}")

    model = build_model(cfg.model)
    total, trainable = count_params(model)
    print(f"{cfg.model.architecture}: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")

    trainer = Trainer(model, cfg, device)
    history = trainer.train(train_loader, val_loader)

    # Final test-set evaluation
    _, test_acc, preds, labels, probs = evaluate(model, test_loader, trainer.criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}\n")
    print_report(labels, preds, class_names)

    metrics = compute_metrics(labels, preds, probs, class_names)
    save_json({"history": history, "metrics": metrics}, trainer.run_dir / "results.json")
    save_confusion_matrix(labels, preds, class_names, trainer.run_dir / "confusion_matrix.png")
    print(f"\nArtifacts saved to {trainer.run_dir}")


if __name__ == "__main__":
    main()
