"""Train all backbones, build a soft-voting ensemble, and print a summary.

Usage:
    python scripts/compare.py --configs configs/resnet50.yaml \
        configs/efficientnet_b0.yaml configs/mobilenet_v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data import get_dataloaders
from src.engine import Trainer, evaluate
from src.ensemble import ensemble_predictions
from src.metrics import compute_metrics
from src.models import build_model, count_params
from src.utils import get_device, save_json, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="Model YAML configs to compare")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    results = {}      # name -> {'preds','labels','probs'}
    metrics_all = {}  # name -> metric dict
    out_dir = None

    for cfg_path in args.configs:
        cfg = Config.from_yaml(cfg_path)
        set_seed(cfg.data.seed)  # identical split across models => fair comparison
        out_dir = Path(cfg.output_dir)

        train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
        model = build_model(cfg.model)
        total, trainable = count_params(model)
        name = cfg.model.architecture
        print(f"\n### {name}  (total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M)")

        trainer = Trainer(model, cfg, device)
        trainer.train(train_loader, val_loader)

        _, _, preds, labels, probs = evaluate(model, test_loader, trainer.criterion, device)
        results[name] = {"preds": preds, "labels": labels, "probs": probs}
        metrics_all[name] = compute_metrics(labels, preds, probs, class_names)

    # Ensemble (soft voting)
    if len(results) > 1:
        ens = ensemble_predictions(results)
        metrics_all["ensemble"] = compute_metrics(ens["labels"], ens["preds"], ens["probs"], class_names)

    # Summary table
    print("\n" + "=" * 74)
    print(f"{'Model':<20}{'Acc':>10}{'Prec':>10}{'Rec':>10}{'F1':>10}{'AUC':>12}")
    print("-" * 74)
    for name, m in metrics_all.items():
        print(f"{name:<20}{m['accuracy']:>10.4f}{m['precision_weighted']:>10.4f}"
              f"{m['recall_weighted']:>10.4f}{m['f1_weighted']:>10.4f}{m['macro_auc']:>12.4f}")
    print("=" * 74)
    best = max(metrics_all, key=lambda k: metrics_all[k]["f1_weighted"])
    print(f"[BEST] by weighted F1: {best}")

    if out_dir:
        save_json(metrics_all, out_dir / "comparison.json")
        print(f"\nSaved comparison to {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
