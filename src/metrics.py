"""Evaluation metrics and plots (headless-safe).

Computes accuracy, weighted precision/recall/F1, per-class report, macro
one-vs-rest AUC, and optionally saves a confusion-matrix heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.utils import ensure_dir


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray,
                    class_names: List[str]) -> Dict:
    """Return headline + per-class metrics as a plain dict."""
    n_classes = len(class_names)
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    try:
        macro_auc = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")

    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_auc": macro_auc,
        "per_class": {c: report[c] for c in class_names if c in report},
    }


def print_report(labels, preds, class_names: List[str]) -> None:
    print(classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0))


def save_confusion_matrix(labels, preds, class_names: List[str], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(path)
    ensure_dir(path.parent)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
