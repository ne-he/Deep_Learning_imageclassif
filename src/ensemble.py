"""Soft-voting ensemble: average softmax probabilities across models.

Combining complementary backbones usually beats any single model — it was
the top performer in the original comparison.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def soft_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    """Average a list of ``(N, C)`` probability arrays into one ``(N, C)``."""
    if not prob_list:
        raise ValueError("prob_list is empty")
    stacked = np.stack(prob_list, axis=0)
    return stacked.mean(axis=0)


def ensemble_predictions(results: Dict[str, dict]) -> dict:
    """Soft-vote over per-model results.

    ``results`` maps model name -> {'probs', 'labels'} (labels identical across
    models because they share the test loader). Returns a result dict shaped
    like a single model's, under the key meaning "Ensemble".
    """
    names = list(results.keys())
    labels = results[names[0]]["labels"]
    probs = soft_vote([results[n]["probs"] for n in names])
    preds = probs.argmax(axis=1)
    return {"preds": preds, "labels": labels, "probs": probs}
