"""Trash image classification — PyTorch, modular, config-driven.

Merge of the best parts of two implementations:
- Modeling (3-model comparison, selective fine-tuning, weighted sampler,
  label smoothing, AdamW + cosine LR, ensemble) from the PyTorch notebook.
- Engineering (config-driven YAML, modular package, CLI, reproducibility)
  from the Keras project.
"""

__version__ = "0.1.0"
