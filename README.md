# Trash Image Classification — CNN Benchmark + Grad-CAM (PyTorch)

Multi-architecture comparison for 6-class waste classification on **TrashNet**
(cardboard, glass, metal, paper, plastic, trash · 2,527 images), with an
imbalance-aware pipeline, a soft-voting ensemble, and explainability.

This project is a **best-of-both merge** of two earlier implementations:

| Taken from | What |
|------------|------|
| PyTorch notebook | 3-model comparison, selective fine-tuning, rich augmentation, WeightedRandomSampler, label smoothing, AdamW + cosine LR, ensemble |
| Keras project | modular `src/` package, config-driven YAML, CLI, reproducibility, headless reporting |

## Models compared

| Model | Approach | Params (≈) |
|-------|----------|------------|
| **ResNet50** | residual learning — accuracy / deep features | 24.6M |
| **EfficientNet-B0** | compound scaling — efficiency / balance | 4.0M |
| **MobileNetV2** | depthwise-separable conv — lightweight / fast | 2.2M |
| **Ensemble** | soft voting over the three | — |

## Why these design choices

- **Selective fine-tuning** — only the deepest block(s) + a fresh head are
  unfrozen; general low-level features stay intact (good for a small dataset).
- **ImageNet normalization** — inputs match the backbones' pretraining
  distribution so transferred features stay valid.
- **WeightedRandomSampler + label smoothing** — the `trash` class has only 137
  images; oversampling it per batch lifts its recall substantially.
- **Train/Val/Test split (70/15/15)** — the test set is untouched during
  training/tuning for an honest final number.

## Setup

> ⚠️ **PyTorch tidak mendukung Python 3.14.** Pakai salah satu:
> - **Google Colab** (disarankan — GPU gratis). Lihat `notebooks/colab_runner.ipynb`.
> - venv lokal dengan **Python 3.10–3.12**.

```bash
python -m venv .venv            # Python 3.10-3.12
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Dataset

Letakkan 6 folder kelas di `data/dataset-resized/`:

```
data/dataset-resized/
├── cardboard/  ├── glass/  ├── metal/
├── paper/      ├── plastic/└── trash/
```

(Punya `dataset-resized.zip`? Unzip ke `data/`.)

## Usage

```bash
# Latih satu model
python scripts/train.py --config configs/resnet50.yaml

# Banding semua + ensemble (fair: split identik via seed)
python scripts/compare.py --configs configs/resnet50.yaml \
    configs/efficientnet_b0.yaml configs/mobilenet_v2.yaml

# Grad-CAM heatmap dari model terlatih (explainability)
python scripts/gradcam.py --config configs/resnet50.yaml
```

Artefak (checkpoint terbaik, `results.json`, confusion matrix) ditulis ke
`outputs/<arch>/`; ringkasan perbandingan ke `outputs/comparison.json`.

## Expected results (test set, reference run)

| Model | Acc | F1 | AUC |
|-------|-----|----|----|
| ResNet50 | ~0.91 | ~0.91 | ~0.99 |
| MobileNetV2 | ~0.84 | ~0.84 | ~0.97 |
| EfficientNet-B0 | ~0.81 | ~0.82 | ~0.97 |
| **Ensemble** | **~0.92** | **~0.92** | ~0.99 |

## Structure

```
configs/   YAML (base + per-model, with inheritance)
src/       config · data · models · engine · metrics · ensemble · utils
scripts/   train.py · compare.py
data/      dataset (gitignored)
outputs/   checkpoints + metrics (gitignored)
```

## Roadmap

- [x] **Phase 1** — config-driven 3-model comparison + ensemble
- [x] **Phase 2a** — Grad-CAM (hook-based, removable hooks)
- [ ] **Phase 2b** — FastAPI inference (serve best model / ensemble)
- [ ] **Phase 3** — tests + docs + ONNX export
- [ ] **Phase 4** — t-SNE, error analysis, auto-generated report
