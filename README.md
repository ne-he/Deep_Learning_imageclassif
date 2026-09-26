# Trash Image Classification: CNN Benchmark + Grad-CAM (PyTorch)

Multi-architecture comparison for 6-class waste classification on **TrashNet**
(cardboard, glass, metal, paper, plastic, trash · 2,527 images), with an
imbalance-aware pipeline, a soft-voting ensemble, and explainability.

**Live demo:** <https://deep-learning-imageclassif.vercel.app>, upload a waste photo and get
the predicted material plus a Grad-CAM heatmap. ResNet50 serves the demo from a FastAPI
backend (`backend/`) on a Hugging Face Space: <https://ne-he-nem-vision.hf.space>
([`/health`](https://ne-he-nem-vision.hf.space/health), [`/docs`](https://ne-he-nem-vision.hf.space/docs)).

This project is a **best-of-both merge** of two earlier implementations:

| Taken from | What |
|------------|------|
| PyTorch notebook | 3-model comparison, selective fine-tuning, rich augmentation, WeightedRandomSampler, label smoothing, AdamW + cosine LR, ensemble |
| Keras project | modular `src/` package, config-driven YAML, CLI, reproducibility, headless reporting |

## Models compared

| Model | Approach | Params (≈) |
|-------|----------|------------|
| **ResNet50** | residual learning: accuracy / deep features | 24.6M |
| **EfficientNet-B0** | compound scaling: efficiency / balance | 4.0M |
| **MobileNetV2** | depthwise-separable conv: lightweight / fast | 2.2M |
| **Ensemble** | soft voting over the three | n/a |

## Why these design choices

- **Selective fine-tuning**: only the deepest block(s) + a fresh head are
  unfrozen; general low-level features stay intact (good for a small dataset).
- **ImageNet normalization**: inputs match the backbones' pretraining
  distribution so transferred features stay valid.
- **WeightedRandomSampler + label smoothing**: the `trash` class has only 137
  images; oversampling it per batch lifts its recall substantially.
- **Train/Val/Test split (70/15/15)**: the test set is untouched during
  training/tuning for an honest final number.

## Setup

> ⚠️ **PyTorch tidak mendukung Python 3.14.** Pakai salah satu:
> - **Google Colab** (disarankan, GPU gratis). Lihat `notebooks/colab_runner.ipynb`.
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

## Results (test set)

Measured on the held-out test split (70/15/15, identical seed-locked split for every model),
as written to `outputs/comparison.json` by `scripts/compare.py`:

| Model | Accuracy | Weighted F1 | Macro AUC |
|-------|----------|-------------|-----------|
| **ResNet50** (serves the live demo) | **0.9156** | **0.9158** | 0.9897 |
| MobileNetV2 | 0.8417 | 0.8474 | 0.9710 |
| EfficientNet-B0 | 0.8047 | 0.8061 | 0.9735 |
| Soft-voting ensemble | 0.9103 | 0.9109 | **0.9912** |

## Structure

```
configs/   YAML (base + per-model, with inheritance)
src/       config · data · models · engine · metrics · ensemble · utils
scripts/   train.py · compare.py
data/      dataset (gitignored)
outputs/   checkpoints + metrics (gitignored)
```

## Roadmap

- [x] **Phase 1**: config-driven 3-model comparison + ensemble
- [x] **Phase 2a**: Grad-CAM (hook-based, removable hooks)
- [x] **Phase 2b**: FastAPI inference, ResNet50 served from `backend/` and deployed as a
  Hugging Face Space behind the live demo
- [ ] **Phase 3**: tests + docs + ONNX export
- [ ] **Phase 4**: t-SNE, error analysis, auto-generated report
