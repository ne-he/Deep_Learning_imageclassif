# Waste Image Classification with CNN and Grad-CAM

Image classification model for waste sorting using Convolutional Neural Networks with Grad-CAM explainability. Built as part of a deep learning final project.

## Overview

The goal of this project is to classify waste images into six categories using a CNN-based model. Beyond classification accuracy, the project also incorporates Grad-CAM to visualize which parts of an image the model focuses on when making a prediction, addressing the interpretability problem common in deep learning systems.

**Task:** Multi-class image classification (6 categories)  
**Dataset:** TrashNet — 2,527 images  
**Framework:** TensorFlow / Keras  
**Runtime:** Google Colab (GPU T4)

## Dataset

The dataset used is TrashNet, available on Kaggle. It contains waste images in six categories:

| Class | Images |
|-------|--------|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |
| **Total** | **2,527** |

Original image resolution: 384×512 pixels. All images resized to 224×224 for training.

## Model Architecture

Two models were built and compared:

**Baseline CNN (from scratch)**
- 3x Conv2D + MaxPooling2D blocks (32, 64, 128 filters)
- Dense(128, ReLU) + Dropout(0.5)
- Dense(6, softmax)
- Final validation accuracy: 54.5%

**MobileNetV2 Transfer Learning (final model)**
- MobileNetV2 backbone pretrained on ImageNet (frozen)
- GlobalAveragePooling2D
- Dense(128, ReLU) + Dropout(0.5)
- Dense(6, softmax)
- Final validation accuracy: 77.1%

## Results

| Metric | Baseline CNN | MobileNetV2 |
|--------|-------------|-------------|
| Val Accuracy | 54.5% | 77.1% |
| Val Loss | 1.2125 | 0.6038 |
| Macro AUC | — | 0.9617 |

Per-class F1-scores (MobileNetV2):

| Class | F1-Score |
|-------|----------|
| paper | 0.864 |
| cardboard | 0.833 |
| metal | 0.800 |
| glass | 0.745 |
| plastic | 0.690 |
| trash | 0.390 |

## Preprocessing

- Resize to 224×224
- Normalize pixel values to [0, 1]
- 80/20 train/validation split
- Training augmentation: rotation ±20°, zoom 20%, horizontal flip

## How to Run

1. Open `image_classification.ipynb` in Google Colab
2. Set runtime to GPU (T4 recommended)
3. Run all cells in order

The notebook downloads the dataset from a local server exposed via VS Code Dev Tunnel. If the tunnel URL is no longer active, replace the download cell with a direct upload or mount from Google Drive.

Dataset source: https://www.kaggle.com/datasets/feyzazkefe/trashnet

## Key Findings

- Transfer learning with MobileNetV2 outperformed the scratch CNN by 22.7 percentage points on validation accuracy
- The trash class had the lowest performance (F1 = 0.39) due to severe class imbalance (only 137 images) and high intra-class visual variability
- Grad-CAM confirmed the model focuses on semantically relevant regions: cylindrical edges for metal, transparent surfaces for glass, flat texture for paper and cardboard
- A misclassification between plastic and glass was observed and expected — transparent plastic bottles are visually nearly identical to glass bottles at this scale

## Project Structure

```
├── image_classification.ipynb   # Main notebook (EDA, training, evaluation, Grad-CAM)
├── nomor1_report.md             # Individual report answering GSLC task
└── README.md
```
