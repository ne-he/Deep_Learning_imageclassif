"""FastAPI inference server for the trash classifier.

Loads the trained ResNet50 checkpoint once at startup and serves Grad-CAM
explained predictions over HTTP. Reuses the project's own model / transform /
Grad-CAM code in ``src/`` so the served model is identical to training.

Run locally:
    uvicorn backend.app:app --reload --port 7860

Endpoints:
    GET  /health   -> {"status": "ok", "model": ...}
    POST /predict  -> multipart form field ``file`` (image) ->
                      {label, key, confidence, distribution, gradcam, original}
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.config import Config
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import GradCAM, denormalize, overlay_heatmap
from src.models import build_model, get_gradcam_target_layer

# --------------------------------------------------------------------------- #
# Configuration (override via environment variables on the host)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
ARCH = os.environ.get("MODEL_ARCH", "resnet50")
CONFIG_PATH = os.environ.get("MODEL_CONFIG", str(ROOT / "configs" / f"{ARCH}.yaml"))
CKPT_PATH = os.environ.get(
    "MODEL_CKPT", str(ROOT / "outputs" / ARCH / "best_model.pth")
)
# Comma-separated list of allowed origins, or "*" for any.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Load model once at import time
# --------------------------------------------------------------------------- #
cfg = Config.from_yaml(CONFIG_PATH)
cfg.model.architecture = ARCH  # keep config + checkpoint in sync
CLASS_NAMES = cfg.classes

model = build_model(cfg.model)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE).eval()

_target_layer = get_gradcam_target_layer(model, ARCH)

# Eval transform: identical to src/data.build_transforms eval branch.
_img_size = cfg.data.img_size
eval_tf = T.Compose(
    [
        T.Resize((_img_size, _img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def _to_data_uri(arr_or_img) -> str:
    """Encode an HxWx3 float[0..1] array or a PIL image as a PNG data URI."""
    if isinstance(arr_or_img, np.ndarray):
        img = Image.fromarray((np.clip(arr_or_img, 0, 1) * 255).astype(np.uint8))
    else:
        img = arr_or_img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="NemVision Trash Classifier", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": ARCH, "classes": CLASS_NAMES, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image.")

    tensor = eval_tf(pil)  # (C, H, W)

    # Grad-CAM: heatmap + predicted class + its confidence.
    gradcam = GradCAM(model, _target_layer)
    try:
        cam, pred_idx, conf = gradcam.generate(tensor, device=DEVICE)
    finally:
        gradcam.remove()

    # Full probability distribution over all classes.
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    distribution = {
        CLASS_NAMES[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)
    }

    overlay = overlay_heatmap(tensor, cam, img_size=_img_size)
    original = denormalize(tensor).permute(1, 2, 0).cpu().numpy()

    return {
        "label": CLASS_NAMES[pred_idx].capitalize(),
        "key": CLASS_NAMES[pred_idx],
        "confidence": round(float(conf) * 100, 1),
        "distribution": distribution,
        "gradcam": _to_data_uri(overlay),
        "original": _to_data_uri(original),
    }
