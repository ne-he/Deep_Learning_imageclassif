# Dockerfile for Hugging Face Spaces (SDK: docker). Serves the FastAPI app on
# port 7860 (HF default). Build context must be the REPO ROOT so that src/,
# configs/ and the model checkpoint are available.
#
# Build/run locally from repo root:
#   docker build -f backend/Dockerfile -t nemvision .
#   docker run -p 7860:7860 nemvision
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    HF_HOME=/tmp/hf

WORKDIR /app

# CPU-only torch first (much smaller than the default CUDA wheels), then the rest.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install --no-cache-dir -r backend/requirements.txt

# Project code + config + the trained checkpoint.
COPY src ./src
COPY configs ./configs
COPY backend ./backend
COPY outputs/resnet50/best_model.pth ./outputs/resnet50/best_model.pth

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
