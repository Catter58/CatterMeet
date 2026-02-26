# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python dependency builder (clean install, no stripping) ────────
FROM python:3.11-slim AS python-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — avoids the default 2.5 GB CUDA wheel from PyPI
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 3: Download ML models ─────────────────────────────────────────────
# Inherits from the CLEAN (unstripped) builder so models download reliably
FROM python-builder AS model-downloader

ENV HF_HOME=/models/hf
ENV MODELS_DIR=/models/sb

RUN python -c "\
from faster_whisper import WhisperModel; \
print('Downloading Whisper large-v3-turbo...'); \
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); \
print('Whisper OK')"

RUN python -c "\
from silero_vad import load_silero_vad; \
print('Downloading silero-vad...'); \
load_silero_vad(); \
print('silero-vad OK')"

RUN python -c "\
from speechbrain.inference.speaker import SpeakerRecognition; \
print('Downloading ECAPA-TDNN...'); \
SpeakerRecognition.from_hparams( \
    source='speechbrain/spkrec-ecapa-voxceleb', \
    savedir='/models/sb/spkrec-ecapa-voxceleb', \
    run_opts={'device': 'cpu'}); \
print('ECAPA-TDNN OK')"


# ── Stage 4: Final runtime image ────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system libs + binutils for strip (removed in same layer to save space)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    binutils \
    && \
    # ── Copy Python packages from builder then strip in one layer ──────────
    true

# Copy site-packages from the clean builder
COPY --from=python-builder \
    /usr/local/lib/python3.11/site-packages \
    /usr/local/lib/python3.11/site-packages

# Strip debug symbols and remove bytecode cache — all in one RUN to keep layers thin
RUN \
    # Strip debug symbols from compiled extensions (~10-20% size reduction on .so files)
    find /usr/local/lib/python3.11/site-packages \
        -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true \
    && \
    # Remove ctranslate2 CUDA shared libs (we are CPU-only)
    find /usr/local/lib/python3.11/site-packages/ctranslate2 \
        \( -name "*cuda*" -o -name "*cublas*" -o -name "*cudnn*" \) \
        -exec rm -f {} \; 2>/dev/null || true \
    && \
    # Remove __pycache__ and .pyc bytecode (regenerated on first import, not needed in image)
    find /usr/local/lib/python3.11/site-packages \
        \( -type d -name "__pycache__" -o -name "*.pyc" -o -name "*.pyo" \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && \
    # Remove .dist-info and .egg-info metadata (pip is not used at runtime)
    find /usr/local/lib/python3.11/site-packages \
        \( -type d -name "*.dist-info" -o -type d -name "*.egg-info" \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && \
    # Remove test suites from SAFE packages only (not torch — its internals are fragile)
    for pkg in scipy sklearn speechbrain huggingface_hub transformers tqdm \
                aiofiles fastapi starlette uvicorn pydantic; do \
        find /usr/local/lib/python3.11/site-packages/$pkg \
            -type d \( -name "tests" -o -name "test" \) \
            -exec rm -rf {} + 2>/dev/null || true; \
    done \
    && \
    # Remove binutils after stripping (no longer needed at runtime)
    apt-get remove -y --purge binutils \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Pre-downloaded ML models
ENV HF_HOME=/models/hf
ENV MODELS_DIR=/models/sb
COPY --from=model-downloader /models /models

WORKDIR /app

COPY backend/ ./
COPY --from=frontend-builder /app/frontend/dist ./static

RUN mkdir -p uploads db

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
