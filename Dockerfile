# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python dependency builder ─────────────────────────────────────
FROM python:3.11-slim AS python-builder

# Build tools needed to compile some Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — avoids the default 2.5 GB CUDA wheel from PyPI
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Strip everything that is not needed at runtime ──────────────────────────
# 1. Strip debug symbols from all compiled extensions (~10-20% size reduction)
RUN find /usr/local/lib/python3.11/site-packages \
    -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true

# 2. Remove __pycache__ and .pyc bytecode (they are regenerated at runtime)
RUN find /usr/local/lib/python3.11/site-packages \
    \( -type d -name "__pycache__" -o -name "*.pyc" -o -name "*.pyo" \) \
    -exec rm -rf {} + 2>/dev/null || true

# 3. Remove package metadata (.dist-info / .egg-info) — not needed at runtime
RUN find /usr/local/lib/python3.11/site-packages \
    \( -type d -name "*.dist-info" -o -type d -name "*.egg-info" \) \
    -exec rm -rf {} + 2>/dev/null || true

# 4. Remove test suites bundled inside packages
RUN find /usr/local/lib/python3.11/site-packages \
    -type d \( -name "tests" -o -name "test" -o -name "testing" \) \
    -exec rm -rf {} + 2>/dev/null || true

# 5. Remove torch's own test directories (but NOT torch/utils — it's a core module)
RUN find /usr/local/lib/python3.11/site-packages/torch \
    -maxdepth 1 -type d \
    \( -name "test" -o -name "testing" \) \
    -exec rm -rf {} + 2>/dev/null || true

# 6. Remove ctranslate2 CUDA libraries (we only use CPU)
RUN find /usr/local/lib/python3.11/site-packages/ctranslate2 \
    -name "*cuda*" -o -name "*cublas*" -o -name "*cudnn*" \
    | xargs rm -f 2>/dev/null || true


# ── Stage 3: Download ML models ─────────────────────────────────────────────
FROM python-builder AS model-downloader

# Deterministic cache paths so final stage knows where to COPY from
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
import os; \
from speechbrain.inference.speaker import SpeakerRecognition; \
print('Downloading ECAPA-TDNN...'); \
SpeakerRecognition.from_hparams( \
    source='speechbrain/spkrec-ecapa-voxceleb', \
    savedir='/models/sb/spkrec-ecapa-voxceleb', \
    run_opts={'device': 'cpu'}); \
print('ECAPA-TDNN OK')"


# ── Stage 4: Final runtime image ────────────────────────────────────────────
FROM python:3.11-slim

# Only the runtime system libraries — no build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python packages from builder (stripped, no pip cache, no build tools)
COPY --from=python-builder \
    /usr/local/lib/python3.11/site-packages \
    /usr/local/lib/python3.11/site-packages

# Pre-downloaded ML models
ENV HF_HOME=/models/hf
ENV MODELS_DIR=/models/sb
COPY --from=model-downloader /models /models

WORKDIR /app

# Backend source (tests and pytest.ini excluded via .dockerignore)
COPY backend/ ./

# Built frontend static files
COPY --from=frontend-builder /app/frontend/dist ./static

# Persistent data directories (overridden by docker-compose volumes)
RUN mkdir -p uploads db

EXPOSE 8000

# Use python -m to avoid needing the uvicorn script in /usr/local/bin
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
