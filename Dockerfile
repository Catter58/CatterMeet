# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (layer cache)
COPY backend/requirements.txt ./

# Install CPU-only torch first (avoids downloading the 2.5 GB CUDA wheel)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Copy built frontend into static folder
COPY --from=frontend-builder /app/frontend/dist ./static

# Pre-download all ML models during build so first run is instant
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
SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec-ecapa-voxceleb', run_opts={'device': 'cpu'}); \
print('ECAPA-TDNN OK')"

# Create upload dir (volume mount will overlay this)
RUN mkdir -p uploads db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
