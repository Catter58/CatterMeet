# backend/processor.py
import gc
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert any audio/video to 16kHz mono WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def _run_vad(wav_path: str) -> list:
    """Find speech segments with silero-vad. Evict model after use."""
    import torch
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    model = load_silero_vad()
    audio = read_audio(wav_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(
        audio, model, sampling_rate=16000, return_seconds=True
    )

    del model
    del audio
    gc.collect()

    return [{"start": float(ts["start"]), "end": float(ts["end"])} for ts in timestamps]


def _run_stt(wav_path: str, vad_segments: list) -> list:
    """Transcribe with faster-whisper, only over VAD segments. Evict model after use."""
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

    results = []
    for seg in vad_segments:
        segments_iter, _ = model.transcribe(
            wav_path,
            language=None,          # auto-detect (ru/en)
            beam_size=5,
            clip_timestamps=[seg["start"], seg["end"]],
            word_timestamps=False,
        )
        for s in segments_iter:
            text = s.text.strip()
            if text:
                results.append({
                    "start_time": float(s.start),
                    "end_time": float(s.end),
                    "text": text,
                })

    del model
    gc.collect()

    return results


def _run_diarization(wav_path: str, segments: list) -> list:
    """Extract speaker embeddings (ECAPA-TDNN), cluster with KMeans. Evict model after use."""
    import torch
    import torchaudio
    from speechbrain.inference.speaker import SpeakerRecognition

    spk_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    embeddings = []
    valid_indices = []
    for i, seg in enumerate(segments):
        start_sample = int(seg["start_time"] * 16000)
        end_sample = int(seg["end_time"] * 16000)
        chunk = waveform[:, start_sample:end_sample]
        if chunk.shape[1] < 1600:  # skip segments shorter than 0.1s
            continue
        with torch.no_grad():
            emb = spk_model.encode_batch(chunk)
        embeddings.append(emb.squeeze().numpy())
        valid_indices.append(i)

    del spk_model
    del waveform
    gc.collect()

    if not embeddings:
        # No valid segments — assign all to SPEAKER_0
        for seg in segments:
            seg["speaker"] = "SPEAKER_0"
        return segments

    from sklearn.cluster import KMeans

    X = np.array(embeddings)
    n_speakers = min(2, len(embeddings))
    kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Assign labels back to segments
    label_ptr = 0
    for i, seg in enumerate(segments):
        if label_ptr < len(valid_indices) and valid_indices[label_ptr] == i:
            seg["speaker"] = f"SPEAKER_{labels[label_ptr]}"
            label_ptr += 1
        else:
            seg["speaker"] = "SPEAKER_0"

    return segments


def process_file(file_path: str) -> list:
    """
    Full pipeline: convert -> VAD -> STT -> Diarization.
    Returns list of dicts: {start_time, end_time, speaker, text}
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        logger.info(f"Converting {file_path} to WAV...")
        _convert_to_wav(file_path, wav_path)

        logger.info("Running VAD...")
        vad_segments = _run_vad(wav_path)
        logger.info(f"VAD found {len(vad_segments)} speech segments")

        if not vad_segments:
            logger.warning("No speech detected in file")
            return []

        logger.info("Running STT...")
        transcript_segments = _run_stt(wav_path, vad_segments)
        logger.info(f"STT produced {len(transcript_segments)} segments")

        if not transcript_segments:
            logger.warning("STT produced no segments")
            return []

        logger.info("Running diarization...")
        diarized = _run_diarization(wav_path, transcript_segments)
        logger.info("Diarization complete")

        return diarized

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
            logger.info(f"Cleaned up temp WAV: {wav_path}")
