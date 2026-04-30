"""
Local ASR backends.

The IndicConformer backend uses the non-gated ONNX exports of AI4Bharat
IndicConformer models. It is intended for offline/beta corpus creation where
Sarvam API cost is a blocker.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


INDICCONFORMER_ONNX_REPO = "sulabhkatiyar/indicconformer-120m-onnx"


@dataclass(frozen=True)
class ASRResult:
    transcript: str
    words: list[dict]
    backend: str
    version: str


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _language_code(value: str) -> str:
    lang = (value or "").strip().lower()
    if not lang:
        return "te"
    if "-" in lang:
        lang = lang.split("-", 1)[0]
    aliases = {
        "tel": "te",
        "telugu": "te",
        "hin": "hi",
        "hindi": "hi",
        "kan": "kn",
        "kannada": "kn",
        "tam": "ta",
        "tamil": "ta",
    }
    return aliases.get(lang, lang)


def _audio_bytes_to_wav(audio_bytes: bytes, suffix: str) -> Path:
    src = tempfile.NamedTemporaryFile(suffix=suffix or ".audio", delete=False)
    dst = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    src_path = Path(src.name)
    dst_path = Path(dst.name)
    try:
        src.write(audio_bytes)
        src.close()
        dst.close()
        cmd = [
            _ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst_path),
        ]
        run = subprocess.run(cmd, capture_output=True)
        if run.returncode != 0:
            raise RuntimeError(f"ffmpeg audio conversion failed: {run.stderr.decode()[:400]}")
        return dst_path
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def _nemo_mel_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros((80, 1), dtype=np.float32)
    audio = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]])
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=8000,
        norm="slaney",
        power=2.0,
    )
    log_mel = np.log(mel + math.pow(2, -24)).astype(np.float32)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, ddof=1, keepdims=True) + 1e-5
    return ((log_mel - mean) / std).astype(np.float32)


@lru_cache(maxsize=8)
def _load_indicconformer(lang: str):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(INDICCONFORMER_ONNX_REPO, f"{lang}/model.onnx")
    vocab_path = hf_hub_download(INDICCONFORMER_ONNX_REPO, f"{lang}/vocab.json")
    providers = [
        provider
        for provider in ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        if provider in ort.get_available_providers()
    ]
    if not providers:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return session, vocab


def _ctc_greedy_decode(logits: np.ndarray, vocab: list[str]) -> str:
    blank_id = len(vocab)
    ids = np.argmax(logits[0], axis=-1)
    prev = -1
    tokens: list[str] = []
    for token_id in ids:
        token = int(token_id)
        if token != prev:
            if token != blank_id and token < len(vocab):
                tokens.append(vocab[token])
        prev = token
    return "".join(tokens).replace("\u2581", " ").strip()


def transcribe_indicconformer(
    audio_bytes: bytes,
    *,
    filename: str,
    language_code: str,
    suffix: str,
) -> ASRResult:
    lang = _language_code(language_code)
    if lang not in {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"}:
        raise RuntimeError(f"IndicConformer ONNX backend does not support language code: {language_code}")

    wav_path = _audio_bytes_to_wav(audio_bytes, suffix)
    try:
        import librosa

        audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    mel = _nemo_mel_spectrogram(audio, sr=sr)
    mel_batch = mel[np.newaxis, :, :].astype(np.float32)
    mel_length = np.array([mel_batch.shape[2]], dtype=np.int64)
    session, vocab = _load_indicconformer(lang)
    logits = session.run(None, {"audio_signal": mel_batch, "length": mel_length})[0]
    transcript = _ctc_greedy_decode(logits, vocab)
    return ASRResult(
        transcript=transcript,
        words=[],
        backend="indicconformer",
        version=f"{INDICCONFORMER_ONNX_REPO}:{lang}",
    )
