import asyncio
import json
import os
import subprocess
import re
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

import httpx
import structlog
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from api.auth import require_admin_api_key
from models.schemas import AudioTranscribeResponse
from core import dataset_store
from core.config import get_settings
from core.insight_store import record_ingestion_run
from pipelines.audio_chunker import chunk_and_index
from core.translation import reset_translation_provider_state
from utils.naming import canonical_jsonl_filename, normalize_language_tag, part_number_from_filename, source_stem_from_audio_filename

router = APIRouter()
log = structlog.get_logger()
settings = get_settings()

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}

# Files larger than this are split into 29s segments before STT.
# Sarvam saaras:v3 processes ~30s max per request; 2MB ≈ ~2 min @ 128kbps.
_SPLIT_THRESHOLD_BYTES = 2 * 1024 * 1024
_SEGMENT_SECS = 29


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _split_audio_to_segments(source_path: str) -> tuple[list[str], str]:
    """Split source audio into 29s MP3 segments. Returns (segment_paths, tmp_dir)."""
    tmp_dir = tempfile.mkdtemp(prefix="dharma_split_")
    out_pattern = os.path.join(tmp_dir, "seg%04d.mp3")
    cmd = [
        _ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", source_path,
        "-f", "segment", "-segment_time", str(_SEGMENT_SECS),
        "-reset_timestamps", "1", "-c:a", "libmp3lame", out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {result.stderr.decode()[:400]}")
    segs = sorted(Path(tmp_dir).glob("seg*.mp3"))
    if not segs:
        raise RuntimeError("ffmpeg produced no segments")
    return [str(s) for s in segs], tmp_dir
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "processed"
TRANSCRIPT_DIR = PROCESSED_DIR / "audio_transcript"
RAW_AUDIO_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "uploads" / "audio_sources"
AUDIT_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "audit"

_MIME_MAP = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac", ".opus": "audio/opus",
}
_LANG_SHORT = {
    "te-in": "te", "hi-in": "hi", "sa-in": "sa", "en-in": "en", "en-us": "en",
}


class FatalProviderError(RuntimeError):
    """Provider error that should stop a segmented upload immediately."""


def _safe_filename(filename: str) -> str:
    name = Path(filename or "audio").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "audio"


def _save_audio_source(filename: str, audio_bytes: bytes) -> tuple[str, str]:
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_AUDIO_DIR / f"{ts}_{_safe_filename(filename)}"
    path.write_bytes(audio_bytes)
    return str(path), sha256(audio_bytes).hexdigest()


def _append_audio_audit(record: dict) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with (AUDIT_DIR / "audio_uploads.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── STT backends ──────────────────────────────────────────────────────────────

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"


async def _transcribe_with_sarvam(audio_bytes: bytes, filename: str, language_code: str, suffix: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": settings.sarvam_api_key},
            files={"file": (filename, audio_bytes, _MIME_MAP.get(suffix, "audio/mpeg"))},
            data={"model": "saaras:v3", "language_code": language_code, "with_timestamps": "true"},
        )
        response.raise_for_status()
        return response.json()


async def _translate_with_sarvam(audio_bytes: bytes, filename: str, suffix: str) -> str | None:
    """Call Sarvam speech-to-text-translate to get English directly from audio."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            SARVAM_TRANSLATE_URL,
            headers={"api-subscription-key": settings.sarvam_api_key},
            files={"file": (filename, audio_bytes, _MIME_MAP.get(suffix, "audio/mpeg"))},
            data={"model": "saaras:v2.5"},
        )
        response.raise_for_status()
        data = response.json()
    return (data.get("transcript") or "").strip() or None


async def _sarvam_stt_and_translate_parallel(
    audio_bytes: bytes,
    filename: str,
    language_code: str,
    suffix: str,
) -> tuple[dict, str | None]:
    """Fire Sarvam STT (Telugu + timestamps) and Sarvam Translate (English) in parallel.

    Returns (stt_result, english_text_or_None).
    Failures are logged as notifications; the pipeline continues with whatever succeeded.
    """
    from core.dataset_store import push_notification

    stt_task = asyncio.create_task(_transcribe_with_sarvam(audio_bytes, filename, language_code, suffix))
    translate_task = asyncio.create_task(_translate_with_sarvam(audio_bytes, filename, suffix))

    stt_result, translate_result = await asyncio.gather(stt_task, translate_task, return_exceptions=True)

    # Handle STT failure
    if isinstance(stt_result, Exception):
        fatal = _is_auth_provider_error(stt_result)
        push_notification(
            event="sarvam_stt_failed",
            detail=str(stt_result),
            file_name=filename,
            level="error",
        )
        log.error("sarvam_stt_failed", file=filename, error=str(stt_result))
        if fatal:
            raise FatalProviderError(str(stt_result)) from stt_result
        raise stt_result  # STT is required — re-raise so caller can fall back

    # Handle translate failure (non-fatal — pipeline continues without English)
    en_text: str | None = None
    if isinstance(translate_result, Exception):
        push_notification(
            event="sarvam_translate_failed",
            detail=str(translate_result),
            file_name=filename,
            level="warning",
        )
        log.warning("sarvam_translate_failed", file=filename, error=str(translate_result))
    else:
        en_text = translate_result
        if en_text:
            log.info("sarvam_translate_ok", file=filename, chars=len(en_text))

    return stt_result, en_text


def _is_auth_provider_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {401, 403}
    return "401" in str(exc) or "403" in str(exc) or "Forbidden" in str(exc)


async def _transcribe_with_claude(audio_bytes: bytes, filename: str, language_code: str, suffix: str) -> dict:
    import base64
    audio_b64 = base64.standard_b64encode(audio_bytes).decode()
    media_type = _MIME_MAP.get(suffix, "audio/mpeg")
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "audio-1",
                "content-type": "application/json",
            },
            json={
                "model": settings.anthropic_model,
                "max_tokens": 4096,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "source": {"type": "base64", "media_type": media_type, "data": audio_b64},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Transcribe this {language_code} audio exactly as spoken. "
                                "Return only the transcription text, nothing else."
                            ),
                        },
                    ],
                }],
            },
        )
        response.raise_for_status()
        data = response.json()
    transcript = data["content"][0]["text"].strip()
    return {"transcript": transcript, "words": []}


async def _transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    language_code: str,
    suffix: str,
) -> tuple[dict, str, str]:
    """Returns (transcript_data, transcription_mode, transcription_version).

    STT_BACKEND=sarvam (production default).
    No fallback — if Sarvam fails the exception propagates and kills the pipeline.
    """
    data, en_text = await _sarvam_stt_and_translate_parallel(
        audio_bytes, filename, language_code, suffix
    )
    if en_text:
        data["text_en_sarvam"] = en_text
    return data, "sarvam_stt", "saaras:v3"


async def _transcribe_with_auto_split(
    audio_bytes: bytes,
    filename: str,
    language_code: str,
    suffix: str,
    progress: Callable[[dict], None] | None = None,
) -> tuple[dict, str, str]:
    """
    Transparently splits large audio files into 29s segments before STT.
    Sarvam saaras:v3 has a ~30s limit per request; anything larger must be chunked first.
    Results from all segments are merged into a single transcript + word list.
    """
    if len(audio_bytes) <= _SPLIT_THRESHOLD_BYTES:
        if progress:
            progress({"stage": "transcription", "segments_total": 1, "segments_done": 0, "segments_failed": 0})
        result = await _transcribe_audio(audio_bytes, filename, language_code, suffix)
        if progress:
            progress({"stage": "transcription", "segments_total": 1, "segments_done": 1, "segments_failed": 0})
        return result

    log.info("audio_auto_split_start", file=filename, size_mb=round(len(audio_bytes) / 1e6, 2))

    # Write source to a temp file so ffmpeg can read it
    tmp_src = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp_src.write(audio_bytes)
        tmp_src.close()
        seg_paths, tmp_dir = await asyncio.to_thread(_split_audio_to_segments, tmp_src.name)
    finally:
        os.unlink(tmp_src.name)

    log.info("audio_auto_split_done", segments=len(seg_paths))
    if progress:
        progress({
            "stage": "transcription",
            "segments_total": len(seg_paths),
            "segments_done": 0,
            "segments_failed": 0,
        })

    all_text: list[str] = []
    all_words: list[dict] = []
    all_en_sarvam: list[str] = []
    last_mode = "sarvam_stt"
    last_version = "saaras:v3"
    time_offset = 0.0
    failed_segments = 0

    try:
        for i, seg_path in enumerate(seg_paths):
            seg_bytes = Path(seg_path).read_bytes()
            seg_name = f"{Path(filename).stem}_seg{i:04d}.mp3"
            try:
                data, mode, version = await _transcribe_audio(seg_bytes, seg_name, language_code, ".mp3")
                last_mode, last_version = mode, version
                text = (data.get("transcript") or "").strip()
                if text:
                    all_text.append(text)
                en = (data.get("text_en_sarvam") or "").strip()
                if en:
                    all_en_sarvam.append(en)
                for w in data.get("words") or []:
                    all_words.append({
                        **w,
                        "start": (w.get("start") or 0.0) + time_offset,
                        "end": (w.get("end") or 0.0) + time_offset,
                    })
            except Exception as exc:
                failed_segments += 1
                log.warning("segment_transcribe_failed", segment=i, error=str(exc))
                if progress:
                    progress({
                        "stage": "transcription",
                        "segments_total": len(seg_paths),
                        "segments_done": i + 1,
                        "segments_failed": failed_segments,
                        "current_segment": i + 1,
                        "last_error": str(exc)[:500],
                    })
                if isinstance(exc, FatalProviderError) or _is_auth_provider_error(exc):
                    from core.dataset_store import push_notification
                    push_notification(
                        event="audio_upload_aborted",
                        detail=f"Stopping segmented upload after provider auth failure on segment {i}: {exc}",
                        file_name=filename,
                        level="error",
                    )
                    raise RuntimeError(f"Provider auth failed on segment {i}; aborting remaining {len(seg_paths) - i - 1} segment(s)") from exc
            else:
                if progress:
                    progress({
                        "stage": "transcription",
                        "segments_total": len(seg_paths),
                        "segments_done": i + 1,
                        "segments_failed": failed_segments,
                        "current_segment": i + 1,
                    })
            time_offset += _SEGMENT_SECS
    finally:
        for p in seg_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if not all_text:
        raise RuntimeError("All segments failed to transcribe")

    result: dict = {"transcript": " ".join(all_text), "words": all_words}
    if all_en_sarvam:
        result["text_en_sarvam"] = " ".join(all_en_sarvam)

    log.info("audio_auto_split_merged", segments=len(seg_paths), words=len(all_words),
             sarvam_en_segments=len(all_en_sarvam))
    return result, last_mode, last_version


@router.post("/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    section: str = Form(None),
    description: str = Form(None),
    dataset_name: str = Form(""),
    source_title: str = Form(None),
    source: str = Form(None),
    _: None = Depends(require_admin_api_key),
) -> AudioTranscribeResponse:
    """
    Upload a Sanskrit/Hindi audio file (chanting, pravachanam, discourse).
    STT backend is controlled by STT_BACKEND in .env: "sarvam" | "whisper" | "auto".
    """
    suffix = "." + (file.filename or "").split(".")[-1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {SUPPORTED_FORMATS}")

    audio_bytes = await file.read()
    if len(audio_bytes) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 100MB.")
    source_file_path, content_sha256 = _save_audio_source(file.filename or "audio", audio_bytes)
    reset_translation_provider_state()
    resolved_source = source or source_stem_from_audio_filename(
        file.filename or "audio",
        language=normalize_language_tag(language_code),
    )
    resolved_title = source_title or description or file.filename or ""
    ds_id = dataset_name.strip()
    run_id = record_ingestion_run(
        kind="audio",
        source=resolved_source,
        source_title=resolved_title,
        file_name=file.filename or "",
        language=language_code,
        dataset_id=ds_id,
        status="running",
        chunks=0,
        vectors=0,
        vector_db="postgres",
        embedding_backend="",
        transcription_mode=settings.stt_backend,
        transcription_version="",
        metadata={
            "source_file_path": source_file_path,
            "content_sha256": content_sha256,
            "stage": "transcription",
            "size_bytes": len(audio_bytes),
        },
    )

    last_progress: dict = {}

    def run_metadata(extra: dict | None = None) -> dict:
        return {
            "source_file_path": source_file_path,
            "content_sha256": content_sha256,
            "size_bytes": len(audio_bytes),
            **last_progress,
            **(extra or {}),
        }

    def update_progress(extra: dict) -> None:
        last_progress.update(extra)
        record_ingestion_run(
            id=run_id,
            kind="audio",
            source=resolved_source,
            source_title=resolved_title,
            file_name=file.filename or "",
            language=language_code,
            dataset_id=ds_id,
            status="running",
            chunks=0,
            vectors=0,
            vector_db="postgres",
            embedding_backend="",
            transcription_mode=settings.stt_backend,
            transcription_version="",
            metadata=run_metadata(),
        )

    log.info("audio_transcribe_start", file=file.filename, lang=language_code, size_mb=round(len(audio_bytes)/1e6, 2), stt_backend=settings.stt_backend)

    try:
        transcript_data, transcription_mode, transcription_version = await _transcribe_with_auto_split(
            audio_bytes, file.filename, language_code, suffix, progress=update_progress
        )
    except Exception as e:
        dataset_store.push_notification(
            event="audio_transcription_failed",
            detail=str(e),
            file_name=file.filename or "",
            level="error",
        )
        record_ingestion_run(
            id=run_id,
            kind="audio",
            source=resolved_source,
            source_title=resolved_title,
            file_name=file.filename or "",
            language=language_code,
            dataset_id=ds_id,
            status="failed",
            chunks=0,
            vectors=0,
            vector_db="postgres",
            embedding_backend="",
            transcription_mode=settings.stt_backend,
            transcription_version="",
            error=str(e),
            metadata=run_metadata({
                "stage": "transcription",
                "last_error": str(e)[:500],
            }),
        )
        log.error("stt_error", error=str(e), stt_backend=settings.stt_backend)
        raise HTTPException(status_code=502, detail="Audio transcription service unavailable.")

    transcript_text = transcript_data.get("transcript", "")
    if not transcript_text:
        record_ingestion_run(
            id=run_id,
            kind="audio",
            source=resolved_source,
            source_title=resolved_title,
            file_name=file.filename or "",
            language=language_code,
            dataset_id=ds_id,
            status="failed",
            chunks=0,
            vectors=0,
            vector_db="postgres",
            transcription_mode=settings.stt_backend,
            error="No speech detected in audio file.",
            metadata=run_metadata({
                "stage": "transcription",
                "last_error": "No speech detected in audio file.",
            }),
        )
        raise HTTPException(status_code=422, detail="No speech detected in audio file.")

    # Register dataset and chunk/index
    if ds_id:
        dataset_store.register(ds_id)

    file_metadata = {
        "language_code": language_code,
        "section": section,
        "description": description or file.filename,
        "source_title": resolved_title,
        "source": resolved_source,
        "transcription_mode": transcription_mode,
        "transcription_version": transcription_version,
        "text_source": "Valmiki Ramayana",
        "source_file": file.filename,
        "source_file_path": source_file_path,
        "content_sha256": content_sha256,
    }
    try:
        chunk_result = await chunk_and_index(transcript_data, file.filename, file_metadata, dataset_id=ds_id)
        record_ingestion_run(
            id=run_id,
            kind="audio",
            source=file_metadata["source"],
            source_title=file_metadata["source_title"],
            file_name=file.filename or "",
            language=language_code,
            dataset_id=ds_id,
            status="running",
            chunks=chunk_result.get("chunks_created") or 0,
            vectors=0,
            vector_db="postgres",
            embedding_backend="",
            transcription_mode=transcription_mode,
            transcription_version=transcription_version,
            translation_backend=chunk_result.get("translation_backend"),
            translation_version=chunk_result.get("translation_version"),
            metadata=run_metadata({
                "stage": "staging",
                "chunks_created": chunk_result.get("chunks_created") or 0,
            }),
        )
    except Exception as exc:
        record_ingestion_run(
            id=run_id,
            kind="audio",
            source=file_metadata["source"],
            source_title=file_metadata["source_title"],
            file_name=file.filename or "",
            language=language_code,
            dataset_id=ds_id,
            status="failed",
            error=str(exc),
            transcription_mode=transcription_mode,
            transcription_version=transcription_version,
            metadata=run_metadata({
                "stage": "staging",
                "last_error": str(exc)[:500],
            }),
        )
        raise HTTPException(status_code=503, detail="Translation/indexing providers unavailable; retry later.") from exc
    if ds_id:
        dataset_store.increment_count(ds_id, chunk_result.get("chunks_created", 0))

    transcript_language = normalize_language_tag(language_code)
    transcript_base = source_stem_from_audio_filename(file.filename, language=transcript_language)
    transcript_part = part_number_from_filename(file.filename)
    transcript_file_name = canonical_jsonl_filename(
        transcript_base,
        language=transcript_language,
        kind="transcript",
        part=transcript_part,
    )
    transcript_path = TRANSCRIPT_DIR / transcript_base / transcript_file_name
    transcript_record = {
        "id": transcript_path.stem,
        "text": transcript_text,
        "text_en": chunk_result.get("translated_transcript"),
        "source": transcript_base,
        "section": section,
        "citation": f"Audio transcription: {description or file.filename}",
        "language": transcript_language,
        "source_type": "audio_transcript",
        "tags": ["shloka"] if "॥" in transcript_text or "।" in transcript_text else ["story"],
        "topics": [],
        "characters": [],
        "is_shloka": "॥" in transcript_text or "।" in transcript_text,
        "url": "",
        "notes": f"Transcribed from {file.filename}",
        "source_file": file.filename,
        "source_file_path": source_file_path,
        "content_sha256": content_sha256,
        "description": description or file.filename,
        "transcription_mode": transcription_mode,
        "transcription_version": transcription_version,
        "translation_mode": chunk_result.get("translation_mode"),
        "translation_backend": chunk_result.get("translation_backend"),
        "translation_version": chunk_result.get("translation_version"),
        "translation_fallback_reason": chunk_result.get("translation_fallback_reason"),
        "translation_attempted_backends": chunk_result.get("translation_attempted_backends") or [],
        "chunks_created": chunk_result["chunks_created"],
        "vector_db": chunk_result.get("vector_db"),
        "vectors_upserted": chunk_result.get("vectors_upserted"),
        "embedding_backend": chunk_result.get("embedding_backend"),
    }

    try:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(transcript_record, ensure_ascii=False) + "\n")
    except OSError as exc:
        record_ingestion_run(
            id=run_id,
            kind="audio",
            source=file_metadata["source"],
            source_title=file_metadata["source_title"],
            file_name=file.filename or "",
            language=language_code,
            dataset_id=ds_id,
            status="failed",
            chunks=chunk_result.get("chunks_created") or 0,
            vectors=chunk_result.get("vectors_upserted") or 0,
            vector_db=chunk_result.get("vector_db"),
            embedding_backend=chunk_result.get("embedding_backend"),
            transcription_mode=transcription_mode,
            transcription_version=transcription_version,
            translation_backend=chunk_result.get("translation_backend"),
            translation_version=chunk_result.get("translation_version"),
            error=str(exc),
            metadata=run_metadata({
                "stage": "transcript_artifact",
            }),
        )
        log.error("audio_transcript_save_failed", file=file.filename, path=str(transcript_path), error=str(exc))
        raise HTTPException(status_code=500, detail="Transcript artifact could not be saved.")

    log.info(
        "audio_transcribe_done",
        file=file.filename,
        chunks=chunk_result["chunks_created"],
        vector_db=chunk_result.get("vector_db"),
        vectors=chunk_result.get("vectors_upserted"),
        translation_backend=chunk_result.get("translation_backend"),
        translation_version=chunk_result.get("translation_version"),
        transcript_file=transcript_file_name,
    )
    run_id = record_ingestion_run(
        id=run_id,
        kind="audio",
        source=file_metadata["source"],
        source_title=file_metadata["source_title"],
        file_name=file.filename or "",
        language=language_code,
        dataset_id=ds_id,
        status="ok",
        chunks=chunk_result["chunks_created"],
        vectors=chunk_result.get("vectors_upserted") or 0,
        vector_db=chunk_result.get("vector_db"),
        embedding_backend=chunk_result.get("embedding_backend"),
        transcription_mode=transcription_mode,
        transcription_version=transcription_version,
        translation_backend=chunk_result.get("translation_backend"),
        translation_version=chunk_result.get("translation_version"),
        metadata=run_metadata({
            "transcript_file": str(transcript_path),
            "stage": "complete",
        }),
    )
    _append_audio_audit({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "role": "audio_transcript_indexed",
        "file_path": source_file_path,
        "original_filename": file.filename,
        "bytes": len(audio_bytes),
        "sha256": content_sha256,
        "source": file_metadata["source"],
        "source_title": file_metadata["source_title"],
        "language_code": language_code,
        "section": section,
        "transcript_file": str(transcript_path),
        "chunks_created": chunk_result["chunks_created"],
        "vectors_upserted": chunk_result.get("vectors_upserted"),
        "vector_db": chunk_result.get("vector_db"),
        "embedding_backend": chunk_result.get("embedding_backend"),
        "transcription_mode": transcription_mode,
        "transcription_version": transcription_version,
        "translation_backend": chunk_result.get("translation_backend"),
        "translation_version": chunk_result.get("translation_version"),
    })

    return AudioTranscribeResponse(
        transcript=transcript_text,
        translated_transcript=chunk_result.get("translated_transcript"),
        chunks_created=chunk_result["chunks_created"],
        file_name=file.filename,
        transcript_file_name=transcript_file_name,
        translation_mode=chunk_result.get("translation_mode"),
        translation_backend=chunk_result.get("translation_backend"),
        translation_version=chunk_result.get("translation_version"),
        translation_fallback_reason=chunk_result.get("translation_fallback_reason"),
    )
