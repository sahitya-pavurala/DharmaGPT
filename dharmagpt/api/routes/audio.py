import httpx
import structlog
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import AudioTranscribeResponse
from core.config import get_settings
from pipelines.audio_chunker import chunk_and_index

router = APIRouter()
log = structlog.get_logger()
settings = get_settings()

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}


@router.post("/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
    kanda: str = Form(None),
    description: str = Form(None),
) -> AudioTranscribeResponse:
    """
    Upload a Sanskrit/Hindi audio file (chanting, pravachanam, discourse).
    Transcribes via Sarvam Saaras v3, chunks intelligently, indexes to Pinecone.
    """
    suffix = "." + (file.filename or "").split(".")[-1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {SUPPORTED_FORMATS}")

    audio_bytes = await file.read()
    if len(audio_bytes) > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 100MB.")

    log.info("audio_transcribe_start", file=file.filename, lang=language_code, size_mb=round(len(audio_bytes)/1e6, 2))

    # Call Sarvam STT
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": settings.sarvam_api_key},
                files={"file": (file.filename, audio_bytes, f"audio/{suffix.lstrip('.')}")},
                data={
                    "model": "saaras:v3",
                    "language_code": language_code,
                    "with_timestamps": "true",
                    "with_diarization": "true",
                },
            )
            response.raise_for_status()
            transcript_data = response.json()
    except httpx.HTTPError as e:
        log.error("sarvam_stt_error", error=str(e))
        raise HTTPException(status_code=502, detail="Audio transcription service unavailable.")

    transcript_text = transcript_data.get("transcript", "")
    if not transcript_text:
        raise HTTPException(status_code=422, detail="No speech detected in audio file.")

    # Chunk and index
    file_metadata = {
        "language_code": language_code,
        "kanda": kanda,
        "description": description or file.filename,
        "text_source": "Valmiki Ramayana",
        "source_file": file.filename,
    }
    chunks_created = await chunk_and_index(transcript_data, file.filename, file_metadata)

    log.info("audio_transcribe_done", file=file.filename, chunks=chunks_created)

    return AudioTranscribeResponse(
        transcript=transcript_text,
        chunks_created=chunks_created,
        file_name=file.filename,
    )
