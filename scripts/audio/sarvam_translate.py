"""
scripts/audio/sarvam_translate.py

Prototype Telugu audio pipeline:
1. Split audio into 5 × 29-second chunks
2. Transcribe each chunk via Sarvam STT (Telugu)
3. Translate each transcript from Telugu to English (Anthropic or local Ollama)
4. Optionally attach manual English translation in parallel
5. Output JSONL with Telugu + model English + manual English

Usage:
    python scripts/audio/sarvam_translate.py --input data/audio/sample.mp3 --chunks 5
    python scripts/audio/sarvam_translate.py --input downloads/.../part_1.mp3 --output translated_chunks.jsonl
"""

import os
import sys
import json
import time
import uuid
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Any
from tqdm import tqdm
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dharmagpt.core.translation import TranslationBackend, TranslationConfig, translate_text

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv(REPO_ROOT / "dharmagpt" / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
MAX_SEGMENT_MS = 29_000
SUPPORTED = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma"}


def _ffmpeg_path() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _check_keys():
    """Validate API keys are present."""
    if not SARVAM_API_KEY:
        raise RuntimeError(
            f"SARVAM_API_KEY is not set. Add it to {REPO_ROOT / 'dharmagpt' / '.env'}"
        )


def split_audio(path: Path, num_chunks: int) -> list[Path]:
    """
    Split audio into `num_chunks` equal-length segments.
    Returns list of temp file paths (caller manages cleanup).
    """
    ffmpeg = _ffmpeg_path()
    tmp_dir = Path(tempfile.mkdtemp(prefix="sarvam_translate_"))

    # Sarvam STT has a strict 30-second limit; keep a 1-second buffer.
    segment_time_sec = MAX_SEGMENT_MS // 1000
    
    print(f"  Splitting audio into {segment_time_sec}s segments for Sarvam STT (30s limit)...")
    
    # Split using ffmpeg
    out_pattern = str(tmp_dir / "chunk_%03d.mp3")
    split_cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(path),
        "-f", "segment",
        "-segment_time", str(segment_time_sec),
        "-c:a", "libmp3lame",
        out_pattern,
    ]
    run = subprocess.run(split_cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {run.stderr}")

    chunks = sorted(tmp_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no segments")
    
    # If num_chunks <= 0, process all 29-second chunks.
    if num_chunks <= 0:
        print(f"  ✓ Created {len(chunks)} segments (processing all)")
    # If we got more than requested, take only the first num_chunks.
    elif len(chunks) > num_chunks:
        chunks = chunks[:num_chunks]
        print(f"  ✓ Created {len(chunks)} segments (limited to {num_chunks} requested)")
    else:
        print(f"  ✓ Created {len(chunks)} segments")
    
    return chunks


def transcribe_chunk(path: Path, lang: str = "te-IN") -> str:
    """Transcribe a single audio chunk via Sarvam STT (Telugu)."""
    mime_map = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
        ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
        ".opus": "audio/opus", ".wma": "audio/x-ms-wma",
    }
    mime = mime_map.get(path.suffix.lower(), "audio/mpeg")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    resp = requests.post(
        SARVAM_STT_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        files={"file": (path.name, audio_bytes, mime)},
        data={
            "model": "saaras:v3",
            "language_code": lang,
            "with_timestamps": "false",
        },
        timeout=120,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if resp.status_code in {401, 403}:
            raise RuntimeError("Sarvam auth failed (401/403). Check SARVAM_API_KEY.") from exc
        msg = ""
        try:
            msg = (resp.json().get("error") or {}).get("message", "")
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Sarvam STT failed ({resp.status_code}): {msg}") from exc

    data = resp.json()
    return (data.get("transcript") or "").strip()


def translate_telugu_to_english(
    telugu_text: str,
    translator: str,
    anthropic_model: str,
    anthropic_api_key: str,
    ollama_model: str,
    ollama_url: str,
    indictrans2_model: str,
) -> tuple[str, str]:
    """Translate Telugu text to English using a selectable backend."""
    backend = TranslationBackend(translator)
    config = TranslationConfig(
        backend=backend,
        anthropic_model=anthropic_model,
        anthropic_api_key=anthropic_api_key,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        indictrans2_model=indictrans2_model,
        indictrans2_src_lang="tel_Telu",
        indictrans2_tgt_lang="eng_Latn",
    )
    try:
        return translate_text(telugu_text, config=config, source_lang="tel_Telu", target_lang="eng_Latn")
    except Exception as exc:
        return f"[Translation failed via {translator}: {exc}]", translator


def _extract_chunk_index(obj: dict[str, Any]) -> int | None:
    idx = obj.get("chunk_index")
    if idx is None and isinstance(obj.get("metadata"), dict):
        idx = obj["metadata"].get("chunk_index")
    if isinstance(idx, int):
        return idx
    return None


def _extract_manual_text(obj: dict[str, Any]) -> str | None:
    candidates = ["text_en_manual", "manual_translation", "text_en"]
    for key in candidates:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_manual_translations(path: Path | None) -> dict[int, str]:
    """
    Load manual translations keyed by chunk index.

    Supported formats:
    - JSONL: one JSON object per line with `chunk_index` and `text_en_manual`
    - JSON: list[object] or dict[int -> text]
    """
    if path is None:
        return {}
    if not path.exists():
        raise RuntimeError(f"Manual translations file not found: {path}")

    result: dict[int, str] = {}
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    continue
                idx = _extract_chunk_index(obj)
                txt = _extract_manual_text(obj)
                if idx is not None and txt:
                    result[idx] = txt
        return result

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for k, v in data.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                result[idx] = v.strip()
        return result

    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            idx = _extract_chunk_index(obj)
            txt = _extract_manual_text(obj)
            if idx is not None and txt:
                result[idx] = txt
        return result

    return result


def build_chunk_record(
    telugu_text: str,
    english_text_model: str,
    english_text_manual: str,
    chunk_idx: int,
    audio_name: str,
    translation_source: str,
) -> dict:
    """Build a JSONL-ready chunk record with both languages."""
    return {
        "id": f"audio_telugu_{audio_name}_{uuid.uuid4().hex[:6]}_{chunk_idx:04d}",
        "text_te": telugu_text,
        "text_en": english_text_model,
        "text_en_model": english_text_model,
        "text_en_manual": english_text_manual,
        "metadata": {
            "source_type": "audio_transcript",
            "source_file": audio_name,
            "language_original": "te",
            "language": "en",
            "chunk_index": chunk_idx,
            "translation_source": translation_source,
            "word_count_te": len(telugu_text.split()),
            "word_count_en": len(english_text_model.split()),
            "word_count_en_manual": len(english_text_manual.split()) if english_text_manual else 0,
            "has_manual_translation": bool(english_text_manual),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe and translate Telugu audio chunks"
    )
    parser.add_argument("--input", required=True, help="Audio file path")
    parser.add_argument("--output", default=None, help="Output JSONL file (default: auto)")
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,
        help="Number of chunks to process; use 0 or negative to process all chunks",
    )
    parser.add_argument(
        "--translator",
        choices=["auto", "skip", "anthropic", "ollama", "indictrans2"],
        default="auto",
        help="Translation backend: auto (local Ollama -> Anthropic), skip, anthropic, ollama, indictrans2",
    )
    parser.add_argument(
        "--manual-translations",
        default=None,
        help="Optional JSON/JSONL file with manual English text by chunk index (stored in parallel)",
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-3-5-sonnet-20241022",
        help="Anthropic model name",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:7b",
        help="Local Ollama model name",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for local Ollama server",
    )
    parser.add_argument(
        "--indictrans2-model",
        default="ai4bharat/indictrans2-indic-en-dist-200M",
        help="Hugging Face model name or local path for IndicTrans2",
    )
    args = parser.parse_args()

    _check_keys()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: {inp} not found")
        sys.exit(1)

    if inp.suffix.lower() not in SUPPORTED:
        print(f"Error: unsupported format {inp.suffix}")
        sys.exit(1)

    out_file = Path(args.output) if args.output else Path(f"telugu_chunks_{inp.stem}.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("\nTelugu to English Audio Translation Pipeline")
    print(f"   Input: {inp.name}")
    print(f"   Chunks: {args.chunks}")
    print(f"   Translator: {args.translator}")
    print(f"   Output: {out_file.name}\n")

    manual_path = Path(args.manual_translations) if args.manual_translations else None
    manual_map = load_manual_translations(manual_path)
    if manual_map:
        print(f"   Manual parallel translations loaded: {len(manual_map)}\n")

    # Split audio
    chunk_paths = split_audio(inp, args.chunks)
    print()

    # Process each chunk
    records = []
    for i, chunk_path in enumerate(tqdm(chunk_paths, desc="Processing chunks", unit="chunk")):
        # Transcribe Telugu
        telugu_text = transcribe_chunk(chunk_path, lang="te-IN")
        if not telugu_text:
            print(f"  ⚠ Chunk {i} produced empty transcript, skipping")
            continue

        # Always produce model translation; keep manual translation in parallel.
        english_text_model, translation_source = translate_telugu_to_english(
            telugu_text=telugu_text,
            translator=args.translator,
            anthropic_model=args.anthropic_model,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            indictrans2_model=args.indictrans2_model,
        )
        english_text_manual = manual_map.get(i, "")

        # Build record
        record = build_chunk_record(
            telugu_text=telugu_text,
            english_text_model=english_text_model,
            english_text_manual=english_text_manual,
            chunk_idx=i,
            audio_name=inp.stem,
            translation_source=translation_source,
        )
        records.append(record)

        print(f"\n  [{i+1}/{len(chunk_paths)}]")
        print(f"  Telugu: {telugu_text[:80]}...")
        print(f"  English (model): {english_text_model[:80]}...")
        if english_text_manual:
            print(f"  English (manual): {english_text_manual[:80]}...")

        time.sleep(0.5)  # Rate limiting

    # Write output
    print(f"\n  Writing {len(records)} chunks to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅  Complete! {len(records)} translated chunks → {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
