"""
transcribe_audio_batch.py — split audio files into 29s clips and transcribe each via the API.

For each source audio file found in the input directory:
1. Splits the file into persistent 29-second MP3 clips using ffmpeg.
2. Uploads each clip to the running DharmaGPT transcription API endpoint.
3. Writes a batch manifest JSON with transcript file names and per-chunk results.

The API writes each transcript as a canonical JSONL artifact under
knowledge/processed/, so this script is the orchestration layer that turns
large source recordings into the chunked pipeline format the system expects.
Requires the DharmaGPT API server to be running (see --api-url).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    import imageio_ffmpeg
except ImportError as exc:  # pragma: no cover - bundled runtime should provide this
    raise SystemExit("imageio_ffmpeg is required to segment audio") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / "dharmagpt" / ".env")

SUPPORTED = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dharmagpt.utils.naming import normalize_language_tag, slugify  # noqa: E402


def resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    repo_path = REPO_ROOT / value
    if repo_path.exists():
        return repo_path
    return path


def discover_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED and "clips_29s" not in p.parts
    ]
    return sorted(files)


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def split_audio_to_chunks(source: Path, chunk_dir: Path, segment_seconds: int, overwrite: bool) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunk_dir.glob("*.mp3"))
    if existing and not overwrite:
        return existing

    for old in existing:
        old.unlink()

    source_slug = slugify(source.stem, default="audio")
    out_pattern = str(chunk_dir / f"{source_slug}_te_audio_part%04d.mp3")
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        "-c:a",
        "libmp3lame",
        out_pattern,
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed for {source.name}: {run.stderr.strip() or 'unknown error'}")

    chunks = sorted(chunk_dir.glob("*.mp3"))
    if not chunks:
        raise RuntimeError(f"ffmpeg produced no segments for {source.name}")
    return chunks


def expected_transcript_path(chunk_path: Path, language_tag: str) -> Path:
    from dharmagpt.utils.naming import canonical_jsonl_filename, part_number_from_filename, source_stem_from_audio_filename

    part = part_number_from_filename(chunk_path.name)
    base = source_stem_from_audio_filename(chunk_path.name, language=language_tag)
    fname = canonical_jsonl_filename(base, language=language_tag, kind="transcript", part=part)
    return REPO_ROOT / "dharmagpt" / "knowledge" / "processed" / "audio_transcript" / base / fname


def expected_transcript_name(chunk_path: Path, language_tag: str) -> str:
    from dharmagpt.utils.naming import canonical_jsonl_filename, part_number_from_filename, source_stem_from_audio_filename

    part = part_number_from_filename(chunk_path.name)
    base = source_stem_from_audio_filename(chunk_path.name, language=language_tag)
    return canonical_jsonl_filename(
        base,
        language=language_tag,
        kind="transcript",
        part=part,
    )


def upload_chunk(
    chunk_path: Path,
    *,
    language_code: str,
    description: str,
    api_url: str,
    admin_api_key: str | None,
    timeout: int,
    retries: int,
    retry_delay: float,
) -> dict:
    mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".opus": "audio/opus",
        ".wma": "audio/x-ms-wma",
    }
    mime = mime_map.get(chunk_path.suffix.lower(), "audio/mpeg")
    headers = {"X-API-Key": admin_api_key} if admin_api_key else None
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with chunk_path.open("rb") as fh:
                resp = requests.post(
                    api_url,
                    headers=headers,
                    files={"file": (chunk_path.name, fh, mime)},
                    data={"language_code": language_code, "description": description},
                    timeout=timeout,
                )
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"{resp.status_code} Server Error: {resp.text[:500]}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay * attempt)
    if last_error:
        raise last_error
    raise RuntimeError("upload failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment audio sources into 29s clips and transcribe them")
    parser.add_argument("--input-dir", help="Directory with source audio files")
    parser.add_argument("--file", help="Single source audio file to process")
    parser.add_argument("--output-dir", default="downloads/clips_29s_full", help="Directory for persistent 29s chunks")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild clips even if they already exist")
    parser.add_argument("--split-only", action="store_true", help="Only create persistent clips; do not call the API")
    parser.add_argument("--segment-seconds", type=int, default=29, help="Chunk length in seconds")
    parser.add_argument("--language-code", default="te-IN", help="Sarvam language code for transcription")
    parser.add_argument("--language-tag", default="te", help="Short language tag used in filenames")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1/audio/transcribe", help="Local transcription API URL")
    parser.add_argument(
        "--admin-api-key",
        default=(
            os.getenv("ADMIN_API_KEY")
            or os.getenv("ADMIN_OPERATOR_API_KEY")
            or os.getenv("STAGING_API_KEY")
            or ""
        ),
        help="Admin key for protected API routes (defaults to ADMIN_API_KEY / ADMIN_OPERATOR_API_KEY / STAGING_API_KEY)",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Upload timeout per chunk in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for transient upload failures")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Base delay between retries in seconds")
    parser.add_argument("--chunk-delay", type=float, default=1.0, help="Delay between chunk uploads in seconds")
    args = parser.parse_args()

    if bool(args.file) == bool(args.input_dir):
        raise SystemExit("Pass exactly one of --file or --input-dir")

    input_dir: Path | None = None
    if args.file:
        target_file = resolve_input_path(args.file)
        if not target_file.exists() or not target_file.is_file():
            raise SystemExit(f"Input file not found: {target_file}")
        if target_file.suffix.lower() not in SUPPORTED:
            supported = ", ".join(sorted(SUPPORTED))
            raise SystemExit(f"Unsupported audio file: {target_file}. Supported: {supported}")
        files = [target_file]
    else:
        input_dir = resolve_input_path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")
        files = discover_audio_files(input_dir, recursive=args.recursive)
        if not files:
            raise SystemExit(f"No supported audio files found in {input_dir}")

    output_dir = resolve_input_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    language_tag = normalize_language_tag(args.language_tag)
    admin_api_key = (args.admin_api_key or "").strip()

    summary: list[dict] = []
    total_chunks = 0
    total_ok = 0
    total_failed = 0

    print(f"Found {len(files)} source file(s)")
    print(f"Input:  {input_dir or files[0]}")
    print(f"Output: {output_dir}")
    print(f"Chunk:  {args.segment_seconds}s")
    print(f"API:    {args.api_url}")
    print(f"Mode:   {'split only' if args.split_only else 'transcribe'}")
    print(f"Auth:   {'enabled' if admin_api_key else 'disabled'}\n")

    for index, source in enumerate(files, start=1):
        source_slug = slugify(source.stem, default="audio")
        chunk_dir = output_dir / source_slug
        print(f"[{index}/{len(files)}] {source.name}")
        try:
            chunks = split_audio_to_chunks(source, chunk_dir, args.segment_seconds, args.overwrite)
        except Exception as exc:
            print(f"  split failed: {exc}")
            summary.append(
                {
                    "source": str(source),
                    "status": "split_failed",
                    "error": str(exc),
                }
            )
            total_failed += 1
            continue

        source_result = {
            "source": str(source),
            "chunk_dir": str(chunk_dir),
            "chunks": [],
            "status": "ok",
        }

        for chunk_idx, chunk_path in enumerate(chunks, start=1):
            total_chunks += 1
            expected_name = expected_transcript_name(chunk_path, language_tag)
            expected_path = expected_transcript_path(chunk_path, language_tag)
            if args.split_only:
                print(f"  - split {chunk_path.name}")
                source_result["chunks"].append(
                    {
                        "chunk": str(chunk_path),
                        "status": "split_only",
                        "transcript_file_name": expected_name,
                    }
                )
                continue
            if expected_path.exists() and not args.overwrite:
                print(f"  - skip {chunk_path.name} (transcript exists)")
                source_result["chunks"].append(
                    {
                        "chunk": str(chunk_path),
                        "status": "skipped_existing",
                        "transcript_file_name": expected_name,
                    }
                )
                continue

            description = f"{source.stem} chunk {chunk_idx}"
            try:
                response = upload_chunk(
                    chunk_path,
                    language_code=args.language_code,
                    description=description,
                    api_url=args.api_url,
                    admin_api_key=admin_api_key,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_delay=args.retry_delay,
                )
                total_ok += 1
                transcript_name = response.get("transcript_file_name") or expected_name
                print(f"  - ok {chunk_path.name} -> {transcript_name}")
                source_result["chunks"].append(
                    {
                        "chunk": str(chunk_path),
                        "status": "ok",
                        "response": response,
                        "transcript_file_name": transcript_name,
                    }
                )
                time.sleep(args.chunk_delay)
            except Exception as exc:
                total_failed += 1
                print(f"  - fail {chunk_path.name}: {exc}")
                source_result["chunks"].append(
                    {
                        "chunk": str(chunk_path),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                time.sleep(args.chunk_delay)

        summary.append(source_result)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "language_code": args.language_code,
        "language_tag": language_tag,
        "segment_seconds": args.segment_seconds,
        "sources": len(files),
        "chunks": total_chunks,
        "ok": total_ok,
        "failed": total_failed,
        "results": summary,
    }

    manifest_path = output_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBatch complete")
    print(f"  Sources: {len(files)}")
    print(f"  Chunks:  {total_chunks}")
    print(f"  OK:      {total_ok}")
    print(f"  Failed:  {total_failed}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
