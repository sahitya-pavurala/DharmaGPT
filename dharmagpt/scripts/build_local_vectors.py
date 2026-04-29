"""
build_local_vectors.py — embed all processed JSONL records and write to local_vectors.sqlite3.

Reads every JSONL under knowledge/processed/, embeds the best available text (text_en >
text_en_model > text), and upserts into local_vectors.sqlite3 for local RAG + later
migration to Pinecone.

Resume-safe: records already present (by id) are skipped unless --force is passed.

Usage:
    python scripts/build_local_vectors.py                    # all files
    python scripts/build_local_vectors.py --source-type text # only text corpus
    python scripts/build_local_vectors.py --source-type audio_transcript
    python scripts/build_local_vectors.py --dry-run          # count only, no writes
    python scripts/build_local_vectors.py --force            # re-embed all
    python scripts/build_local_vectors.py --limit 100        # stop after N records
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DHARMAGPT_DIR = REPO_ROOT / "dharmagpt"
load_dotenv(DHARMAGPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")  # fallback

if str(DHARMAGPT_DIR) not in sys.path:
    sys.path.insert(0, str(DHARMAGPT_DIR))

from core.config import get_settings
from core.local_vector_store import upsert_vectors, _connect

log = structlog.get_logger()
settings = get_settings()

PROCESSED_DIR = DHARMAGPT_DIR / "knowledge" / "processed"
EMBED_BATCH = 20        # texts per OpenAI call
UPSERT_BATCH = 100      # records per SQLite upsert
EMBED_DELAY = 0.2       # seconds between OpenAI batches
MAX_TEXT_CHARS = 2500   # truncate before embedding


def get_openai_client():
    from openai import OpenAI
    key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        sys.exit("ERROR: OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def embed_batch(texts: list[str], client, model: str, dims: int) -> list[list[float]]:
    truncated = [t[:MAX_TEXT_CHARS] for t in texts]
    resp = client.embeddings.create(input=truncated, model=model, dimensions=dims)
    return [item.embedding for item in resp.data]


def get_existing_ids() -> set[str]:
    """Return all ids already in local_vectors."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id FROM vector_chunks WHERE index_name = ?",
            (settings.local_vector_index_name,),
        ).fetchall()
    return {row["id"] for row in rows}


def best_embed_text(record: dict) -> str:
    """Pick the richest available text for embedding."""
    for field in ("text_en", "text_en_model", "text"):
        val = record.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def build_metadata(record: dict) -> dict:
    meta = {
        "text": record.get("text_en") or record.get("text") or "",
        "citation": record.get("citation") or "",
        "section": record.get("kanda") or record.get("section") or "",
        "source": record.get("source") or "",
        "source_type": record.get("source_type") or "text",
        "language": record.get("language") or "en",
        "url": record.get("url") or "",
        "tags": record.get("tags") or [],
        "characters": record.get("characters") or [],
        "is_shloka": record.get("is_shloka") or False,
    }
    # audio-specific
    for field in ("description", "speaker_type", "source_file",
                  "translation_backend", "transcription_mode"):
        if record.get(field):
            meta[field] = record[field]
    # text-specific
    for field in ("sarga", "kanda", "verse_start", "verse_end"):
        if record.get(field) is not None:
            meta[field] = record[field]
    return meta


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def discover_files(source_type_filter: str | None) -> list[Path]:
    files = sorted(PROCESSED_DIR.glob("**/*.jsonl"))
    if source_type_filter:
        files = [f for f in files if source_type_filter in str(f)]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed corpus JSONL → local_vectors.sqlite3")
    parser.add_argument("--source-type", default=None,
                        help="Filter by source type path segment: 'text', 'audio_transcript'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count records but skip embedding and writing")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed even if id already in local_vectors")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N records total (0 = all)")
    args = parser.parse_args()

    files = discover_files(args.source_type)
    if not files:
        sys.exit(f"No JSONL files found under {PROCESSED_DIR}")

    client = None if args.dry_run else get_openai_client()
    existing_ids = set() if (args.dry_run or args.force) else get_existing_ids()
    model = settings.embedding_model or "text-embedding-3-large"
    dims = settings.embedding_dims or 3072
    index_name = settings.local_vector_index_name
    namespace = settings.local_vector_namespace

    print(f"Files: {len(files)} | Model: {model} ({dims}d) | Index: {index_name}/{namespace}")
    if not args.dry_run:
        print(f"Already in store: {len(existing_ids):,} records (will skip unless --force)")
    print()

    total_embedded = 0
    total_skipped = 0
    total_failed = 0

    pending_texts: list[str] = []
    pending_records: list[dict] = []

    def flush():
        nonlocal total_embedded, total_failed
        if not pending_texts:
            return
        try:
            vectors = embed_batch(pending_texts, client, model, dims)
        except Exception as exc:
            log.error("embed_failed", error=str(exc), batch_size=len(pending_texts))
            total_failed += len(pending_texts)
            pending_texts.clear()
            pending_records.clear()
            return

        upsert_payload = []
        for record, vec in zip(pending_records, vectors):
            upsert_payload.append({
                "id": record["id"],
                "values": vec,
                "metadata": build_metadata(record),
            })

        upsert_vectors(index_name=index_name, namespace=namespace, records=upsert_payload)
        total_embedded += len(upsert_payload)
        pending_texts.clear()
        pending_records.clear()
        time.sleep(EMBED_DELAY)

    for path in files:
        records = load_records(path)
        if not records:
            continue

        file_new = 0
        for record in records:
            if args.limit and (total_embedded + total_skipped + total_failed) >= args.limit:
                break

            rec_id = record.get("id")
            if not rec_id:
                total_skipped += 1
                continue

            if rec_id in existing_ids:
                total_skipped += 1
                continue

            text = best_embed_text(record)
            if not text:
                total_skipped += 1
                continue

            if args.dry_run:
                total_embedded += 1
                file_new += 1
                continue

            pending_texts.append(text)
            pending_records.append(record)
            file_new += 1

            if len(pending_texts) >= EMBED_BATCH:
                flush()

        if file_new:
            print(f"  {path.relative_to(PROCESSED_DIR)}: +{file_new} new")

    if not args.dry_run:
        flush()

    print()
    print("-" * 55)
    label = "Would embed" if args.dry_run else "Embedded"
    print(f"{label}:  {total_embedded:,}")
    print(f"Skipped:   {total_skipped:,} (already in store or no text)")
    if total_failed:
        print(f"Failed:    {total_failed:,}")


if __name__ == "__main__":
    main()
