"""
ingest_to_pinecone.py — embed corpus records and upsert vectors to Pinecone.

Reads JSONL files from knowledge/processed/, validates each record, embeds the
text using OpenAI text-embedding-3-large (3072 dims), and upserts to the
configured Pinecone index. Supports dry-run validation and full index re-builds.

Usage:
    python scripts/ingest_to_pinecone.py                          # all JSONL in knowledge/processed/
    python scripts/ingest_to_pinecone.py --file seed_corpus.jsonl # specific file
    python scripts/ingest_to_pinecone.py --dry-run                # validate + count, no upsert
    python scripts/ingest.py --delete-index           # wipe index and re-ingest
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Generator

import structlog
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from utils.naming import dataset_id_from_path, is_canonical_part_file

load_dotenv()
log = structlog.get_logger()

# ─── Config ───────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "knowledge" / "processed"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "3072"))
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "dharma-gpt")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
BATCH_SIZE = 50          # records per Pinecone upsert batch
EMBED_BATCH_SIZE = 20    # texts per OpenAI embeddings call
MAX_TEXT_CHARS = 2000    # truncate text before embedding if needed

REQUIRED_FIELDS = {"id", "text", "source", "citation", "language", "source_type",
                   "tags", "is_shloka"}

# ─── Clients ──────────────────────────────────────────────────────────────────

def get_openai() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("❌  OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def get_pinecone_index(pc: Pinecone):
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        log.info("creating_pinecone_index", name=PINECONE_INDEX)
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )
        # Wait for index to be ready
        for _ in range(30):
            status = pc.describe_index(PINECONE_INDEX).status
            if status.get("ready"):
                break
            time.sleep(2)
    return pc.Index(PINECONE_INDEX)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_record(record: dict, line_no: int, filename: str) -> list[str]:
    """Returns list of validation errors (empty = valid)."""
    errors = []
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        errors.append(f"{filename}:{line_no} missing fields: {missing}")
    if "text" in record and len(record["text"].strip()) < 20:
        errors.append(f"{filename}:{line_no} text too short (<20 chars)")
    if "id" in record and not record["id"].strip():
        errors.append(f"{filename}:{line_no} empty id")
    if "language" in record and record["language"] not in {"sa", "en", "hi", "te", "ta"}:
        errors.append(f"{filename}:{line_no} unknown language: {record['language']}")
    if "source_type" in record and record["source_type"] not in {"text", "commentary", "audio_transcript"}:
        errors.append(f"{filename}:{line_no} unknown source_type: {record['source_type']}")
    return errors


# ─── Loading ──────────────────────────────────────────────────────────────────

def iter_records(files: list[Path]) -> Generator[tuple[dict, str, str], None, None]:
    """Yield (record, filename, dataset_id) for all valid records across files."""
    total_errors = 0
    for f in files:
        dataset_id = dataset_id_from_path(f, root=PROCESSED_DIR)
        log.info("reading_file", path=str(f))
        with f.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    log.error("json_parse_error", file=f.name, line=line_no, error=str(e))
                    total_errors += 1
                    continue
                errors = validate_record(record, line_no, f.name)
                if errors:
                    for err in errors:
                        log.warning("validation_error", msg=err)
                    total_errors += 1
                    continue
                yield record, f.name, dataset_id
    if total_errors:
        log.warning("validation_summary", total_errors=total_errors)


# ─── Embedding ────────────────────────────────────────────────────────────────

def batch_embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts; respects EMBED_BATCH_SIZE."""
    all_vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        truncated = [t[:MAX_TEXT_CHARS] for t in batch]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=truncated)
        all_vectors.extend([r.embedding for r in response.data])
        log.debug("embedded_batch", start=i, count=len(batch))
    return all_vectors


# ─── Pinecone Metadata Builder ───────────────────────────────────────────────

def build_metadata(record: dict, dataset_id: str) -> dict:
    """Extract Pinecone-storable metadata from a corpus record."""
    # Pinecone metadata values must be str, int, float, bool, or list[str]
    meta: dict = {
        "source": record.get("source", ""),
        "source_type": record.get("source_type", "text"),
        "citation": record.get("citation", ""),
        "language": record.get("language", "en"),
        "is_shloka": bool(record.get("is_shloka", False)),
        "tags": record.get("tags", []),
        "characters": record.get("characters", []),
        "topics": record.get("topics", []),
        "text_preview": record.get("text", "")[:500],
        "url": record.get("url", ""),
        "has_telugu": bool(record.get("text_te", "").strip()),
        "has_english": bool(record.get("text_en", "").strip()),
        "dataset_id": dataset_id,
    }
    if record.get("kanda"):
        meta["kanda"] = record["kanda"]
    if record.get("sarga") is not None:
        meta["sarga"] = int(record["sarga"])
    if record.get("verse_start") is not None:
        meta["verse_start"] = int(record["verse_start"])
    if record.get("verse_end") is not None:
        meta["verse_end"] = int(record["verse_end"])
    return meta


# ─── Main Ingestion ───────────────────────────────────────────────────────────

def build_embed_text(record: dict) -> str:
    """Construct the text to embed — combines multilingual fields for richer retrieval."""
    parts = [record["text"]]
    if record.get("text_te"):
        parts.append(record["text_te"])
    if record.get("text_en") and record.get("language") != "en":
        parts.append(record["text_en"])
    # Append citation for semantic grounding
    parts.append(record.get("citation", ""))
    return " | ".join(p for p in parts if p.strip())


def ingest(files: list[Path], dry_run: bool = False, delete_index: bool = False) -> None:
    openai_client = None

    if not dry_run:
        openai_client = get_openai()
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        if delete_index:
            existing = [i["name"] for i in pc.list_indexes()]
            if PINECONE_INDEX in existing:
                log.warning("deleting_index", name=PINECONE_INDEX)
                pc.delete_index(PINECONE_INDEX)
                time.sleep(5)
        index = get_pinecone_index(pc)
    else:
        index = None

    # Collect all records
    records_list = list(iter_records(files))
    log.info("corpus_loaded", total_valid=len(records_list))

    if dry_run:
        log.info("dry_run_complete", would_ingest=len(records_list))
        print(f"\n✅  Dry run: {len(records_list)} valid records found across {len(files)} file(s).")
        return

    # Build embed texts
    embed_texts = [build_embed_text(r) for r, _, _ in records_list]
    log.info("embedding_start", count=len(embed_texts))
    vectors = batch_embed(embed_texts, openai_client)
    log.info("embedding_done", count=len(vectors))

    # Build Pinecone records
    pinecone_records = []
    for (record, _, dataset_id), vector in zip(records_list, vectors):
        pinecone_records.append({
            "id": record["id"],
            "values": vector,
            "metadata": build_metadata(record, dataset_id=dataset_id),
        })

    # Upsert in batches
    upserted = 0
    for i in range(0, len(pinecone_records), BATCH_SIZE):
        batch = pinecone_records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        upserted += len(batch)
        log.info("upserted_batch", done=upserted, total=len(pinecone_records))

    log.info("ingestion_complete", total_upserted=upserted)
    print(f"\n✅  Ingestion complete: {upserted} chunks upserted to '{PINECONE_INDEX}'.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest DharmaGPT text corpus into Pinecone")
    parser.add_argument("--file", type=str, default=None,
                        help="Ingest a single JSONL file from knowledge/processed/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate records without embedding or upserting")
    parser.add_argument("--delete-index", action="store_true",
                        help="Delete the Pinecone index before ingestion (fresh re-index)")
    parser.add_argument("--recursive", action="store_true", help="Scan knowledge/processed recursively")
    parser.add_argument(
        "--partitioned-only",
        action="store_true",
        help="Only ingest partitioned files named part-*.jsonl",
    )
    args = parser.parse_args()

    if args.file:
        target = PROCESSED_DIR / args.file
        if not target.exists():
            sys.exit(f"❌  File not found: {target}")
        files = [target]
    else:
        pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
        files = sorted(PROCESSED_DIR.glob(pattern))
        if args.partitioned_only:
            files = [f for f in files if is_canonical_part_file(f)]
        if not files:
            sys.exit(f"❌  No .jsonl files found in {PROCESSED_DIR}")

    print(f"📚  Files to ingest: {[str(f.relative_to(PROCESSED_DIR)) for f in files]}")
    ingest(files, dry_run=args.dry_run, delete_index=args.delete_index)


if __name__ == "__main__":
    main()
