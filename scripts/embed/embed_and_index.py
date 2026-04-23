"""
scripts/embed/embed_and_index.py

Reads JSONL chunk files, generates embeddings via OpenAI,
and upserts into Pinecone. Handles both text and audio chunks.

Usage:
    pip install openai pinecone python-dotenv tqdm
    python scripts/embed/embed_and_index.py --input data/chunks/sundara_chunks.jsonl
    python scripts/embed/embed_and_index.py --input data/chunks/  # entire directory
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv("../../dharmagpt/.env")

EMBED_MODEL    = "text-embedding-3-large"
EMBED_DIMS     = 3072
EMBED_BATCH    = 50    # OpenAI max is 2048 inputs; 50 is safe
UPSERT_BATCH   = 100   # Pinecone upsert batch size
MIN_TEXT_LEN   = 30    # skip trivially short chunks


def load_jsonl(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    vectors = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="  Embedding", leave=False):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(r.embedding for r in resp.data)
        time.sleep(0.25)
    return vectors


def flatten_meta(meta: dict) -> dict:
    """Pinecone metadata values must be str, int, float, bool, or list[str]."""
    flat = {}
    for k, v in meta.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, list):
            flat[k] = [str(x) for x in v]
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat


def upsert_chunks(index, chunks: list[dict], vectors: list[list[float]]):
    records = []
    for chunk, vec in zip(chunks, vectors):
        meta = flatten_meta(chunk.get("metadata", {}))
        meta["text_preview"] = chunk["text"][:300]
        records.append({"id": chunk["id"], "values": vec, "metadata": meta})

    for i in tqdm(range(0, len(records), UPSERT_BATCH), desc="  Upserting", leave=False):
        index.upsert(vectors=records[i : i + UPSERT_BATCH])
        time.sleep(0.1)


def process_file(path: Path, index, openai_client: OpenAI):
    print(f"\n── {path.name}")
    raw = load_jsonl(path)
    chunks = [c for c in raw if len(c.get("text", "")) >= MIN_TEXT_LEN]
    print(f"  {len(raw)} loaded → {len(chunks)} after filtering")

    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, openai_client)
    upsert_chunks(index, chunks, vectors)
    print(f"  ✓ {len(chunks)} chunks indexed")


def get_or_create_index(pc: Pinecone, name: str) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        print(f"Creating Pinecone index '{name}'...")
        pc.create_index(
            name=name,
            dimension=EMBED_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(8)
    return pc.Index(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="JSONL file or directory of JSONL files")
    parser.add_argument("--index",  default=os.getenv("PINECONE_INDEX_NAME", "dharma-gpt"))
    args = parser.parse_args()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = get_or_create_index(pc, args.index)

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        print(f"Found {len(files)} JSONL files in {input_path}")
        for f in files:
            process_file(f, index, openai_client)
    elif input_path.is_file():
        process_file(input_path, index, openai_client)
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    stats = index.describe_index_stats()
    print(f"\nPinecone index '{args.index}': {stats['total_vector_count']:,} total vectors")
