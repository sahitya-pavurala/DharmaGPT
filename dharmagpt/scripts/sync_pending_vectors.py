#!/usr/bin/env python3
"""
sync_pending_vectors.py — incrementally embed pending Postgres chunks into Pinecone.

Upload flows write chunks to chunk_store with vector_status='pending'. This script
claims the next batch only, upserts those vectors, and marks just those chunk IDs
as indexed. It never scans/reindexes the full table unless rows are reset.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def _run(args) -> None:
    from core.vector_sync import sync_pending_chunks_to_pinecone

    result = await sync_pending_chunks_to_pinecone(
        limit=args.limit,
        index_name=args.index_name,
        namespace=args.namespace,
        source=args.source,
        dataset_id=args.dataset_id,
        create_index=args.create_index,
    )
    print(
        "selected={selected} vectors_upserted={vectors_upserted} "
        "index={index_name} namespace={namespace} embedding={embedding_backend}".format(**result)
    )
    print(f"vector_status={result.get('vector_status')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync pending chunk_store rows to Pinecone incrementally")
    parser.add_argument("--limit", type=int, default=100, help="Max pending chunks to sync in this run")
    parser.add_argument("--index-name", default="", help="Override Pinecone index name")
    parser.add_argument("--namespace", default="", help="Pinecone namespace")
    parser.add_argument("--source", default="", help="Only sync this source")
    parser.add_argument("--dataset-id", default="", help="Only sync this dataset_id")
    parser.add_argument("--create-index", action="store_true", help="Create the Pinecone index if missing")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
