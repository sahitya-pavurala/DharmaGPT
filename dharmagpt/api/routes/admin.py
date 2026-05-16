from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core import dataset_store
from core.chunk_store import count_chunks_by_vector_status, upsert_chunk
from api.auth import require_admin_api_key
from core.config import get_settings
from core.insight_store import record_ingestion_run
from core.postgres_db import connect as pg_connect, ensure_schema as pg_ensure_schema, use_postgres
from core.retrieval import embed_texts, get_pinecone
from core.usage_stats import summarize_usage
from core.vector_sync import sync_pending_chunks_to_pinecone

router = APIRouter()
settings = get_settings()
KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
SOURCE_FILE_DIR = KNOWLEDGE_DIR / "uploads" / "source_files"
AUDIT_DIR = KNOWLEDGE_DIR / "audit"


def _safe_filename(filename: str) -> str:
  name = Path(filename or "upload.txt").name
  return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "upload.txt"


def _normalize_text(raw: str) -> str:
  return re.sub(r"\s+", " ", raw).strip()


def _chunk_text(raw: str, chunk_words: int = 650, overlap_words: int = 80) -> list[str]:
  text = _normalize_text(raw)
  if not text:
    return []

  words = text.split()
  if len(words) <= chunk_words:
    return [text] if len(words) >= 20 else []

  step = max(1, chunk_words - overlap_words)
  chunks: list[str] = []
  for start in range(0, len(words), step):
    chunk = " ".join(words[start : start + chunk_words]).strip()
    if len(chunk.split()) >= 20:
      chunks.append(chunk)
    if start + chunk_words >= len(words):
      break
  return chunks


def _extract_pdf(raw: bytes) -> str:
  try:
    from pypdf import PdfReader
  except ImportError as exc:
    raise HTTPException(status_code=500, detail="PDF upload requires pypdf to be installed") from exc
  reader = PdfReader(BytesIO(raw))
  return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text(filename: str, raw: bytes) -> str:
  suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"

  if suffix == "pdf":
    return _extract_pdf(raw)

  decoded = raw.decode("utf-8", errors="ignore")

  if suffix in {"txt", "md", "rst", "csv", "tsv"}:
    return decoded

  if suffix == "jsonl":
    lines: list[str] = []
    for line in decoded.splitlines():
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        lines.append(line)
        continue
      if isinstance(obj, dict):
        val = obj.get("text") or obj.get("content") or obj.get("body") or obj.get("text_en") or ""
        if val:
          lines.append(str(val))
    return "\n".join(lines)

  if suffix == "json":
    try:
      obj = json.loads(decoded)
    except json.JSONDecodeError:
      return decoded
    if isinstance(obj, dict):
      return str(obj.get("text") or obj.get("content") or obj.get("body") or decoded)
    if isinstance(obj, list):
      parts: list[str] = []
      for item in obj:
        if isinstance(item, dict):
          val = item.get("text") or item.get("content") or item.get("body") or ""
          if val:
            parts.append(str(val))
        elif isinstance(item, str):
          parts.append(item)
      return "\n".join(parts)
    return decoded

  return decoded


def _save_source_file(filename: str, raw: bytes) -> tuple[str, str]:
  SOURCE_FILE_DIR.mkdir(parents=True, exist_ok=True)
  ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  safe_name = _safe_filename(filename)
  path = SOURCE_FILE_DIR / f"{ts}_{safe_name}"
  path.write_bytes(raw)
  return str(path), sha256(raw).hexdigest()


def _append_upload_audit(record: dict) -> None:
  AUDIT_DIR.mkdir(parents=True, exist_ok=True)
  audit_path = AUDIT_DIR / "corpus_uploads.jsonl"
  with audit_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_audit_records(path: Path) -> list[dict]:
  if not path.exists():
    return []
  rows: list[dict] = []
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      raw = line.strip()
      if not raw:
        continue
      try:
        obj = json.loads(raw)
      except json.JSONDecodeError:
        continue
      if isinstance(obj, dict):
        rows.append(obj)
  return rows


def _aggregate_indexed_sources(limit: int = 300) -> list[dict]:
  corpus_rows = _iter_audit_records(AUDIT_DIR / "corpus_uploads.jsonl")
  audio_rows = _iter_audit_records(AUDIT_DIR / "audio_uploads.jsonl")

  grouped: dict[str, dict] = {}

  def _upsert(entry: dict) -> None:
    source = str(entry.get("source") or "").strip()
    source_title = str(entry.get("source_title") or source or "Untitled source").strip()
    source_type = str(entry.get("source_type") or "audio").strip()
    key = f"{source or source_title}::{source_type}"

    agg = grouped.get(key)
    if agg is None:
      agg = {
        "source": source,
        "source_title": source_title,
        "source_type": source_type,
        "language": str(entry.get("language") or entry.get("language_code") or "").strip(),
        "vector_db": str(entry.get("vector_db") or "").strip(),
        "index_name": str(entry.get("index_name") or "").strip(),
        "namespace": str(entry.get("namespace") or "").strip(),
        "uploads": 0,
        "chunks_total": 0,
        "vectors_total": 0,
        "last_uploaded_at": "",
        "last_file": "",
      }
      grouped[key] = agg

    agg["uploads"] += 1
    agg["chunks_total"] += int(entry.get("chunks_created") or 0)
    agg["vectors_total"] += int(entry.get("vectors_upserted") or 0)

    ts = str(entry.get("timestamp") or "")
    if ts and (not agg["last_uploaded_at"] or ts > agg["last_uploaded_at"]):
      agg["last_uploaded_at"] = ts
      agg["last_file"] = str(
        entry.get("original_filename")
        or entry.get("document")
        or entry.get("source_file")
        or ""
      )

  for row in corpus_rows:
    _upsert({
      **row,
      "source_type": row.get("source_type") or "text",
    })

  for row in audio_rows:
    _upsert({
      **row,
      "source_type": row.get("source_type") or "audio",
      "language": row.get("language") or row.get("language_code") or "",
      "index_name": row.get("index_name") or settings.pinecone_index_name,
    })

  items = sorted(grouped.values(), key=lambda x: x.get("last_uploaded_at") or "", reverse=True)
  return items[:limit]


@router.post("/admin/vector/upload")
async def upload_document_to_vector_db(
  file: UploadFile = File(...),
  vector_db: str = Form("pinecone"),
  index_name: str = Form(""),
  namespace: str = Form(""),
  dataset_name: str = Form(""),
  source_title: str = Form(""),
  source: str = Form(""),
  language: str = Form("en"),
  source_type: str = Form("text"),
  section: str = Form(""),
  author: str = Form(""),
  translator: str = Form(""),
  url: str = Form(""),
  _: None = Depends(require_admin_api_key),
) -> dict:
  vector_db = (vector_db or "pinecone").strip().lower()
  if vector_db != "pinecone":
    raise HTTPException(status_code=400, detail="vector_db must be 'pinecone'")

  filename = file.filename or "upload.txt"
  raw = await file.read()
  if not raw:
    raise HTTPException(status_code=400, detail="Uploaded file is empty")

  source_file_path, content_sha256 = _save_source_file(filename, raw)
  text = _extract_text(filename, raw)
  chunks = _chunk_text(text)
  if not chunks:
    raise HTTPException(status_code=400, detail="Could not extract usable text from uploaded file")

  default_index = settings.pinecone_index_name
  default_namespace = ""

  target_index = (index_name or default_index).strip()
  if not target_index:
    raise HTTPException(status_code=400, detail="index_name is required")

  target_namespace = (namespace or default_namespace).strip()

  ds_id = dataset_name.strip()
  if ds_id:
    dataset_store.register(ds_id)

  try:
    vectors, embedding_backend = await embed_texts(chunks)
  except Exception as exc:
    raise HTTPException(status_code=502, detail="Embedding provider unavailable") from exc

  records = []
  doc_id = uuid.uuid4().hex[:12]
  resolved_title = (source_title or Path(filename).stem).strip()
  resolved_source = (source or re.sub(r"[^a-z0-9]+", "_", resolved_title.lower()).strip("_") or doc_id).strip()
  now = datetime.now(timezone.utc).isoformat()
  for i, (chunk, vec) in enumerate(zip(chunks, vectors), start=1):
    chunk_id = f"admin-doc-{doc_id}-{i}"
    meta = {
      "source_type": (source_type or "text").strip(),
      "citation": resolved_title,
      "source": resolved_source,
      "source_title": resolved_title,
      "language": (language or "en").strip(),
      "section": (section or "").strip(),
      "author": (author or "").strip(),
      "translator": (translator or "").strip(),
      "url": (url or "").strip(),
      "text_preview": chunk[:500],
    }
    if ds_id:
      meta["dataset_id"] = ds_id
    upsert_chunk(chunk_id, text=chunk, metadata=meta)
    records.append({"id": chunk_id, "values": vec, "metadata": meta})

  batch_size = 50
  upserted = 0
  pc = get_pinecone()
  index = pc.Index(target_index)
  for i in range(0, len(records), batch_size):
    batch = records[i : i + batch_size]
    if target_namespace:
      index.upsert(vectors=batch, namespace=target_namespace)
    else:
      index.upsert(vectors=batch)
    upserted += len(batch)

  if ds_id:
    dataset_store.increment_count(ds_id, upserted)

  run_id = record_ingestion_run(
    kind=(source_type or "text").strip(),
    source=resolved_source,
    source_title=resolved_title,
    file_name=filename,
    language=(language or "en").strip(),
    dataset_id=ds_id,
    status="ok",
    chunks=len(chunks),
    vectors=upserted,
    vector_db=vector_db,
    embedding_backend=embedding_backend,
    metadata={
      "index_name": target_index,
      "namespace": target_namespace,
      "source_file_path": source_file_path,
      "content_sha256": content_sha256,
    },
    finished_at=now,
  )

  _append_upload_audit({
    "timestamp": now,
    "run_id": run_id,
    "role": "indexed_source",
    "file_path": source_file_path,
    "original_filename": filename,
    "bytes": len(raw),
    "sha256": content_sha256,
    "source": resolved_source,
    "source_title": resolved_title,
    "language": (language or "en").strip(),
    "source_type": (source_type or "text").strip(),
    "section": (section or "").strip(),
    "vector_db": vector_db,
    "index_name": target_index,
    "namespace": target_namespace,
    "chunks_created": len(chunks),
    "vectors_upserted": upserted,
    "embedding_backend": embedding_backend,
  })

  return {
    "status": "ok",
    "vector_db": vector_db,
    "index_name": target_index,
    "namespace": target_namespace,
    "dataset_id": ds_id or None,
    "document": filename,
    "chunks_created": len(chunks),
    "vectors_upserted": upserted,
    "embedding_backend": embedding_backend,
    "source": resolved_source,
    "source_title": resolved_title,
    "source_file_path": source_file_path,
    "content_sha256": content_sha256,
  }


@router.post("/admin/vector/sync")
async def sync_pending_vectors(
  limit: int = Query(100, ge=1, le=1000),
  index_name: str = Query(""),
  namespace: str = Query(""),
  source: str = Query(""),
  dataset_id: str = Query(""),
  create_index: bool = Query(False),
  _: None = Depends(require_admin_api_key),
) -> dict:
  try:
    return await sync_pending_chunks_to_pinecone(
      limit=limit,
      index_name=index_name,
      namespace=namespace,
      source=source,
      dataset_id=dataset_id,
      create_index=create_index,
    )
  except Exception as exc:
    raise HTTPException(status_code=502, detail=f"Vector sync failed: {str(exc)[:300]}") from exc


@router.get("/admin/vector/status")
async def vector_status(_: None = Depends(require_admin_api_key)) -> dict:
  return {"vector_status": count_chunks_by_vector_status()}


@router.get("/admin/monitor")
async def admin_monitor(_: None = Depends(require_admin_api_key)) -> dict:
  postgres: dict = {"configured": use_postgres(), "ok": False, "tables": {}, "chunk_status": [], "sources": []}
  if use_postgres():
    try:
      with pg_connect() as conn:
        pg_ensure_schema(conn)
        for table in ("datasets", "chunk_store", "ingestion_runs", "query_runs"):
          postgres["tables"][table] = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        postgres["chunk_status"] = [
          dict(row)
          for row in conn.execute(
            """
            SELECT vector_status, COUNT(*) AS count
            FROM chunk_store
            GROUP BY vector_status
            ORDER BY vector_status
            """
          ).fetchall()
        ]
        postgres["sources"] = [
          dict(row)
          for row in conn.execute(
            """
            SELECT source, source_type, vector_status, COUNT(*) AS count
            FROM chunk_store
            GROUP BY source, source_type, vector_status
            ORDER BY count DESC, source
            LIMIT 50
            """
          ).fetchall()
        ]
        postgres["ok"] = True
    except Exception as exc:
      postgres["error"] = str(exc)[:500]

  pinecone: dict = {
    "configured": bool(settings.pinecone_api_key),
    "ok": False,
    "target_index": settings.pinecone_index_name,
    "indexes": [],
  }
  if settings.pinecone_api_key:
    try:
      pc = get_pinecone()
      indexes = []
      for item in pc.list_indexes():
        name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else str(item))
        index_info = {"name": name}
        if name == settings.pinecone_index_name:
          try:
            stats = pc.Index(name).describe_index_stats()
            index_info["stats"] = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
          except Exception as exc:
            index_info["stats_error"] = str(exc)[:500]
        indexes.append(index_info)
      pinecone["indexes"] = indexes
      pinecone["ok"] = True
    except Exception as exc:
      pinecone["error"] = str(exc)[:500]

  return {"postgres": postgres, "pinecone": pinecone}


@router.get("/admin/chunks")
async def list_admin_chunks(
  limit: int = Query(25, ge=1, le=200),
  source: str = Query(""),
  vector_status: str = Query(""),
  _: None = Depends(require_admin_api_key),
) -> dict:
  if not use_postgres():
    return {"chunks": []}

  filters: list[str] = []
  params: list[object] = []
  if source.strip():
    filters.append("source = %s")
    params.append(source.strip())
  if vector_status.strip():
    filters.append("vector_status = %s")
    params.append(vector_status.strip())
  where = "WHERE " + " AND ".join(filters) if filters else ""
  params.append(int(limit))
  with pg_connect() as conn:
    pg_ensure_schema(conn)
    rows = conn.execute(
      f"""
      SELECT
        id, source, source_title, source_type, language, vector_status,
        preview, translated_preview, created_at
      FROM chunk_store
      {where}
      ORDER BY created_at DESC
      LIMIT %s
      """,
      params,
    ).fetchall()
  return {"chunks": [dict(row) for row in rows]}


@router.get("/admin/notifications")
async def list_admin_notifications(
  limit: int = Query(50, ge=1, le=200),
  _: None = Depends(require_admin_api_key),
) -> dict:
  return {"notifications": dataset_store.list_notifications(limit=limit)}


@router.delete("/admin/notifications")
async def clear_admin_notifications(_: None = Depends(require_admin_api_key)) -> dict:
  return {"deleted": dataset_store.clear_notifications()}


@router.get("/admin/audio/jobs")
async def list_audio_jobs(
  limit: int = Query(10, ge=1, le=100),
  _: None = Depends(require_admin_api_key),
) -> dict:
  if not use_postgres():
    return {"jobs": []}
  with pg_connect() as conn:
    pg_ensure_schema(conn)
    rows = conn.execute(
      """
      SELECT id, source, source_title, file_name, language, dataset_id, status,
             chunks, vectors, transcription_mode, transcription_version, error,
             metadata_json, started_at, finished_at
      FROM ingestion_runs
      WHERE kind = 'audio'
      ORDER BY finished_at DESC
      LIMIT %s
      """,
      (limit,),
    ).fetchall()
  jobs = []
  for row in rows:
    item = dict(row)
    metadata = item.pop("metadata_json") or {}
    try:
      item["metadata"] = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
    except Exception:
      item["metadata"] = {}
    jobs.append(item)
  return {"jobs": jobs}


def _feedback_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DharmaGPT Gold Store Dashboard</title>
  <style>
    :root {
      --bg: #0b1220; --panel: rgba(17,24,39,0.88); --panel-2: rgba(15,23,42,0.92);
      --text: #e5eefb; --muted: #96a7c3; --line: rgba(148,163,184,0.22);
      --accent: #f59e0b; --shadow: 0 24px 80px rgba(0,0,0,0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(245,158,11,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(59,130,246,0.18), transparent 22%),
        linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
      min-height: 100vh;
    }
    .shell { max-width: 960px; margin: 0 auto; padding: 32px 24px; }
    .topbar {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 28px; gap: 16px; flex-wrap: wrap;
    }
    .topbar h1 { margin: 0; font-size: 26px; }
    .controls {
      display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap;
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 20px; padding: 18px 20px; margin-bottom: 24px;
      box-shadow: var(--shadow);
    }
    .field { display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 200px; }
    label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    input {
      border-radius: 12px; border: 1px solid var(--line);
      background: var(--panel-2); color: var(--text);
      padding: 10px 14px; outline: none; font-size: 14px;
    }
    input:focus { border-color: rgba(245,158,11,0.6); }
    button {
      border: 0; border-radius: 12px; padding: 10px 16px;
      font-weight: 600; cursor: pointer; font-size: 14px;
      transition: transform 100ms ease, opacity 100ms ease;
    }
    button:hover { transform: translateY(-1px); }
    .btn-primary { background: linear-gradient(135deg,#f59e0b,#f97316); color: #111827; }
    .btn-approve { background: rgba(34,197,94,0.18); color: #bbf7d0; border: 1px solid rgba(34,197,94,0.35); }
    .btn-reject  { background: rgba(239,68,68,0.18);  color: #fecaca; border: 1px solid rgba(239,68,68,0.35); }
    .notice { font-size: 13px; color: #dbeafe; min-height: 1.2em; margin-top: 4px; }
    .stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 24px; }
    .stat {
      background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
      padding: 16px; box-shadow: var(--shadow);
    }
    .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; }
    .stat-value { font-size: 26px; font-weight: 700; margin-top: 6px; }
    .card {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 20px; padding: 20px; margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    .card-head {
      display: flex; justify-content: space-between; align-items: flex-start;
      gap: 12px; margin-bottom: 14px; flex-wrap: wrap;
    }
    .query-text { font-size: 16px; font-weight: 600; flex: 1; }
    .chip {
      font-size: 11px; padding: 4px 10px; border-radius: 999px;
      border: 1px solid var(--line); color: var(--muted); white-space: nowrap;
    }
    .chip.up { border-color: rgba(34,197,94,0.4); color: #86efac; }
    .block-label {
      font-size: 11px; color: var(--muted); text-transform: uppercase;
      letter-spacing: 0.07em; margin-bottom: 6px;
    }
    .answer-text {
      white-space: pre-wrap; line-height: 1.65; font-size: 14px;
      background: rgba(2,6,23,0.45); border-radius: 14px;
      border: 1px solid var(--line); padding: 14px; margin-bottom: 14px;
    }
    textarea.answer-edit {
      width: 100%; min-height: 130px; resize: vertical;
      white-space: pre-wrap; line-height: 1.65; font-size: 14px;
      background: rgba(2,6,23,0.45); border-radius: 14px;
      border: 1px solid var(--line); color: var(--text);
      padding: 14px; margin-bottom: 14px;
    }
    .note-text { font-size: 13px; color: #fde68a; font-style: italic; margin-bottom: 12px; }
    .sources { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
    .source-chip {
      font-size: 11px; padding: 4px 8px; border-radius: 8px;
      background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25); color: #93c5fd;
    }
    .card-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .ts { font-size: 12px; color: var(--muted); margin-left: auto; }
    .empty {
      color: var(--muted); padding: 40px; text-align: center;
      border: 1px dashed var(--line); border-radius: 18px;
      background: rgba(15,23,42,0.55);
    }
    @media (max-width: 700px) {
      .stats { grid-template-columns: 1fr 1fr; }
      .card-head { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <h1>Gold Store Dashboard</h1>
      <a class="nav-link" href="/docs" style="color: var(--muted); font-size: 13px; text-decoration: none; padding: 6px 14px; border: 1px solid var(--line); border-radius: 999px;">API Docs</a>
    </div>

    <div class="controls" style="align-items: center;">
      <div class="field" style="min-width: 180px;">
        <label for="vectorDb">Vector DB</label>
        <select id="vectorDb" style="border-radius: 12px; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); padding: 10px 14px;">
          <option value="pinecone" selected>Pinecone</option>
        </select>
      </div>
      <div class="field" style="min-width: 220px;">
        <label for="indexName">Index Name</label>
        <input id="indexName" type="text" placeholder="dharma-gpt" />
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="namespace">Namespace</label>
        <input id="namespace" type="text" placeholder="optional" />
      </div>
      <div class="field" style="min-width: 220px;">
        <label for="sourceTitle">Source Title</label>
        <input id="sourceTitle" type="text" placeholder="Valmiki Ramayana" />
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="sourceId">Source ID</label>
        <input id="sourceId" type="text" placeholder="valmiki_ramayana" />
      </div>
      <div class="field" style="min-width: 160px;">
        <label for="language">Language</label>
        <input id="language" type="text" placeholder="en, sa, te, hi" />
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="sourceType">Source Type</label>
        <select id="sourceType" style="border-radius: 12px; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); padding: 10px 14px;">
          <option value="text" selected>Text</option>
          <option value="commentary">Commentary</option>
          <option value="translation">Translation</option>
          <option value="audio_transcript">Audio Transcript</option>
        </select>
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="section">Section</label>
        <input id="section" type="text" placeholder="Bala Kanda" />
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="author">Author</label>
        <input id="author" type="text" placeholder="Valmiki" />
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="translator">Translator</label>
        <input id="translator" type="text" placeholder="optional" />
      </div>
      <div class="field" style="min-width: 220px;">
        <label for="sourceUrl">Source URL</label>
        <input id="sourceUrl" type="text" placeholder="optional" />
      </div>
      <div class="field" style="min-width: 240px;">
        <label for="docFile">Document File</label>
        <input id="docFile" type="file" />
      </div>
      <div class="field" style="min-width: 220px;">
        <label for="adminKey">Admin Key</label>
        <input id="adminKey" type="password" placeholder="X-Admin-Key" />
      </div>
      <button class="btn-primary" id="uploadBtn" style="margin-top: 18px;">Upload & Index</button>
    </div>

    <div class="controls">
      <button class="btn-primary" id="loadBtn">Load</button>
      <div class="notice" id="notice"></div>
    </div>

    <div class="stats">
      <div class="stat"><div class="stat-label">Pending review</div><div class="stat-value" id="pendingCount">-</div></div>
      <div class="stat"><div class="stat-label">Gold entries</div><div class="stat-value" id="goldCount">-</div></div>
      <div class="stat"><div class="stat-label">Showing</div><div class="stat-value" id="showingCount">-</div></div>
    </div>

    <div class="tab-bar" style="display:flex; gap:4px; margin-bottom:20px;">
      <button class="tab active" id="tabPending" onclick="switchTab('pending')" style="padding:8px 18px;border-radius:999px;border:1px solid var(--line);background:rgba(245,158,11,0.18);color:#ffd28b;">Pending</button>
      <button class="tab" id="tabGold" onclick="switchTab('gold')" style="padding:8px 18px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--muted);">Gold</button>
    </div>

    <div id="cards"></div>
  </div>

  <script>
    const API = "/api/v1";
    let currentTab = "pending";
    let pendingData = [];
    let goldData = [];

    const savedAdminKey = localStorage.getItem("dharmagpt.adminKey") || "";
    document.getElementById("adminKey").value = savedAdminKey;

    function adminKey() {
      const key = document.getElementById("adminKey").value.trim();
      if (key) localStorage.setItem("dharmagpt.adminKey", key);
      return key;
    }
    function headers() {
      const key = adminKey();
      const h = { "Content-Type": "application/json" };
      if (key) h["X-Admin-Key"] = key;
      return h;
    }
    function setNotice(msg, err=false) {
      const n = document.getElementById("notice");
      n.style.color = err ? "#fecaca" : "#dbeafe";
      n.textContent = msg;
    }
    function switchTab(tab) {
      currentTab = tab;
      document.getElementById("tabPending").style.background = tab === "pending" ? "rgba(245,158,11,0.18)" : "transparent";
      document.getElementById("tabPending").style.color = tab === "pending" ? "#ffd28b" : "var(--muted)";
      document.getElementById("tabGold").style.background = tab === "gold" ? "rgba(245,158,11,0.18)" : "transparent";
      document.getElementById("tabGold").style.color = tab === "gold" ? "#ffd28b" : "var(--muted)";
      render();
    }
    function fmtDate(iso) {
      try { return new Date(iso).toLocaleString(); } catch(e) { return iso || ""; }
    }
    function sourceChips(sources) {
      if (!sources || !sources.length) return "";
      return sources.slice(0,4).map(s => {
        const label = s.citation || (s.section || "") + (s.chapter ? " Ch." + s.chapter : "");
        return label ? '<span class="source-chip">' + label + '</span>' : "";
      }).join("");
    }
    function pendingCard(r) {
      const d = document.createElement("div");
      d.className = "card";
      d.id = "card-" + r.query_id;
      d.innerHTML =
        '<div class="card-head">' +
          '<div class="query-text">' + escHtml(r.query) + '</div>' +
          '<span class="chip up">&#128077; upvoted</span>' +
          '<span class="chip">' + (r.mode || "") + '</span>' +
        '</div>' +
        (r.note ? '<div class="note-text">Note: ' + escHtml(r.note) + '</div>' : '') +
        '<div class="block-label">Answer (editable before approval)</div>' +
        '<textarea class="answer-edit" id="answer-' + escHtml(r.query_id) + '">' + escHtml(r.answer) + '</textarea>' +
        '<div class="sources">' + sourceChips(r.sources) + '</div>' +
        '<div class="card-actions">' +
          '<button class="btn-approve" onclick="review(' + JSON.stringify(r.query_id) + ', \\'approved\\')">Approve &rarr; Gold</button>' +
          '<button class="btn-reject"  onclick="review(' + JSON.stringify(r.query_id) + ', \\'rejected\\')">Reject</button>' +
          '<span class="ts">' + fmtDate(r.timestamp) + '</span>' +
        '</div>';
      return d;
    }
    function goldCard(r) {
      const d = document.createElement("div");
      d.className = "card";
      d.innerHTML =
        '<div class="card-head">' +
          '<div class="query-text">' + escHtml(r.query) + '</div>' +
          '<span class="chip" style="border-color:rgba(245,158,11,0.4);color:#fde68a;">&#10003; gold</span>' +
          '<span class="chip">' + (r.mode || "") + '</span>' +
        '</div>' +
        '<div class="block-label">Gold answer</div>' +
        '<div class="answer-text">' + escHtml(r.gold_answer) + '</div>' +
        '<div class="sources">' + sourceChips(r.evidence || []) + '</div>' +
        '<div class="card-actions"><span class="ts">Promoted ' + fmtDate(r.promoted_at) + '</span></div>';
      return d;
    }
    function escHtml(s) {
      return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
    }
    function render() {
      const cards = document.getElementById("cards");
      cards.innerHTML = "";
      const data = currentTab === "pending" ? pendingData : goldData;
      document.getElementById("showingCount").textContent = String(data.length);
      if (!data.length) {
        cards.innerHTML = '<div class="empty">' + (currentTab === "pending" ? "No pending responses." : "No gold responses yet.") + '</div>';
        return;
      }
      for (const r of data) {
        cards.appendChild(currentTab === "pending" ? pendingCard(r) : goldCard(r));
      }
    }
    async function load() {
      setNotice("Loading...");
      try {
        const [pResp, gResp] = await Promise.all([
          fetch(API + "/feedback/pending", { headers: headers() }),
          fetch(API + "/feedback/gold",    { headers: headers() }),
        ]);
        if (!pResp.ok) throw new Error("Pending: " + pResp.statusText);
        if (!gResp.ok) throw new Error("Gold: " + gResp.statusText);
        const p = await pResp.json();
        const g = await gResp.json();
        pendingData = p.pending || [];
        goldData    = g.gold    || [];
        document.getElementById("pendingCount").textContent = String(pendingData.length);
        document.getElementById("goldCount").textContent    = String(goldData.length);
        setNotice("Loaded.");
        render();
      } catch(e) { setNotice(e.message, true); }
    }
    async function review(queryId, status) {
      setNotice("Saving...");
      try {
        const answerEl = document.getElementById("answer-" + queryId);
        const editedAnswer = answerEl ? answerEl.value.trim() : "";
        const resp = await fetch(API + "/feedback/" + encodeURIComponent(queryId), {
          method: "PATCH",
          headers: headers(),
          body: JSON.stringify({
            review_status: status,
            gold_answer: status === "approved" ? editedAnswer : null,
          }),
        });
        if (!resp.ok) {
          let msg = resp.statusText;
          try {
            const body = await resp.json();
            if (body && body.detail) msg = body.detail;
          } catch (_) {}
          throw new Error(msg);
        }
        pendingData = pendingData.filter(r => r.query_id !== queryId);
        document.getElementById("pendingCount").textContent = String(pendingData.length);
        const card = document.getElementById("card-" + queryId);
        if (card) card.remove();
        if (status === "approved") {
          const gResp = await fetch(API + "/feedback/gold", { headers: headers() });
          if (!gResp.ok) throw new Error("Gold: " + gResp.statusText);
          const g = await gResp.json();
          goldData = g.gold || [];
          document.getElementById("goldCount").textContent = String(goldData.length);
          setNotice("Approved and added to gold set.");
        } else {
          setNotice("Rejected.");
        }
        render();
      } catch(e) { setNotice(e.message, true); }
    }

    async function uploadDocument() {
      const fileInput = document.getElementById("docFile");
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setNotice("Please choose a document file first.", true);
        return;
      }

      const fd = new FormData();
      fd.append("file", file);
      fd.append("vector_db", document.getElementById("vectorDb").value || "pinecone");
      fd.append("index_name", document.getElementById("indexName").value || "");
      fd.append("namespace", document.getElementById("namespace").value || "");
      fd.append("source_title", document.getElementById("sourceTitle").value || "");
      fd.append("source", document.getElementById("sourceId").value || "");
      fd.append("language", document.getElementById("language").value || "en");
      fd.append("source_type", document.getElementById("sourceType").value || "text");
      fd.append("section", document.getElementById("section").value || "");
      fd.append("author", document.getElementById("author").value || "");
      fd.append("translator", document.getElementById("translator").value || "");
      fd.append("url", document.getElementById("sourceUrl").value || "");
      const key = adminKey();
      if (!key) {
        setNotice("Enter the admin key before uploading.", true);
        return;
      }

      setNotice("Uploading and indexing document...");
      try {
        const resp = await fetch("/admin/vector/upload", {
          method: "POST",
          headers: { "X-Admin-Key": key },
          body: fd,
        });
        if (!resp.ok) {
          let msg = resp.statusText;
          try {
            const body = await resp.json();
            if (body && body.detail) msg = body.detail;
          } catch (_) {}
          throw new Error(msg);
        }
        const data = await resp.json();
        setNotice(
          "Indexed " + data.vectors_upserted + " chunks from " + data.source_title + " into " + data.vector_db + ":" + data.index_name +
          (data.namespace ? (" (namespace: " + data.namespace + ")") : "") + "."
        );
      } catch (e) {
        setNotice("Upload failed: " + e.message, true);
      }
    }

    document.getElementById("loadBtn").addEventListener("click", load);
    document.getElementById("uploadBtn").addEventListener("click", uploadDocument);
    load();
  </script>
</body>
</html>"""


@router.get("/admin/feedback", response_class=HTMLResponse)
async def feedback_admin() -> HTMLResponse:
    return HTMLResponse(_feedback_page())


# ── Dataset management endpoints ──────────────────────────────────────────────

@router.get("/admin/datasets")
async def list_datasets(_: None = Depends(require_admin_api_key)) -> dict:
    return {"datasets": dataset_store.list_all()}


class DatasetToggle(BaseModel):
    active: bool


class TranslationCreate(BaseModel):
    source: str
    source_title: str = ""
    chunk_index: int | None = None
    vector_chunk_id: str | None = None
    original_text: str
    original_language: str = "te"
    translated_text: str
    translated_language: str = "en"
    translator_name: str = ""
    section: str | None = None
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    notes: str = ""
    verified: bool = False


class TranslationUpdate(BaseModel):
    translated_text: str | None = None
    verified: bool | None = None
    notes: str | None = None
    translator_name: str | None = None


@router.patch("/admin/datasets/{name}")
async def toggle_dataset(name: str, body: DatasetToggle, _: None = Depends(require_admin_api_key)) -> dict:
    ok = dataset_store.set_active(name, body.active)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return {"name": name, "active": body.active}


@router.delete("/admin/datasets/{name}")
async def delete_dataset(name: str, purge_vectors: bool = False, _: None = Depends(require_admin_api_key)) -> dict:
    if purge_vectors and settings.vector_db_backend.lower() == "pinecone":
        try:
            pc = get_pinecone()
            index = pc.Index(settings.pinecone_index_name)
            index.delete(filter={"dataset_id": {"$eq": name}})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pinecone purge failed: {exc}")
    dataset_store.remove(name)
    return {"deleted": name, "vectors_purged": purge_vectors}


@router.get("/admin/sources")
async def list_indexed_sources(limit: int = 300, _: None = Depends(require_admin_api_key)) -> dict:
    safe_limit = max(1, min(limit, 1000))
    return {"sources": _aggregate_indexed_sources(limit=safe_limit)}


@router.get("/admin/usage-stats")
async def usage_stats(limit: int = 50, _: None = Depends(require_admin_api_key)) -> dict:
    safe_limit = max(1, min(limit, 500))
    return summarize_usage(limit=safe_limit)


# ── discourse_translations CRUD ───────────────────────────────────────────────

@router.get("/admin/translations")
async def list_translations(
    source: str | None = None,
    verified: bool | None = None,
    limit: int = 100,
    _: None = Depends(require_admin_api_key),
) -> dict:
    from core.postgres_db import list_discourse_translations, use_postgres
    if not use_postgres():
        raise HTTPException(status_code=503, detail="Postgres not configured")
    items = list_discourse_translations(source=source, verified=verified, limit=limit)
    return {"translations": items, "count": len(items)}


@router.post("/admin/translations", status_code=201)
async def create_translation(
    body: TranslationCreate,
    _: None = Depends(require_admin_api_key),
) -> dict:
    from core.postgres_db import create_discourse_translation, use_postgres
    if not use_postgres():
        raise HTTPException(status_code=503, detail="Postgres not configured")
    result = create_discourse_translation(**body.model_dump())
    return {"status": "created", **result}


@router.patch("/admin/translations/{translation_id}")
async def update_translation(
    translation_id: str,
    body: TranslationUpdate,
    _: None = Depends(require_admin_api_key),
) -> dict:
    from core.postgres_db import update_discourse_translation, use_postgres
    if not use_postgres():
        raise HTTPException(status_code=503, detail="Postgres not configured")
    ok = update_discourse_translation(translation_id, **body.model_dump(exclude_none=True))
    if not ok:
        raise HTTPException(status_code=404, detail="Translation not found")
    return {"status": "updated", "id": translation_id}


@router.delete("/admin/translations/{translation_id}")
async def delete_translation(
    translation_id: str,
    _: None = Depends(require_admin_api_key),
) -> dict:
    from core.postgres_db import delete_discourse_translation, use_postgres
    if not use_postgres():
        raise HTTPException(status_code=503, detail="Postgres not configured")
    ok = delete_discourse_translation(translation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Translation not found")
    return {"status": "deleted", "id": translation_id}


# ── Main admin dashboard ───────────────────────────────────────────────────────

def _admin_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>DharmaGPT Admin</title>
  <style>
    :root{--bg:#0b1220;--panel:rgba(17,24,39,.88);--panel2:rgba(15,23,42,.92);--text:#e5eefb;--muted:#96a7c3;--line:rgba(148,163,184,.22);--accent:#f59e0b;--shadow:0 24px 80px rgba(0,0,0,.35);--green:rgba(34,197,94,.18);--greenb:rgba(34,197,94,.35);--greentext:#bbf7d0;--red:rgba(239,68,68,.18);--redb:rgba(239,68,68,.35);--redtext:#fecaca}
    *{box-sizing:border-box;min-width:0}
    html{min-height:100%;background:#0b1220;overflow-x:hidden}
    body{margin:0;font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:var(--text);background:radial-gradient(circle at top left,rgba(245,158,11,.18),transparent 28%),radial-gradient(circle at top right,rgba(59,130,246,.18),transparent 22%),linear-gradient(180deg,#0b1220 0%,#0f172a 100%);min-height:100vh;min-height:100dvh;overflow-x:hidden}
    .shell{width:100%;max-width:min(980px,100vw);margin:0 auto;padding:28px 20px;overflow-x:hidden}
    h1{margin:0 0 24px;font-size:24px}
    .tabs{display:flex;gap:4px;margin-bottom:24px;border-bottom:1px solid var(--line);padding-bottom:0;overflow-x:auto;scrollbar-width:thin}
    .tab{flex:0 0 auto;padding:10px 20px;border-radius:10px 10px 0 0;border:1px solid transparent;background:transparent;color:var(--muted);cursor:pointer;font-size:14px;font-weight:500;border-bottom:none;transition:color .15s}
    .tab.active{background:var(--panel);border-color:var(--line);border-bottom-color:var(--bg);color:var(--text)}
    .pane{display:none}.pane.active{display:block}
    .card{width:100%;max-width:100%;overflow:hidden;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:16px;box-shadow:var(--shadow)}
    label{display:block;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
    input,select,textarea{width:100%;max-width:100%;border-radius:10px;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:9px 13px;font-size:14px;outline:none;font-family:inherit}
    input:focus,select:focus,textarea:focus{border-color:rgba(245,158,11,.6)}
    .row{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin-bottom:14px}
    .row.two{grid-template-columns:1fr 1fr}
    .row.three{grid-template-columns:1fr 1fr 1fr}
    button{border:0;border-radius:10px;padding:9px 16px;font-weight:600;cursor:pointer;font-size:14px;transition:transform .1s,opacity .1s}
    button:hover{transform:translateY(-1px)}
    .btn-primary{background:linear-gradient(135deg,#f59e0b,#f97316);color:#111827}
    .btn-sm{padding:5px 11px;font-size:12px}
    .btn-green{background:var(--green);color:var(--greentext);border:1px solid var(--greenb)}
    .btn-red{background:var(--red);color:var(--redtext);border:1px solid var(--redb)}
    .btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
    .notice{font-size:13px;color:#dbeafe;min-height:1.3em;margin-top:8px}
    .notice.err{color:#fecaca}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th{text-align:left;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;padding:8px 10px;border-bottom:1px solid var(--line)}
    td{padding:10px 10px;border-bottom:1px solid rgba(148,163,184,.1);vertical-align:middle}
    tr:last-child td{border-bottom:0}
    .badge{display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:3px 10px;border-radius:999px}
    .badge.on{background:var(--green);color:var(--greentext);border:1px solid var(--greenb)}
    .badge.off{background:var(--red);color:var(--redtext);border:1px solid var(--redb)}
    .empty{color:var(--muted);padding:30px;text-align:center;border:1px dashed var(--line);border-radius:14px;overflow-wrap:anywhere}
    .answer-box{white-space:pre-wrap;line-height:1.65;font-size:14px;background:rgba(2,6,23,.45);border-radius:12px;border:1px solid var(--line);padding:14px;margin:12px 0}
    .source-chip{font-size:11px;padding:4px 8px;border-radius:8px;background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);color:#93c5fd;display:inline-block;margin:3px 3px 3px 0}
    .score{font-size:11px;color:var(--muted)}
    .section-title{font-size:16px;font-weight:600;margin-bottom:14px}
    .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
    .metric{background:rgba(2,6,23,.35);border:1px solid var(--line);border-radius:12px;padding:14px}
    .metric-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
	    .metric-value{font-size:24px;font-weight:700;margin-top:4px}
	    .mini-list{display:grid;gap:8px}
	    .mini-item{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid rgba(148,163,184,.12);padding-bottom:7px;font-size:13px}
	    .progress-list{display:grid;gap:8px;margin-top:12px}
	    .progress-step{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted)}
	    .progress-dot{width:10px;height:10px;border-radius:50%;background:rgba(148,163,184,.35);flex:0 0 auto}
	    .progress-step.active{color:#fde68a}.progress-step.active .progress-dot{background:#f59e0b;box-shadow:0 0 0 4px rgba(245,158,11,.16)}
	    .progress-step.done{color:var(--greentext)}.progress-step.done .progress-dot{background:#22c55e}
	    .progress-step.err{color:var(--redtext)}.progress-step.err .progress-dot{background:#ef4444}
	    .divider{border:0;border-top:1px solid var(--line);margin:20px 0}
    @media(max-width:520px){
      .shell{padding:24px 14px}
      h1{font-size:26px;line-height:1.15}
      .card{padding:18px}
      .row,.row.two,.row.three{grid-template-columns:1fr}
      .tabs{display:flex;flex-direction:column;overflow:visible;border-bottom:0;gap:6px}
      .tab{width:100%;padding:10px 12px;border-radius:10px;border:1px solid transparent;text-align:center}
      .tab.active{border-color:var(--line)}
      .empty{padding:28px 16px}
    }
  </style>
</head>
<body>
<div class="shell">
  <h1>DharmaGPT Admin</h1>
  <div class="card" style="padding:14px;margin-bottom:18px">
    <label>Admin/API Key</label>
    <input id="admin-key" type="password" placeholder="X-Admin-Key" autocomplete="off"/>
  </div>
  <div class="tabs">
    <button class="tab active" onclick="showTab('datasets')">Datasets</button>
    <button class="tab" onclick="showTab('sources')">Sources</button>
    <button class="tab" onclick="showTab('stats')">Stats</button>
    <button class="tab" onclick="showTab('monitor')">Monitor</button>
    <button class="tab" onclick="showTab('upload')">Upload</button>
    <button class="tab" onclick="showTab('test')">Test Query</button>
    <button class="tab" onclick="showTab('gold')">Gold Store</button>
    <button class="tab" onclick="showTab('translations')">Translations</button>
  </div>

  <!-- ── DATASETS ── -->
  <div class="pane active" id="pane-datasets">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div class="section-title" style="margin:0">Active Datasets</div>
        <button class="btn-ghost btn-sm" onclick="loadDatasets()">Refresh</button>
      </div>
      <div id="datasets-table"><div class="empty">Loading…</div></div>
      <div class="notice" id="ds-notice"></div>
    </div>
    <div class="notice" style="font-size:12px;color:var(--muted)">Disabled datasets are excluded from all queries. Deleting with "Purge vectors" removes them from Pinecone permanently.</div>
  </div>

  <!-- ── SOURCES ── -->
  <div class="pane" id="pane-sources">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div class="section-title" style="margin:0">Indexed Sources</div>
        <button class="btn-ghost btn-sm" onclick="loadSources()">Refresh</button>
      </div>
      <div id="sources-table"><div class="empty">Loading…</div></div>
      <div class="notice" id="src-notice"></div>
    </div>
    <div class="notice" style="font-size:12px;color:var(--muted)">Shows source titles from document/audio indexing audit logs.</div>
  </div>

  <!-- ── STATS ── -->
  <div class="pane" id="pane-stats">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div class="section-title" style="margin:0">Ingestion Insights</div>
        <button class="btn-ghost btn-sm" onclick="loadStats()">Refresh</button>
      </div>
      <div id="stats-summary"><div class="empty">Loading…</div></div>
      <div class="notice" id="stats-notice"></div>
    </div>
    <div class="card">
      <div class="section-title">Latest Runs</div>
      <div id="stats-runs"><div class="empty">No runs loaded.</div></div>
    </div>
	  </div>

	  <!-- ── MONITOR ── -->
	  <div class="pane" id="pane-monitor">
	    <div class="card">
	      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;gap:12px;flex-wrap:wrap">
	        <div class="section-title" style="margin:0">Postgres Checkpoint</div>
	        <button class="btn-ghost btn-sm" onclick="loadMonitor()">Refresh</button>
	      </div>
	      <div id="monitor-postgres"><div class="empty">Loading…</div></div>
	      <div class="notice" id="monitor-notice"></div>
	    </div>
	    <div class="card">
	      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;gap:12px;flex-wrap:wrap">
	        <div class="section-title" style="margin:0">Audio Job Progress</div>
	        <button class="btn-ghost btn-sm" onclick="loadAudioJobs()">Refresh Jobs</button>
	      </div>
	      <div id="monitor-audio-jobs"><div class="empty">Loading…</div></div>
	      <div class="notice" id="audio-jobs-notice"></div>
	    </div>
	    <div class="card">
	      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;gap:12px;flex-wrap:wrap">
	        <div class="section-title" style="margin:0">Recent Chunk Rows</div>
	        <button class="btn-ghost btn-sm" onclick="loadChunks()">Refresh Rows</button>
	      </div>
	      <div class="row three">
	        <div><label>Source Filter</label><input id="chunk-source-filter" type="text" placeholder="optional source id"/></div>
	        <div><label>Status Filter</label>
	          <select id="chunk-status-filter">
	            <option value="">Any</option>
	            <option value="pending">Pending</option>
	            <option value="indexed">Indexed</option>
	            <option value="error">Error</option>
	          </select>
	        </div>
	        <div><label>Limit</label><input id="chunk-limit" type="number" min="1" max="200" value="25"/></div>
	      </div>
	      <div id="monitor-chunks"><div class="empty">Loading…</div></div>
	      <div class="notice" id="chunks-notice"></div>
	    </div>
	    <div class="card">
	      <div class="section-title">Pinecone Sync</div>
	      <div class="row three">
	        <div><label>Batch Limit</label><input id="sync-limit" type="number" min="1" max="1000" value="100"/></div>
	        <div><label>Index Name</label><input id="sync-index" type="text" placeholder="default from .env"/></div>
	        <div><label>Namespace</label><input id="sync-namespace" type="text" placeholder="optional"/></div>
	      </div>
	      <div style="display:flex;gap:8px;flex-wrap:wrap">
	        <button class="btn-primary" onclick="syncVectors(false)">Sync Pending</button>
	        <button class="btn-ghost" onclick="syncVectors(true)">Sync + Create Index</button>
	      </div>
	      <div class="notice" id="sync-notice"></div>
	      <div id="monitor-pinecone" style="margin-top:16px"><div class="empty">Loading…</div></div>
	    </div>
	    <div class="card">
	      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;gap:12px;flex-wrap:wrap">
	        <div class="section-title" style="margin:0">API Failure Notifications</div>
	        <div style="display:flex;gap:8px;flex-wrap:wrap">
	          <button class="btn-ghost btn-sm" onclick="loadNotifications()">Refresh</button>
	          <button class="btn-red btn-sm" onclick="clearNotifications()">Clear</button>
	        </div>
	      </div>
	      <div id="monitor-notifications"><div class="empty">Loading…</div></div>
	      <div class="notice" id="notifications-notice"></div>
	    </div>
	  </div>

  <!-- ── UPLOAD ── -->
  <div class="pane" id="pane-upload">
    <div class="card">
      <div class="section-title">Audio Upload</div>
      <div class="row three">
        <div><label>Audio File (mp3/wav/m4a)</label><input type="file" id="au-file" accept=".mp3,.wav,.m4a,.aac,.ogg,.flac,.opus"/></div>
        <div><label>Language</label>
          <select id="au-lang">
            <option value="te-IN">Telugu (te-IN)</option>
            <option value="hi-IN">Hindi (hi-IN)</option>
            <option value="sa-IN">Sanskrit (sa-IN)</option>
            <option value="en-IN">English (en-IN)</option>
          </select>
        </div>
        <div><label>Dataset Name</label><input id="au-dataset" type="text" placeholder="e.g. ramayanam-chaganti"/></div>
      </div>
      <div class="row two">
        <div><label>Section (optional)</label><input id="au-section" type="text" placeholder="e.g. Bala Kanda"/></div>
        <div><label>Description (optional)</label><input id="au-desc" type="text" placeholder="e.g. Part 1 clip 42"/></div>
      </div>
<<<<<<< HEAD
      <button class="btn-primary" id="au-upload-btn" onclick="uploadAudio()">Transcribe & Stage</button>
      <div class="notice" id="au-notice"></div>
      <div class="progress-list" id="au-progress" style="display:none"></div>
    </div>
=======
	      <button class="btn-primary" id="au-upload-btn" onclick="uploadAudio()">Transcribe & Stage</button>
	      <div class="notice" id="au-notice"></div>
	      <div class="progress-list" id="au-progress" style="display:none"></div>
	    </div>
>>>>>>> main

    <hr class="divider"/>

    <div class="card">
      <div class="section-title">Document Upload <span style="font-size:12px;color:var(--muted);font-weight:400">(pdf / txt / md / jsonl / json / rst / csv / tsv)</span></div>
      <div class="row three">
        <div><label>File</label><input type="file" id="doc-file" accept=".pdf,.txt,.md,.jsonl,.json,.rst,.csv,.tsv"/></div>
        <div><label>Dataset Name</label><input id="doc-dataset" type="text" placeholder="e.g. seed-corpus"/></div>
        <div><label>Vector DB</label>
          <select id="doc-db">
            <option value="pinecone" selected>Pinecone</option>
          </select>
        </div>
      </div>
      <button class="btn-primary" onclick="uploadDoc()">Chunk & Index</button>
      <div class="notice" id="doc-notice"></div>
    </div>
  </div>

  <!-- ── TEST ── -->
  <div class="pane" id="pane-test">
    <div class="card">
      <div class="row two">
        <div style="grid-column:1/-1"><label>Query</label><textarea id="q-text" rows="3" placeholder="Ask anything from the Ramayana or other texts…"></textarea></div>
      </div>
      <div class="row three">
        <div><label>Mode</label>
          <select id="q-mode">
            <option value="guidance">Guidance</option>
            <option value="story">Story</option>
            <option value="children">Children</option>
            <option value="scholar">Scholar</option>
          </select>
        </div>
        <div><label>Language</label>
          <select id="q-lang">
            <option value="en">English</option>
            <option value="te">Telugu</option>
            <option value="hi">Hindi</option>
          </select>
        </div>
        <div><label>Section filter (optional)</label><input id="q-section" type="text" placeholder="e.g. Sundara Kanda"/></div>
      </div>
      <button class="btn-primary" id="q-btn" onclick="runQuery()">Ask</button>
      <div class="notice" id="q-notice"></div>
    </div>
    <div id="q-result" style="display:none">
      <div class="card">
        <div class="section-title">Answer</div>
        <div class="answer-box" id="q-answer"></div>
        <div class="section-title" style="margin-top:12px">Sources</div>
        <div id="q-sources"></div>
      </div>
    </div>
  </div>

  <!-- ── GOLD STORE ── -->
  <div class="pane" id="pane-gold">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div class="section-title" style="margin:0">Pending Reviews</div>
        <button class="btn-ghost btn-sm" onclick="loadGold()">Refresh</button>
      </div>
      <div class="notice" id="gold-notice"></div>
      <div id="gold-cards"><div class="empty">Click Refresh to load.</div></div>
    </div>
  </div>

  <!-- ── TRANSLATIONS ── -->
  <div class="pane" id="pane-translations">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div class="section-title" style="margin:0">Discourse Translations</div>
        <button class="btn-ghost btn-sm" onclick="loadTranslations()">Refresh</button>
      </div>
      <div class="row three" style="margin-bottom:14px">
        <div><label>Filter by Source</label><input id="tl-filter-source" type="text" placeholder="e.g. ramayana_discourse_001"/></div>
        <div><label>Verified</label>
          <select id="tl-filter-verified">
            <option value="">All</option>
            <option value="true">Verified only</option>
            <option value="false">Unverified only</option>
          </select>
        </div>
        <div style="display:flex;align-items:flex-end"><button class="btn-primary btn-sm" onclick="loadTranslations()">Filter</button></div>
      </div>
      <div id="tl-table"><div class="empty">Click Refresh to load.</div></div>
      <div class="notice" id="tl-notice"></div>
    </div>

    <div class="card" id="tl-form-card">
      <div class="section-title" id="tl-form-title">Add Translation</div>
      <div class="row two">
        <div><label>Source *</label><input id="tl-source" type="text" placeholder="ramayana_discourse_001"/></div>
        <div><label>Source Title</label><input id="tl-source-title" type="text" placeholder="optional"/></div>
      </div>
      <div class="row three">
        <div><label>Chunk Index</label><input id="tl-chunk-index" type="number" placeholder="optional"/></div>
        <div><label>Vector Chunk ID</label><input id="tl-vector-id" type="text" placeholder="optional"/></div>
        <div><label>Section</label><input id="tl-section" type="text" placeholder="optional"/></div>
      </div>
      <div class="row two">
        <div><label>Original Language</label>
          <select id="tl-orig-lang">
            <option value="te">Telugu</option>
            <option value="hi">Hindi</option>
            <option value="sa">Sanskrit</option>
            <option value="en">English</option>
          </select>
        </div>
        <div><label>Translator Name</label><input id="tl-translator" type="text" placeholder="optional"/></div>
      </div>
      <div class="row two" style="margin-bottom:12px">
        <div><label>Original Text *</label><textarea id="tl-orig" rows="3" placeholder="Original text in native language..."></textarea></div>
        <div><label>English Translation *</label><textarea id="tl-trans" rows="3" placeholder="English translation..."></textarea></div>
      </div>
      <div style="margin-bottom:14px"><label>Notes</label><input id="tl-notes" type="text" placeholder="optional"/></div>
      <div style="display:flex;gap:10px;align-items:center">
        <button class="btn-primary" id="tl-save-btn" onclick="saveTl()">Add Translation</button>
        <button class="btn-ghost btn-sm" id="tl-cancel-btn" onclick="cancelTl()" style="display:none">Cancel</button>
      </div>
    </div>
  </div>
</div>

<script>
const API = "/api/v1";
const savedAdminKey = localStorage.getItem("dharmagpt.adminKey") || "";
document.getElementById("admin-key").value = savedAdminKey;

function adminKey() {
  const key = document.getElementById("admin-key").value.trim();
  if(key) localStorage.setItem("dharmagpt.adminKey", key);
  return key;
}
function adminHeaders(extra={}) {
  const key = adminKey();
  return key ? {...extra, "X-Admin-Key": key, "X-API-Key": key} : extra;
}

// ── Tab switching ──────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t,i)=>{
    const names=["datasets","sources","stats","monitor","upload","test","gold","translations"];
    t.classList.toggle("active", names[i]===name);
  });
  document.querySelectorAll(".pane").forEach(p => p.classList.remove("active"));
  document.getElementById("pane-"+name).classList.add("active");
  if(name==="datasets") loadDatasets();
<<<<<<< HEAD
  if(name==="sources") loadSources();
  if(name==="stats") loadStats();
  if(name==="monitor") { loadMonitor(); loadAudioJobs(); loadChunks(); loadNotifications(); }
  if(name==="gold") loadGold();
  if(name==="translations") loadTranslations();
}
=======
	  if(name==="sources") loadSources();
	  if(name==="stats") loadStats();
		  if(name==="monitor") { loadMonitor(); loadAudioJobs(); loadChunks(); loadNotifications(); }
	  if(name==="gold") loadGold();
	}
>>>>>>> main

function setNotice(id, msg, err=false) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = "notice" + (err?" err":"");
}

function esc(s){ return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

// ── Datasets ──────────────────────────────────────────────────────────────────
async function loadDatasets() {
  try {
    const r = await fetch("/admin/datasets", {headers: adminHeaders()});
    const d = await r.json();
    renderDatasets(d.datasets || []);
  } catch(e) { setNotice("ds-notice","Load failed: "+e.message,true); }
}

function renderDatasets(datasets) {
  const el = document.getElementById("datasets-table");
  if(!datasets.length){
    el.innerHTML='<div class="empty">No datasets yet. Upload a document or audio file to create one.</div>';
    return;
  }
  let html='<table><thead><tr><th>Name</th><th>Chunks</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead><tbody>';
  for(const ds of datasets){
    const ts = ds.created_at ? new Date(ds.created_at).toLocaleDateString() : "–";
    html+=`<tr>
      <td><strong>${esc(ds.display_name||ds.name)}</strong><br/><span style="font-size:11px;color:var(--muted)">${esc(ds.name)}</span></td>
      <td>${ds.vector_count||0}</td>
      <td><span class="badge ${ds.active?'on':'off'}">${ds.active?'● On':'○ Off'}</span></td>
      <td>${ts}</td>
      <td style="display:flex;gap:6px;flex-wrap:wrap">
        <button class="btn-sm ${ds.active?'btn-red':'btn-green'}" onclick="toggleDs('${esc(ds.name)}',${ds.active?0:1})">${ds.active?'Disable':'Enable'}</button>
        <button class="btn-sm btn-red" onclick="deleteDs('${esc(ds.name)}')">Delete</button>
      </td>
    </tr>`;
  }
  html+='</tbody></table>';
  el.innerHTML=html;
}

// ── Sources ───────────────────────────────────────────────────────────────────
async function loadSources() {
  try {
    const r = await fetch("/admin/sources?limit=300", {headers: adminHeaders()});
    const d = await r.json();
    renderSources(d.sources || []);
    setNotice("src-notice", `Loaded ${(d.sources || []).length} source(s).`);
  } catch(e) { setNotice("src-notice","Load failed: "+e.message,true); }
}

function renderSources(sources) {
  const el = document.getElementById("sources-table");
  if(!sources.length){
    el.innerHTML='<div class="empty">No indexed sources found yet.</div>';
    return;
  }
  let html='<table><thead><tr><th>Source</th><th>Type</th><th>Language</th><th>Uploads</th><th>Vectors</th><th>Last Upload</th></tr></thead><tbody>';
  for(const s of sources){
    const ts = s.last_uploaded_at ? new Date(s.last_uploaded_at).toLocaleString() : "-";
    const subtitle = s.source ? s.source : "-";
    html+=`<tr>
      <td><strong>${esc(s.source_title || "Untitled source")}</strong><br/><span style="font-size:11px;color:var(--muted)">${esc(subtitle)}</span></td>
      <td>${esc(s.source_type || "-")}</td>
      <td>${esc(s.language || "-")}</td>
      <td>${s.uploads || 0}</td>
      <td>${s.vectors_total || 0}<br/><span style="font-size:11px;color:var(--muted)">${s.chunks_total || 0} chunks</span></td>
      <td>${ts}<br/><span style="font-size:11px;color:var(--muted)">${esc(s.last_file || "")}</span></td>
    </tr>`;
  }
  html+='</tbody></table>';
  el.innerHTML=html;
}

// ── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch("/admin/usage-stats?limit=50", {headers: adminHeaders()});
    const d = await r.json();
    renderStats(d);
    setNotice("stats-notice", "Loaded ingestion insights.");
  } catch(e) { setNotice("stats-notice","Load failed: "+e.message,true); }
}

function miniList(items) {
  if(!items || !items.length) return '<div class="empty" style="padding:16px">No data yet.</div>';
  return '<div class="mini-list">' + items.map(x =>
    `<div class="mini-item"><span>${esc(x.name || x.date || "-")}</span><strong>${esc(x.count || x.vectors || 0)}</strong></div>`
  ).join("") + '</div>';
}

function renderStats(d) {
  const totals = d.totals || {};
  const usage = d.usage || {};
  document.getElementById("stats-summary").innerHTML = `
    <div class="metric-grid">
      <div class="metric"><div class="metric-label">Runs</div><div class="metric-value">${totals.runs || 0}</div></div>
      <div class="metric"><div class="metric-label">Queries</div><div class="metric-value">${totals.query_runs || 0}</div></div>
      <div class="metric"><div class="metric-label">Audio Runs</div><div class="metric-value">${totals.audio_runs || 0}</div></div>
      <div class="metric"><div class="metric-label">Chunks</div><div class="metric-value">${totals.chunks || 0}</div></div>
      <div class="metric"><div class="metric-label">Vectors</div><div class="metric-value">${totals.vectors || 0}</div></div>
    </div>
    <div class="row two">
      <div><div class="section-title">STT Models</div>${miniList(usage.transcription)}</div>
      <div><div class="section-title">Embeddings</div>${miniList(usage.embedding)}</div>
      <div><div class="section-title">Vector DB</div>${miniList(usage.vector_db)}</div>
      <div><div class="section-title">Answer Models</div>${miniList(usage.query_models)}</div>
      <div><div class="section-title">Ratings by Model</div>${miniList(usage.query_ratings)}</div>
    </div>`;

  const rows = d.latest || [];
  if(!rows.length){
    document.getElementById("stats-runs").innerHTML = '<div class="empty">No ingestion runs yet.</div>';
    return;
  }
  let html = '<table><thead><tr><th>When</th><th>Source</th><th>Status</th><th>Models</th><th>Vectors</th></tr></thead><tbody>';
  for(const r of rows){
    const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : "-";
    const models = [
      r.transcription_mode ? "STT: " + r.transcription_mode + " / " + (r.transcription_version || "-") : "",
      r.embedding_backend ? "EMB: " + r.embedding_backend : ""
    ].filter(Boolean).join("<br/>");
    html += `<tr>
      <td>${ts}</td>
      <td><strong>${esc(r.source_title || r.source || r.file || "-")}</strong><br/><span style="font-size:11px;color:var(--muted)">${esc(r.kind || "")}</span></td>
      <td><span class="badge ${r.status==="ok"?"on":"off"}">${esc(r.status || "unknown")}</span></td>
      <td style="font-size:12px">${models || "-"}</td>
      <td>${r.vectors || 0}<br/><span style="font-size:11px;color:var(--muted)">${r.chunks || 0} chunks</span></td>
    </tr>`;
  }
  html += '</tbody></table>';
  document.getElementById("stats-runs").innerHTML = html;
}

async function toggleDs(name, active) {
  try {
    const r = await fetch(`/admin/datasets/${encodeURIComponent(name)}`, {
      method:"PATCH", headers:adminHeaders({"Content-Type":"application/json"}),
      body: JSON.stringify({active: !!active})
    });
    if(!r.ok) throw new Error(await r.text());
    setNotice("ds-notice", `${name} ${active?"enabled":"disabled"}.`);
    loadDatasets();
  } catch(e){ setNotice("ds-notice","Error: "+e.message,true); }
}

async function deleteDs(name) {
  const purge = confirm(`Delete dataset "${name}"?\n\nClick OK to also purge vectors from Pinecone, or Cancel to only remove from registry.`);
  try {
    const r = await fetch(`/admin/datasets/${encodeURIComponent(name)}?purge_vectors=${purge}`, {method:"DELETE", headers:adminHeaders()});
    if(!r.ok) throw new Error(await r.text());
    setNotice("ds-notice", `Deleted ${name}${purge?" (vectors purged)":""}.`);
    loadDatasets();
  } catch(e){ setNotice("ds-notice","Error: "+e.message,true); }
	}

	// ── Monitor ──────────────────────────────────────────────────────────────────
	async function loadMonitor() {
	  try {
	    const r = await fetch("/admin/monitor", {headers: adminHeaders()});
	    const d = await r.json();
	    if(!r.ok) throw new Error(d.detail || r.statusText);
	    renderMonitor(d);
	    setNotice("monitor-notice","Loaded machine monitor.");
	  } catch(e) {
	    setNotice("monitor-notice","Load failed: "+e.message,true);
	  }
	}

	async function loadChunks() {
	  const qs = new URLSearchParams({
	    limit: document.getElementById("chunk-limit")?.value || "25"
	  });
	  const source = document.getElementById("chunk-source-filter")?.value.trim();
	  const status = document.getElementById("chunk-status-filter")?.value.trim();
	  if(source) qs.set("source", source);
	  if(status) qs.set("vector_status", status);
	  try {
	    const r = await fetch(`/admin/chunks?${qs.toString()}`, {headers: adminHeaders()});
	    const d = await r.json();
	    if(!r.ok) throw new Error(d.detail || r.statusText);
	    renderChunks(d.chunks || []);
	    setNotice("chunks-notice",`Loaded ${(d.chunks || []).length} row(s).`);
	  } catch(e) {
	    setNotice("chunks-notice","Load failed: "+e.message,true);
	  }
	}

	function renderChunks(rows) {
	  const el = document.getElementById("monitor-chunks");
	  if(!rows.length) {
	    el.innerHTML = '<div class="empty">No chunk rows match this filter.</div>';
	    return;
	  }
	  el.innerHTML = '<table><thead><tr><th>Source</th><th>Status</th><th>Preview</th><th>Created</th></tr></thead><tbody>' +
	    rows.map(r => `<tr>
	      <td><strong>${esc(r.source_title || r.source || "-")}</strong><br/><span style="font-size:11px;color:var(--muted)">${esc(r.source || "")} · ${esc(r.source_type || "")} · ${esc(r.language || "")}</span></td>
	      <td><span class="badge ${r.vector_status==="indexed"?"on":r.vector_status==="error"?"off":""}">${esc(r.vector_status || "-")}</span></td>
	      <td style="max-width:460px;line-height:1.45">${esc(r.preview || "")}${r.translated_preview ? `<br/><span style="color:var(--muted)">${esc(r.translated_preview)}</span>` : ""}<br/><span style="font-size:11px;color:var(--muted)">${esc(r.id || "")}</span></td>
	      <td>${r.created_at ? new Date(r.created_at).toLocaleString() : "-"}</td>
	    </tr>`).join("") + '</tbody></table>';
	}

	async function loadAudioJobs() {
	  try {
	    const r = await fetch("/admin/audio/jobs?limit=10", {headers: adminHeaders()});
	    const d = await r.json();
	    if(!r.ok) throw new Error(d.detail || r.statusText);
	    renderAudioJobs(d.jobs || []);
	    setNotice("audio-jobs-notice",`Loaded ${(d.jobs || []).length} audio job(s).`);
	  } catch(e) {
	    setNotice("audio-jobs-notice","Load failed: "+e.message,true);
	  }
	}

	function renderAudioJobs(jobs) {
	  const el = document.getElementById("monitor-audio-jobs");
	  if(!jobs.length) {
	    el.innerHTML = '<div class="empty">No audio jobs yet.</div>';
	    return;
	  }
	  el.innerHTML = '<table><thead><tr><th>File</th><th>Status</th><th>Stage</th><th>Splits</th><th>Chunks</th><th>Updated</th></tr></thead><tbody>' +
	    jobs.map(j => {
	      const m = j.metadata || {};
	      const total = Number(m.segments_total || 0);
	      const done = Number(m.segments_done || 0);
	      const failed = Number(m.segments_failed || 0);
	      const pct = total ? Math.round((done / total) * 100) : 0;
	      const splitText = total ? `${done}/${total} (${pct}%)${failed ? " · failed " + failed : ""}` : "-";
	      return `<tr>
	        <td><strong>${esc(j.source_title || j.file_name || "-")}</strong><br/><span style="font-size:11px;color:var(--muted)">${esc(j.file_name || "")}</span></td>
	        <td><span class="badge ${j.status==="ok"?"on":j.status==="failed"?"off":""}">${esc(j.status || "-")}</span></td>
	        <td>${esc(m.stage || "-")}${m.last_error ? `<br/><span style="font-size:11px;color:var(--redtext)">${esc(m.last_error)}</span>` : ""}</td>
	        <td>${splitText}</td>
	        <td>${j.chunks || 0}</td>
	        <td>${j.finished_at ? new Date(j.finished_at).toLocaleString() : "-"}</td>
	      </tr>`;
	    }).join("") + '</tbody></table>';
	}

	async function loadNotifications() {
	  try {
	    const r = await fetch("/admin/notifications?limit=50", {headers: adminHeaders()});
	    const d = await r.json();
	    if(!r.ok) throw new Error(d.detail || r.statusText);
	    renderNotifications(d.notifications || []);
	    setNotice("notifications-notice",`Loaded ${(d.notifications || []).length} notification(s).`);
	  } catch(e) {
	    setNotice("notifications-notice","Load failed: "+e.message,true);
	  }
	}

	function renderNotifications(rows) {
	  const el = document.getElementById("monitor-notifications");
	  if(!rows.length) {
	    el.innerHTML = '<div class="empty">No API failures recorded.</div>';
	    return;
	  }
	  el.innerHTML = '<table><thead><tr><th>Level</th><th>Event</th><th>File</th><th>Detail</th><th>When</th></tr></thead><tbody>' +
	    rows.map(n => `<tr>
	      <td><span class="badge ${n.level==="error"?"off":"on"}">${esc(n.level || "-")}</span></td>
	      <td>${esc(n.event || "-")}</td>
	      <td style="overflow-wrap:anywhere">${esc(n.file_name || "-")}</td>
	      <td style="max-width:420px;overflow-wrap:anywhere">${esc(n.detail || "")}</td>
	      <td>${n.created_at ? new Date(n.created_at).toLocaleString() : "-"}</td>
	    </tr>`).join("") + '</tbody></table>';
	}

	async function clearNotifications() {
	  try {
	    const r = await fetch("/admin/notifications", {method:"DELETE", headers: adminHeaders()});
	    const d = await r.json();
	    if(!r.ok) throw new Error(d.detail || r.statusText);
	    setNotice("notifications-notice",`Cleared ${d.deleted || 0} notification(s).`);
	    loadNotifications();
	  } catch(e) {
	    setNotice("notifications-notice","Clear failed: "+e.message,true);
	  }
	}

	const AUDIO_STEPS = [
	  ["upload", "Uploading audio to server"],
	  ["transcribe", "Transcribing speech"],
<<<<<<< HEAD
=======
	  ["translate", "Translating transcript when needed"],
>>>>>>> main
	  ["stage", "Writing chunks to Postgres"],
	  ["done", "Ready for Pinecone sync"]
	];

	function renderAudioProgress(activeKey, doneKeys=[], errorKey="") {
	  const el = document.getElementById("au-progress");
	  el.style.display = "grid";
	  el.innerHTML = AUDIO_STEPS.map(([key,label]) => {
	    const cls = errorKey === key ? "err" : doneKeys.includes(key) ? "done" : activeKey === key ? "active" : "";
	    return `<div class="progress-step ${cls}"><span class="progress-dot"></span><span>${esc(label)}</span></div>`;
	  }).join("");
	}

	function resetAudioProgress() {
	  const el = document.getElementById("au-progress");
	  el.style.display = "none";
	  el.innerHTML = "";
	}

	function statusList(rows) {
	  if(!rows || !rows.length) return '<div class="empty" style="padding:16px">No rows.</div>';
	  return '<div class="mini-list">' + rows.map(x =>
	    `<div class="mini-item"><span>${esc(x.vector_status || x.name || "-")}</span><strong>${esc(x.count || 0)}</strong></div>`
	  ).join("") + '</div>';
	}

	function renderMonitor(d) {
	  const pg = d.postgres || {};
	  const pc = d.pinecone || {};
	  const tables = pg.tables || {};
	  document.getElementById("monitor-postgres").innerHTML = `
	    <div class="metric-grid">
	      <div class="metric"><div class="metric-label">Postgres</div><div class="metric-value">${pg.ok ? "OK" : "Down"}</div></div>
	      <div class="metric"><div class="metric-label">Chunks</div><div class="metric-value">${tables.chunk_store || 0}</div></div>
	      <div class="metric"><div class="metric-label">Datasets</div><div class="metric-value">${tables.datasets || 0}</div></div>
	      <div class="metric"><div class="metric-label">Runs</div><div class="metric-value">${tables.ingestion_runs || 0}</div></div>
	    </div>
	    <div class="row two">
	      <div><div class="section-title">Vector Status</div>${statusList(pg.chunk_status || [])}</div>
	      <div><div class="section-title">Checkpoint Sources</div>${renderMonitorSources(pg.sources || [])}</div>
	    </div>
	    ${pg.error ? `<div class="notice err">${esc(pg.error)}</div>` : ""}`;

	  document.getElementById("monitor-pinecone").innerHTML = `
	    <div class="metric-grid">
	      <div class="metric"><div class="metric-label">Pinecone</div><div class="metric-value">${pc.ok ? "OK" : "Down"}</div></div>
	      <div class="metric"><div class="metric-label">Target Index</div><div class="metric-value" style="font-size:18px;overflow-wrap:anywhere">${esc(pc.target_index || "-")}</div></div>
	      <div class="metric"><div class="metric-label">Indexes</div><div class="metric-value">${(pc.indexes || []).length}</div></div>
	    </div>
	    ${renderPineconeIndexes(pc.indexes || [], pc.target_index)}
	    ${pc.error ? `<div class="notice err">${esc(pc.error)}</div>` : ""}`;
	}

	function renderMonitorSources(rows) {
	  if(!rows.length) return '<div class="empty" style="padding:16px">No checkpoint rows.</div>';
	  return '<table><thead><tr><th>Source</th><th>Type</th><th>Status</th><th>Rows</th></tr></thead><tbody>' +
	    rows.map(r => `<tr>
	      <td>${esc(r.source || "(blank)")}</td>
	      <td>${esc(r.source_type || "-")}</td>
	      <td>${esc(r.vector_status || "-")}</td>
	      <td>${r.count || 0}</td>
	    </tr>`).join("") + '</tbody></table>';
	}

	function renderPineconeIndexes(indexes, target) {
	  if(!indexes.length) return '<div class="empty">No Pinecone indexes visible for the configured API key.</div>';
	  return '<table><thead><tr><th>Index</th><th>Vectors</th><th>Namespaces</th><th>Status</th></tr></thead><tbody>' +
	    indexes.map(i => {
	      const stats = i.stats || {};
	      const ns = stats.namespaces ? Object.keys(stats.namespaces).join(", ") : "-";
	      return `<tr>
	        <td><strong>${esc(i.name)}</strong>${i.name===target?' <span class="badge on">Target</span>':''}</td>
	        <td>${stats.total_vector_count ?? "-"}</td>
	        <td>${esc(ns)}</td>
	        <td>${i.stats_error ? '<span class="badge off">Stats error</span>' : '<span class="badge on">Visible</span>'}</td>
	      </tr>`;
	    }).join("") + '</tbody></table>';
	}

	async function syncVectors(createIndex) {
	  const limit = document.getElementById("sync-limit").value || "100";
	  const indexName = document.getElementById("sync-index").value.trim();
	  const namespace = document.getElementById("sync-namespace").value.trim();
	  const qs = new URLSearchParams({limit, create_index: createIndex ? "true" : "false"});
	  if(indexName) qs.set("index_name", indexName);
	  if(namespace) qs.set("namespace", namespace);
	  setNotice("sync-notice","Syncing pending chunks…");
	  try {
	    const r = await fetch(`/admin/vector/sync?${qs.toString()}`, {method:"POST", headers: adminHeaders()});
	    const d = await r.json();
	    if(!r.ok) throw new Error(d.detail || r.statusText);
	    setNotice("sync-notice",`Synced ${d.vectors_upserted || 0} vector(s); selected ${d.selected || 0}.`);
	    loadMonitor();
	  } catch(e) {
	    setNotice("sync-notice","Sync failed: "+e.message,true);
	    loadMonitor();
	  }
	}

// ── Audio upload ──────────────────────────────────────────────────────────────
async function uploadAudio() {
  const file = document.getElementById("au-file").files?.[0];
  if(!file){ setNotice("au-notice","Choose an audio file first.",true); return; }
  const btn = document.getElementById("au-upload-btn");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("language_code", document.getElementById("au-lang").value);
  fd.append("dataset_name", document.getElementById("au-dataset").value.trim());
  fd.append("section", document.getElementById("au-section").value.trim());
  fd.append("description", document.getElementById("au-desc").value.trim() || file.name);
  const done = [];
  btn.disabled = true;
  btn.textContent = "Working…";
  setNotice("au-notice","Uploading audio…");
  renderAudioProgress("upload", done);
  const progressTimer = setInterval(() => {
    const states = [
      ["transcribe", ["upload"], "Audio uploaded. Transcribing speech…"],
<<<<<<< HEAD
      ["stage", ["upload","transcribe"], "Preparing chunks and writing to Postgres…"]
    ];
    const elapsed = Date.now() - startedAt;
    const idx = elapsed > 12000 ? 1 : elapsed > 2500 ? 0 : -1;
=======
      ["translate", ["upload","transcribe"], "Transcript received. Translating when needed…"],
      ["stage", ["upload","transcribe","translate"], "Preparing chunks and writing to Postgres…"]
    ];
    const elapsed = Date.now() - startedAt;
    const idx = elapsed > 45000 ? 2 : elapsed > 12000 ? 1 : elapsed > 2500 ? 0 : -1;
>>>>>>> main
    if(idx >= 0) {
      renderAudioProgress(states[idx][0], states[idx][1]);
      setNotice("au-notice", states[idx][2]);
    }
  }, 1200);
  const startedAt = Date.now();
  try {
    const r = await fetch(`${API}/audio/transcribe`, {method:"POST", headers:adminHeaders(), body:fd});
    if(!r.ok){ const b=await r.json(); throw new Error(b.detail||r.statusText); }
    const d = await r.json();
    clearInterval(progressTimer);
<<<<<<< HEAD
    renderAudioProgress("done", ["upload","transcribe","stage","done"]);
    setNotice("au-notice",`Done. ${d.chunks_created} chunks staged in Postgres.`);
=======
    renderAudioProgress("done", ["upload","transcribe","translate","stage","done"]);
    setNotice("au-notice",`Done. ${d.chunks_created} chunks staged in Postgres. Translation: ${d.translation_backend||"none"}.`);
>>>>>>> main
    showTab("monitor");
  } catch(e){
    clearInterval(progressTimer);
    renderAudioProgress("", done, "transcribe");
    setNotice("au-notice","Error: "+e.message,true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Transcribe & Stage";
  }
}

// ── Doc upload ────────────────────────────────────────────────────────────────
async function uploadDoc() {
  const file = document.getElementById("doc-file").files?.[0];
  if(!file){ setNotice("doc-notice","Choose a file first.",true); return; }
  const fd = new FormData();
  fd.append("file", file);
  fd.append("vector_db", document.getElementById("doc-db").value);
  fd.append("dataset_name", document.getElementById("doc-dataset").value.trim());
  setNotice("doc-notice","Uploading and indexing…");
  try {
    const r = await fetch("/admin/vector/upload", {method:"POST", headers:adminHeaders(), body:fd});
    if(!r.ok){ const b=await r.json(); throw new Error(b.detail||r.statusText); }
    const d = await r.json();
    setNotice("doc-notice",`Done. ${d.vectors_upserted} chunks indexed to ${d.vector_db}${d.dataset_id?" (dataset: "+d.dataset_id+")":""}.`);
    showTab("datasets");
  } catch(e){ setNotice("doc-notice","Error: "+e.message,true); }
}

// ── Query test ────────────────────────────────────────────────────────────────
async function runQuery() {
  const query = document.getElementById("q-text").value.trim();
  if(!query){ setNotice("q-notice","Enter a query.",true); return; }
  const btn = document.getElementById("q-btn");
  btn.disabled=true; btn.textContent="Asking…";
  setNotice("q-notice","");
  document.getElementById("q-result").style.display="none";
  try {
    const r = await fetch(`${API}/query`, {
      method:"POST", headers:adminHeaders({"Content-Type":"application/json"}),
      body: JSON.stringify({
        query,
        mode: document.getElementById("q-mode").value,
        language: document.getElementById("q-lang").value,
        filter_section: document.getElementById("q-section").value.trim() || null,
        history: []
      })
    });
    if(!r.ok){ const b=await r.json(); throw new Error(b.detail||r.statusText); }
    const d = await r.json();
    document.getElementById("q-answer").textContent = d.answer;
    const srcs = (d.sources||[]).map(s =>
      `<span class="source-chip">${esc(s.citation||"")}${s.score?" <span class='score'>"+s.score.toFixed(2)+"</span>":""}</span>`
    ).join("");
    document.getElementById("q-sources").innerHTML = srcs || "<span style='color:var(--muted);font-size:13px'>No sources retrieved.</span>";
    document.getElementById("q-result").style.display="block";
  } catch(e){ setNotice("q-notice","Error: "+e.message,true); }
  finally{ btn.disabled=false; btn.textContent="Ask"; }
}

document.getElementById("q-text").addEventListener("keydown", e => {
  if(e.key==="Enter" && (e.ctrlKey||e.metaKey)) runQuery();
});

// ── Gold store ────────────────────────────────────────────────────────────────
let pendingData=[], goldData=[], goldTab="pending";

function switchGoldTab(t){
  goldTab=t;
  document.getElementById("gtab-pending").style.opacity=t==="pending"?"1":".5";
  document.getElementById("gtab-gold").style.opacity=t==="gold"?"1":".5";
  renderGold();
}

async function loadGold() {
  setNotice("gold-notice","Loading…");
  try {
    const [pr,gr] = await Promise.all([
      fetch(`${API}/feedback/pending`, {headers:adminHeaders()}), fetch(`${API}/feedback/gold`, {headers:adminHeaders()})
    ]);
    pendingData = (await pr.json()).pending||[];
    goldData    = (await gr.json()).gold||[];
    setNotice("gold-notice",`${pendingData.length} pending, ${goldData.length} gold.`);
    renderGold();
  } catch(e){ setNotice("gold-notice","Load failed: "+e.message,true); }
}

function renderGold(){
  const data = goldTab==="pending" ? pendingData : goldData;
  const el = document.getElementById("gold-cards");
  if(!data.length){ el.innerHTML='<div class="empty">None.</div>'; return; }
  el.innerHTML = data.map(r => goldTab==="pending" ? pendingCard(r) : goldCardHtml(r)).join("");
}

function pendingCard(r){
  return `<div class="card" id="gc-${esc(r.query_id)}">
    <div style="font-weight:600;margin-bottom:8px">${esc(r.query)}</div>
    <textarea style="width:100%;min-height:100px;resize:vertical;background:rgba(2,6,23,.45);border:1px solid var(--line);border-radius:10px;color:var(--text);padding:10px;font-family:inherit" id="ga-${esc(r.query_id)}">${esc(r.answer)}</textarea>
    <div style="display:flex;gap:8px;margin-top:10px">
      <button class="btn-green btn-sm" onclick="reviewGold('${esc(r.query_id)}','approved')">Approve → Gold</button>
      <button class="btn-red btn-sm" onclick="reviewGold('${esc(r.query_id)}','rejected')">Reject</button>
    </div>
  </div>`;
}

function goldCardHtml(r){
  return `<div class="card">
    <div style="font-weight:600;margin-bottom:6px">${esc(r.query)}</div>
    <div class="answer-box" style="font-size:13px">${esc(r.gold_answer)}</div>
    <div style="font-size:11px;color:var(--muted)">Promoted ${r.promoted_at?new Date(r.promoted_at).toLocaleString():""}</div>
  </div>`;
}

async function reviewGold(qid, status){
  const ansEl = document.getElementById("ga-"+qid);
  const body = {review_status:status, gold_answer: status==="approved"?(ansEl?ansEl.value.trim():null):null};
  try {
    const r = await fetch(`${API}/feedback/${encodeURIComponent(qid)}`,{method:"PATCH",headers:adminHeaders({"Content-Type":"application/json"}),body:JSON.stringify(body)});
    if(!r.ok) throw new Error(r.statusText);
    pendingData = pendingData.filter(x=>x.query_id!==qid);
    document.getElementById("gc-"+qid)?.remove();
    setNotice("gold-notice", status==="approved"?"Approved.":"Rejected.");
  } catch(e){ setNotice("gold-notice","Error: "+e.message,true); }
}

// ── Translations ──────────────────────────────────────────────────────────────
let _tlEditId = null;

async function loadTranslations() {
  const source = document.getElementById("tl-filter-source").value.trim();
  const verified = document.getElementById("tl-filter-verified").value;
  const params = new URLSearchParams({limit: "200"});
  if (source) params.set("source", source);
  if (verified) params.set("verified", verified);
  setNotice("tl-notice", "Loading...");
  try {
    const r = await fetch("/admin/translations?" + params, {headers: adminHeaders()});
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();
    renderTl(d.translations || []);
    setNotice("tl-notice", d.count + " translation(s) loaded.");
  } catch(e) { setNotice("tl-notice", "Load failed: " + e.message, true); }
}

function renderTl(items) {
  const el = document.getElementById("tl-table");
  if (!items.length) { el.innerHTML = '<div class="empty">No translations yet.</div>'; return; }
  let html = '<div style="overflow-x:auto"><table><thead><tr><th>Source</th><th>Chunk</th><th>Original</th><th>Translation</th><th>Verified</th><th>Translator</th><th>Actions</th></tr></thead><tbody>';
  for (const t of items) {
    const orig = esc((t.original_text || "").slice(0, 100)) + (t.original_text && t.original_text.length > 100 ? "…" : "");
    const trans = esc((t.translated_text || "").slice(0, 100)) + (t.translated_text && t.translated_text.length > 100 ? "…" : "");
    html += `<tr id="tlrow-${esc(t.id)}">
      <td><strong>${esc(t.source_title || t.source)}</strong><br/><span style="font-size:11px;color:var(--muted)">${esc(t.source)}</span></td>
      <td style="text-align:center">${t.chunk_index !== null && t.chunk_index !== undefined ? t.chunk_index : "–"}</td>
      <td style="max-width:200px;overflow-wrap:anywhere;font-size:13px">${orig}</td>
      <td style="max-width:200px;overflow-wrap:anywhere;font-size:13px">${trans}</td>
      <td><span class="badge ${t.verified ? "on" : "off"}">${t.verified ? "✓ Yes" : "○ No"}</span></td>
      <td style="font-size:12px">${esc(t.translator_name || "–")}</td>
      <td style="display:flex;gap:6px;flex-wrap:wrap">
        <button class="btn-sm btn-green" onclick="editTl(${JSON.stringify(JSON.stringify(t))})">Edit</button>
        <button class="btn-sm ${t.verified ? "btn-ghost" : "btn-green"} btn-sm" onclick="toggleTlVerified('${esc(t.id)}', ${!t.verified})">${t.verified ? "Unverify" : "Verify"}</button>
        <button class="btn-sm btn-red" onclick="deleteTl('${esc(t.id)}')">Delete</button>
      </td>
    </tr>`;
  }
  html += "</tbody></table></div>";
  el.innerHTML = html;
}

function editTl(jsonStr) {
  const t = JSON.parse(jsonStr);
  _tlEditId = t.id;
  document.getElementById("tl-source").value = t.source || "";
  document.getElementById("tl-source-title").value = t.source_title || "";
  document.getElementById("tl-chunk-index").value = t.chunk_index !== null && t.chunk_index !== undefined ? t.chunk_index : "";
  document.getElementById("tl-vector-id").value = t.vector_chunk_id || "";
  document.getElementById("tl-section").value = t.section || "";
  document.getElementById("tl-orig-lang").value = t.original_language || "te";
  document.getElementById("tl-translator").value = t.translator_name || "";
  document.getElementById("tl-orig").value = t.original_text || "";
  document.getElementById("tl-trans").value = t.translated_text || "";
  document.getElementById("tl-notes").value = t.notes || "";
  document.getElementById("tl-form-title").textContent = "Edit Translation";
  document.getElementById("tl-save-btn").textContent = "Save Changes";
  document.getElementById("tl-cancel-btn").style.display = "inline-block";
  document.getElementById("tl-form-card").scrollIntoView({behavior: "smooth", block: "start"});
}

function cancelTl() {
  _tlEditId = null;
  ["tl-source","tl-source-title","tl-chunk-index","tl-vector-id","tl-section","tl-translator","tl-orig","tl-trans","tl-notes"].forEach(id => {
    document.getElementById(id).value = "";
  });
  document.getElementById("tl-orig-lang").value = "te";
  document.getElementById("tl-form-title").textContent = "Add Translation";
  document.getElementById("tl-save-btn").textContent = "Add Translation";
  document.getElementById("tl-cancel-btn").style.display = "none";
}

async function saveTl() {
  const source = document.getElementById("tl-source").value.trim();
  const orig = document.getElementById("tl-orig").value.trim();
  const trans = document.getElementById("tl-trans").value.trim();
  if (!source || !orig || !trans) {
    setNotice("tl-notice", "Source, original text, and translation are required.", true); return;
  }
  const chunkIdxRaw = document.getElementById("tl-chunk-index").value;
  const body = _tlEditId
    ? {
        translated_text: trans,
        notes: document.getElementById("tl-notes").value.trim() || null,
        translator_name: document.getElementById("tl-translator").value.trim() || null,
      }
    : {
        source,
        source_title: document.getElementById("tl-source-title").value.trim(),
        chunk_index: chunkIdxRaw !== "" ? parseInt(chunkIdxRaw) : null,
        vector_chunk_id: document.getElementById("tl-vector-id").value.trim() || null,
        original_text: orig,
        original_language: document.getElementById("tl-orig-lang").value,
        translated_text: trans,
        translator_name: document.getElementById("tl-translator").value.trim(),
        section: document.getElementById("tl-section").value.trim() || null,
        notes: document.getElementById("tl-notes").value.trim(),
      };
  const url = _tlEditId ? `/admin/translations/${encodeURIComponent(_tlEditId)}` : "/admin/translations";
  const method = _tlEditId ? "PATCH" : "POST";
  setNotice("tl-notice", "Saving...");
  try {
    const r = await fetch(url, {method, headers: adminHeaders({"Content-Type": "application/json"}), body: JSON.stringify(body)});
    if (!r.ok) { const b = await r.json(); throw new Error(b.detail || r.statusText); }
    setNotice("tl-notice", _tlEditId ? "Translation updated." : "Translation added.");
    cancelTl();
    loadTranslations();
  } catch(e) { setNotice("tl-notice", "Error: " + e.message, true); }
}

async function toggleTlVerified(id, verified) {
  setNotice("tl-notice", "Updating...");
  try {
    const r = await fetch(`/admin/translations/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: adminHeaders({"Content-Type": "application/json"}),
      body: JSON.stringify({verified}),
    });
    if (!r.ok) { const b = await r.json(); throw new Error(b.detail || r.statusText); }
    setNotice("tl-notice", verified ? "Marked as verified." : "Marked as unverified.");
    loadTranslations();
  } catch(e) { setNotice("tl-notice", "Error: " + e.message, true); }
}

async function deleteTl(id) {
  if (!confirm("Delete this translation?")) return;
  setNotice("tl-notice", "Deleting...");
  try {
    const r = await fetch(`/admin/translations/${encodeURIComponent(id)}`, {method: "DELETE", headers: adminHeaders()});
    if (!r.ok) { const b = await r.json(); throw new Error(b.detail || r.statusText); }
    setNotice("tl-notice", "Deleted.");
    loadTranslations();
  } catch(e) { setNotice("tl-notice", "Error: " + e.message, true); }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadDatasets();
loadSources();
</script>
</body>
</html>"""


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard() -> HTMLResponse:
    return HTMLResponse(_admin_page())
