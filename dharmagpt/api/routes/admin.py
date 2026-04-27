from __future__ import annotations

import json
import re
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core import dataset_store
from api.auth import require_admin_api_key
from core.config import get_settings
from core.local_vector_store import upsert_vectors
from core.retrieval import embed_texts_local, get_openai, get_pinecone

router = APIRouter()
settings = get_settings()


def _normalize_text(raw: str) -> str:
  return re.sub(r"\s+", " ", raw).strip()


def _chunk_text(raw: str, max_chars: int = 1200) -> list[str]:
  text = _normalize_text(raw)
  if not text:
    return []

  # Split first on paragraph breaks, then fold into bounded chunks.
  paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
  if not paragraphs:
    paragraphs = [text]

  chunks: list[str] = []
  buf = ""
  for para in paragraphs:
    para = _normalize_text(para)
    if not para:
      continue

    if len(para) > max_chars:
      sentences = re.split(r"(?<=[.!?])\s+", para)
      for sentence in sentences:
        sentence = _normalize_text(sentence)
        if not sentence:
          continue
        if not buf:
          buf = sentence
        elif len(buf) + 1 + len(sentence) <= max_chars:
          buf += " " + sentence
        else:
          chunks.append(buf)
          buf = sentence
      continue

    if not buf:
      buf = para
    elif len(buf) + 1 + len(para) <= max_chars:
      buf += " " + para
    else:
      chunks.append(buf)
      buf = para

  if buf:
    chunks.append(buf)

  return chunks


def _extract_text(filename: str, raw: bytes) -> str:
  suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
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


@router.post("/admin/vector/upload")
async def upload_document_to_vector_db(
  file: UploadFile = File(...),
  vector_db: str = Form("local"),
  index_name: str = Form(""),
  namespace: str = Form(""),
  dataset_name: str = Form(""),
  _: None = Depends(require_admin_api_key),
) -> dict:
  vector_db = (vector_db or "local").strip().lower()
  if vector_db not in {"local", "pinecone"}:
    raise HTTPException(status_code=400, detail="vector_db must be 'local' or 'pinecone'")

  filename = file.filename or "upload.txt"
  raw = await file.read()
  if not raw:
    raise HTTPException(status_code=400, detail="Uploaded file is empty")

  text = _extract_text(filename, raw)
  chunks = _chunk_text(text)
  if not chunks:
    raise HTTPException(status_code=400, detail="Could not extract usable text from uploaded file")

  if vector_db == "local":
    default_index = settings.local_vector_index_name
    default_namespace = settings.local_vector_namespace
  else:
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
    client = get_openai()
    embed_response = await client.embeddings.create(model=settings.embedding_model, input=chunks)
    vectors = [row.embedding for row in embed_response.data]
    embedding_backend = "openai"
  except Exception as exc:
    if vector_db != "local":
      raise HTTPException(status_code=502, detail="Cloud embedding provider unavailable") from exc
    vectors = embed_texts_local(chunks)
    embedding_backend = "local_hash"

  records = []
  doc_id = uuid.uuid4().hex[:12]
  for i, (chunk, vec) in enumerate(zip(chunks, vectors), start=1):
    meta = {
      "text": chunk,
      "source_type": "admin_upload",
      "citation": f"Admin upload: {filename}",
      "document_name": filename,
      "chunk_index": i,
    }
    if ds_id:
      meta["dataset_id"] = ds_id
    records.append({"id": f"admin-doc-{doc_id}-{i}", "values": vec, "metadata": meta})

  batch_size = 50
  upserted = 0
  if vector_db == "local":
    upserted = upsert_vectors(
      index_name=target_index,
      namespace=target_namespace,
      records=records,
    )
  else:
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
  }


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
          <option value="local" selected>Local (Beta)</option>
          <option value="pinecone">Pinecone (Later)</option>
        </select>
      </div>
      <div class="field" style="min-width: 220px;">
        <label for="indexName">Index Name</label>
        <input id="indexName" type="text" placeholder="dharma-local" />
      </div>
      <div class="field" style="min-width: 180px;">
        <label for="namespace">Namespace</label>
        <input id="namespace" type="text" placeholder="optional" />
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
      fd.append("vector_db", document.getElementById("vectorDb").value || "local");
      fd.append("index_name", document.getElementById("indexName").value || "");
      fd.append("namespace", document.getElementById("namespace").value || "");
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
          "Indexed " + data.vectors_upserted + " chunks into " + data.vector_db + ":" + data.index_name +
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
async def list_datasets() -> dict:
    return {"datasets": dataset_store.list_all()}


class DatasetToggle(BaseModel):
    active: bool


@router.patch("/admin/datasets/{name}")
async def toggle_dataset(name: str, body: DatasetToggle) -> dict:
    ok = dataset_store.set_active(name, body.active)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return {"name": name, "active": body.active}


@router.delete("/admin/datasets/{name}")
async def delete_dataset(name: str, purge_vectors: bool = False) -> dict:
    if purge_vectors and settings.vector_db_backend.lower() == "pinecone":
        try:
            pc = get_pinecone()
            index = pc.Index(settings.pinecone_index_name)
            index.delete(filter={"dataset_id": {"$eq": name}})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pinecone purge failed: {exc}")
    dataset_store.remove(name)
    return {"deleted": name, "vectors_purged": purge_vectors}


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
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:var(--text);background:radial-gradient(circle at top left,rgba(245,158,11,.18),transparent 28%),radial-gradient(circle at top right,rgba(59,130,246,.18),transparent 22%),linear-gradient(180deg,#0b1220 0%,#0f172a 100%);min-height:100vh}
    .shell{max-width:980px;margin:0 auto;padding:28px 20px}
    h1{margin:0 0 24px;font-size:24px}
    .tabs{display:flex;gap:4px;margin-bottom:24px;border-bottom:1px solid var(--line);padding-bottom:0}
    .tab{padding:10px 20px;border-radius:10px 10px 0 0;border:1px solid transparent;background:transparent;color:var(--muted);cursor:pointer;font-size:14px;font-weight:500;border-bottom:none;transition:color .15s}
    .tab.active{background:var(--panel);border-color:var(--line);border-bottom-color:var(--bg);color:var(--text)}
    .pane{display:none}.pane.active{display:block}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:16px;box-shadow:var(--shadow)}
    label{display:block;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
    input,select,textarea{width:100%;border-radius:10px;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:9px 13px;font-size:14px;outline:none;font-family:inherit}
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
    .empty{color:var(--muted);padding:30px;text-align:center;border:1px dashed var(--line);border-radius:14px}
    .answer-box{white-space:pre-wrap;line-height:1.65;font-size:14px;background:rgba(2,6,23,.45);border-radius:12px;border:1px solid var(--line);padding:14px;margin:12px 0}
    .source-chip{font-size:11px;padding:4px 8px;border-radius:8px;background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);color:#93c5fd;display:inline-block;margin:3px 3px 3px 0}
    .score{font-size:11px;color:var(--muted)}
    .section-title{font-size:16px;font-weight:600;margin-bottom:14px}
    .divider{border:0;border-top:1px solid var(--line);margin:20px 0}
  </style>
</head>
<body>
<div class="shell">
  <h1>DharmaGPT Admin</h1>
  <div class="tabs">
    <button class="tab active" onclick="showTab('datasets')">Datasets</button>
    <button class="tab" onclick="showTab('upload')">Upload</button>
    <button class="tab" onclick="showTab('test')">Test Query</button>
    <button class="tab" onclick="showTab('gold')">Gold Store</button>
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
      <button class="btn-primary" onclick="uploadAudio()">Transcribe & Index</button>
      <div class="notice" id="au-notice"></div>
    </div>

    <hr class="divider"/>

    <div class="card">
      <div class="section-title">Document Upload <span style="font-size:12px;color:var(--muted);font-weight:400">(txt / md / jsonl / json)</span></div>
      <div class="row three">
        <div><label>File</label><input type="file" id="doc-file" accept=".txt,.md,.jsonl,.json,.rst,.csv"/></div>
        <div><label>Dataset Name</label><input id="doc-dataset" type="text" placeholder="e.g. seed-corpus"/></div>
        <div><label>Vector DB</label>
          <select id="doc-db">
            <option value="pinecone">Pinecone</option>
            <option value="local">Local (SQLite)</option>
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
</div>

<script>
const API = "/api/v1";

// ── Tab switching ──────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t,i)=>{
    const names=["datasets","upload","test","gold"];
    t.classList.toggle("active", names[i]===name);
  });
  document.querySelectorAll(".pane").forEach(p => p.classList.remove("active"));
  document.getElementById("pane-"+name).classList.add("active");
  if(name==="datasets") loadDatasets();
  if(name==="gold") loadGold();
}

function setNotice(id, msg, err=false) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = "notice" + (err?" err":"");
}

function esc(s){ return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

// ── Datasets ──────────────────────────────────────────────────────────────────
async function loadDatasets() {
  try {
    const r = await fetch("/admin/datasets");
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

async function toggleDs(name, active) {
  try {
    const r = await fetch(`/admin/datasets/${encodeURIComponent(name)}`, {
      method:"PATCH", headers:{"Content-Type":"application/json"},
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
    const r = await fetch(`/admin/datasets/${encodeURIComponent(name)}?purge_vectors=${purge}`, {method:"DELETE"});
    if(!r.ok) throw new Error(await r.text());
    setNotice("ds-notice", `Deleted ${name}${purge?" (vectors purged)":""}.`);
    loadDatasets();
  } catch(e){ setNotice("ds-notice","Error: "+e.message,true); }
}

// ── Audio upload ──────────────────────────────────────────────────────────────
async function uploadAudio() {
  const file = document.getElementById("au-file").files?.[0];
  if(!file){ setNotice("au-notice","Choose an audio file first.",true); return; }
  const fd = new FormData();
  fd.append("file", file);
  fd.append("language_code", document.getElementById("au-lang").value);
  fd.append("dataset_name", document.getElementById("au-dataset").value.trim());
  fd.append("section", document.getElementById("au-section").value.trim());
  fd.append("description", document.getElementById("au-desc").value.trim() || file.name);
  setNotice("au-notice","Transcribing and indexing… this may take 30–90s.");
  try {
    const r = await fetch(`${API}/audio/transcribe`, {method:"POST", body:fd});
    if(!r.ok){ const b=await r.json(); throw new Error(b.detail||r.statusText); }
    const d = await r.json();
    setNotice("au-notice",`Done. ${d.chunks_created} chunks indexed. Translation: ${d.translation_backend||"none"}.`);
    showTab("datasets");
  } catch(e){ setNotice("au-notice","Error: "+e.message,true); }
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
    const r = await fetch("/admin/vector/upload", {method:"POST", body:fd});
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
      method:"POST", headers:{"Content-Type":"application/json"},
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
      fetch(`${API}/feedback/pending`), fetch(`${API}/feedback/gold`)
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
    const r = await fetch(`${API}/feedback/${encodeURIComponent(qid)}`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!r.ok) throw new Error(r.statusText);
    pendingData = pendingData.filter(x=>x.query_id!==qid);
    document.getElementById("gc-"+qid)?.remove();
    setNotice("gold-notice", status==="approved"?"Approved.":"Rejected.");
  } catch(e){ setNotice("gold-notice","Error: "+e.message,true); }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadDatasets();
</script>
</body>
</html>"""


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard() -> HTMLResponse:
    return HTMLResponse(_admin_page())
