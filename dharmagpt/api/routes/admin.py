from __future__ import annotations

import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from core.config import get_settings

router = APIRouter()
settings = get_settings()


def _admin_page() -> str:
    datasets = [d.strip() for d in settings.manual_translation_allowed_datasets.split(",") if d.strip()]
    datasets_json = json.dumps(datasets, ensure_ascii=False)
    default_dataset = datasets[0] if datasets else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DharmaGPT Manual Translation Admin</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: rgba(17, 24, 39, 0.88);
      --panel-2: rgba(15, 23, 42, 0.92);
      --text: #e5eefb;
      --muted: #96a7c3;
      --line: rgba(148, 163, 184, 0.22);
      --accent: #f59e0b;
      --accent-2: #22c55e;
      --danger: #ef4444;
      --warn: #f97316;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 22%),
        linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 16px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    .hero-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }}
    .hero-card {{
      padding: 24px;
    }}
    .eyebrow {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 6px 10px;
      border: 1px solid rgba(245, 158, 11, 0.35);
      border-radius: 999px;
      color: #ffd28b;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 8px;
      font-size: clamp(28px, 4vw, 48px);
      line-height: 1.04;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 70ch;
      margin: 0;
      font-size: 15px;
      line-height: 1.6;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat-value {{
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      padding: 20px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    input, select, textarea {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px 14px;
      outline: none;
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
    }}
    input:focus, select:focus, textarea:focus {{
      border-color: rgba(245, 158, 11, 0.6);
      box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.12);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      border-radius: 12px;
      padding: 11px 14px;
      font-weight: 650;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    .btn-primary {{ background: linear-gradient(135deg, #f59e0b, #f97316); color: #111827; }}
    .btn-secondary {{ background: rgba(148, 163, 184, 0.16); color: var(--text); border: 1px solid var(--line); }}
    .btn-good {{ background: rgba(34, 197, 94, 0.18); color: #bbf7d0; border: 1px solid rgba(34, 197, 94, 0.35); }}
    .btn-warn {{ background: rgba(249, 115, 22, 0.18); color: #fed7aa; border: 1px solid rgba(249, 115, 22, 0.35); }}
    .btn-danger {{ background: rgba(239, 68, 68, 0.18); color: #fecaca; border: 1px solid rgba(239, 68, 68, 0.35); }}
    .grid {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      padding: 18px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .list {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      background: rgba(15, 23, 42, 0.84);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      border: 1px solid var(--line);
      color: var(--muted);
    }}
    .chunk-index {{
      font-size: 18px;
      font-weight: 700;
    }}
    .text-block {{
      margin-top: 10px;
      display: grid;
      gap: 10px;
    }}
    .block-label {{
      color: #cbd5e1;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }}
    .text-preview {{
      white-space: pre-wrap;
      line-height: 1.6;
      color: #f8fafc;
      background: rgba(2, 6, 23, 0.45);
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px;
      min-height: 48px;
    }}
    .card-footer {{
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }}
    .notice {{
      margin-top: 12px;
      color: #dbeafe;
      font-size: 13px;
      min-height: 1.2em;
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
    }}
    .topline {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .keyline {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(148, 163, 184, 0.1);
      color: #dbeafe;
      font-size: 12px;
    }}
    .empty {{
      color: var(--muted);
      padding: 24px;
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.55);
    }}
    @media (max-width: 1100px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: 1fr; }}
      .card-footer {{ grid-template-columns: 1fr; }}
      .row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <section class="hero-card">
        <div class="eyebrow">Internal Review Console</div>
        <h1>DharmaGPT Manual Translation Admin</h1>
        <p class="subtitle">
          Review Telugu-to-English translations, approve or reject human edits, and keep a full audit trail.
          This page talks to the same JSONL-backed dataset API your ingestion and retrieval pipeline uses.
        </p>
        <div class="stats">
          <div class="stat"><div class="stat-label">Dataset scope</div><div class="stat-value" id="datasetCount">0</div></div>
          <div class="stat"><div class="stat-label">Pending chunks</div><div class="stat-value" id="pendingCount">-</div></div>
          <div class="stat"><div class="stat-label">API access</div><div class="stat-value">X-API-Key</div></div>
        </div>
      </section>
      <section class="hero-card controls">
        <div class="topline">
          <div>
            <div class="block-label">Access</div>
            <div class="meta">Store your API key locally in the browser and use it for all review actions.</div>
          </div>
        </div>
        <div>
          <label for="apiKey">API Key</label>
          <input id="apiKey" type="password" placeholder="Paste the employee API key" />
        </div>
        <div>
          <label for="datasetId">Dataset ID</label>
          <div id="datasetContainer"></div>
        </div>
        <div class="actions">
          <button class="btn-primary" id="loadBtn">Load Pending</button>
          <button class="btn-secondary" id="saveKeyBtn">Save Key</button>
          <button class="btn-secondary" id="clearKeyBtn">Clear Key</button>
        </div>
        <div class="notice" id="globalNotice"></div>
      </section>
    </div>

    <div class="grid">
      <aside class="panel">
        <h2>How it works</h2>
        <div class="meta">
          <p>1. Load a dataset by ID.</p>
          <p>2. Edit a translation and click <strong>Save Draft</strong>.</p>
          <p>3. Mark it <strong>Approved</strong>, <strong>Needs Work</strong>, or <strong>Rejected</strong>.</p>
          <p>4. Every change is written to the JSONL audit log.</p>
        </div>
        <div style="margin-top:16px;">
          <div class="block-label">Allowed datasets</div>
          <div id="datasetPills" class="keyline"></div>
        </div>
      </aside>

      <main class="panel">
        <div class="topline">
          <h2 style="margin:0;">Pending review items</h2>
          <div class="small" id="summaryLine">No dataset loaded yet.</div>
        </div>
        <div class="list" id="cards"></div>
      </main>
    </div>
  </div>

  <script>
    const ALLOWED_DATASETS = {datasets_json};
    const DEFAULT_DATASET = {json.dumps(default_dataset)};
    const API_BASE = "/api/v1/audio/manual-translations";
    const keyInput = document.getElementById("apiKey");
    const datasetContainer = document.getElementById("datasetContainer");
    const cards = document.getElementById("cards");
    const globalNotice = document.getElementById("globalNotice");
    const summaryLine = document.getElementById("summaryLine");
    const pendingCount = document.getElementById("pendingCount");
    const datasetCount = document.getElementById("datasetCount");
    const datasetPills = document.getElementById("datasetPills");

    function setNotice(text, isError=false) {{
      globalNotice.style.color = isError ? "#fecaca" : "#dbeafe";
      globalNotice.textContent = text;
    }}

    function getApiKey() {{
      return localStorage.getItem("dharmagpt_manual_api_key") || "";
    }}

    function setApiKey(value) {{
      localStorage.setItem("dharmagpt_manual_api_key", value);
      keyInput.value = value;
    }}

    function datasetField() {{
      return document.getElementById("datasetId");
    }}

    function getDataset() {{
      const field = datasetField();
      return (field && field.value ? field.value : DEFAULT_DATASET).trim();
    }}

    function apiHeaders() {{
      const headers = {{"Content-Type": "application/json"}};
      const key = getApiKey().trim();
      if (key) headers["X-API-Key"] = key;
      return headers;
    }}

    async function apiFetch(path, options={{}}) {{
      const resp = await fetch(path, {{
        ...options,
        headers: {{
          ...apiHeaders(),
          ...(options.headers || {{}})
        }}
      }});
      if (!resp.ok) {{
        let detail = resp.statusText;
        try {{
          const body = await resp.json();
          detail = body.detail || JSON.stringify(body);
        }} catch (e) {{}}
        throw new Error(detail);
      }}
      return resp.json();
    }}

    function buildDatasetUI() {{
      if (ALLOWED_DATASETS.length > 0) {{
        datasetContainer.innerHTML = '<select id="datasetId"></select>';
        const select = datasetField();
        for (const ds of ALLOWED_DATASETS) {{
          const opt = document.createElement("option");
          opt.value = ds;
          opt.textContent = ds;
          select.appendChild(opt);
        }}
      }} else {{
        datasetContainer.innerHTML = '<input id="datasetId" type="text" placeholder="telugu_ramayan" />';
        const input = datasetField();
        input.value = DEFAULT_DATASET;
      }}
      const field = datasetField();
      if (DEFAULT_DATASET && field) field.value = DEFAULT_DATASET;
      datasetCount.textContent = String(ALLOWED_DATASETS.length || 0);
      datasetPills.innerHTML = "";
      if (ALLOWED_DATASETS.length > 0) {{
        for (const ds of ALLOWED_DATASETS) {{
          const pill = document.createElement("span");
          pill.className = "pill";
          pill.textContent = ds;
          datasetPills.appendChild(pill);
        }}
      }} else {{
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = "No allowlist configured";
        datasetPills.appendChild(pill);
      }}
    }}

    function chunkCard(item) {{
      const wrapper = document.createElement("section");
      wrapper.className = "card";
      wrapper.innerHTML = `
        <div class="card-head">
          <div>
            <div class="chunk-index">Chunk #${{item.chunk_index}}</div>
            <div class="small">${{item.review_status || "pending"}}${{item.reviewer ? " • " + item.reviewer : ""}}</div>
          </div>
          <div class="chip">Reviewed: ${{item.reviewed_at || "not yet"}}</div>
        </div>
        <div class="text-block">
          <div>
            <div class="block-label">Telugu source</div>
            <div class="text-preview">${{item.text_te || ""}}</div>
          </div>
          <div>
            <div class="block-label">Model translation</div>
            <div class="text-preview">${{item.text_en_model || ""}}</div>
          </div>
          <div>
            <div class="block-label">Manual translation</div>
            <textarea class="manualText" placeholder="Enter or refine the English translation">${{item.text_en_manual || ""}}</textarea>
          </div>
        </div>
        <div class="card-footer">
          <div>
            <label>Reviewer</label>
            <input class="reviewer" type="text" placeholder="employee@company.com" value="${{item.reviewer || ""}}" />
          </div>
          <div>
            <label>Review note</label>
            <input class="note" type="text" placeholder="Optional note for the review log" value="${{item.review_note || ""}}" />
          </div>
          <div>
            <label>Status</label>
            <select class="status">
              <option value="pending" ${{item.review_status === "pending" ? "selected" : ""}}>Pending</option>
              <option value="approved" ${{item.review_status === "approved" ? "selected" : ""}}>Approved</option>
              <option value="needs_work" ${{item.review_status === "needs_work" ? "selected" : ""}}>Needs work</option>
              <option value="rejected" ${{item.review_status === "rejected" ? "selected" : ""}}>Rejected</option>
            </select>
          </div>
        </div>
        <div class="actions" style="margin-top: 12px;">
          <button class="btn-secondary saveBtn">Save Draft</button>
          <button class="btn-good approveBtn">Approve</button>
          <button class="btn-warn needsWorkBtn">Needs Work</button>
          <button class="btn-danger rejectBtn">Reject</button>
        </div>
        <div class="notice"></div>
      `;

      const notice = wrapper.querySelector(".notice");
      const manualText = wrapper.querySelector(".manualText");
      const reviewer = wrapper.querySelector(".reviewer");
      const note = wrapper.querySelector(".note");
      const status = wrapper.querySelector(".status");

      async function saveDraft(nextStatus = status.value) {{
        try {{
          notice.textContent = "Saving...";
          const dataset = getDataset();
          await apiFetch(`${{API_BASE}}/chunk`, {{
            method: "POST",
            body: JSON.stringify({{
              dataset_id: dataset,
              chunk_index: item.chunk_index,
              text_en_manual: manualText.value.trim(),
              reviewer: reviewer.value.trim() || null,
              review_status: nextStatus,
              review_note: note.value.trim() || null
            }})
          }});
          notice.textContent = "Saved.";
          await loadPending();
        }} catch (err) {{
          notice.textContent = err.message;
          notice.style.color = "#fecaca";
        }}
      }}

      async function setReview(nextStatus) {{
        try {{
          notice.textContent = "Updating review...";
          const dataset = getDataset();
          await apiFetch(`${{API_BASE}}/review`, {{
            method: "POST",
            body: JSON.stringify({{
              dataset_id: dataset,
              chunk_index: item.chunk_index,
              review_status: nextStatus,
              reviewer: reviewer.value.trim() || null,
              review_note: note.value.trim() || null
            }})
          }});
          notice.textContent = `Marked ${{nextStatus}}.`;
          await loadPending();
        }} catch (err) {{
          notice.textContent = err.message;
          notice.style.color = "#fecaca";
        }}
      }}

      wrapper.querySelector(".saveBtn").addEventListener("click", () => saveDraft(status.value));
      wrapper.querySelector(".approveBtn").addEventListener("click", () => saveDraft("approved"));
      wrapper.querySelector(".needsWorkBtn").addEventListener("click", () => saveDraft("needs_work"));
      wrapper.querySelector(".rejectBtn").addEventListener("click", () => saveDraft("rejected"));
      wrapper.querySelector(".status").addEventListener("change", (e) => {{
        if (e.target.value === "approved") saveDraft("approved");
      }});

      return wrapper;
    }}

    async function loadPending() {{
      try {{
        const dataset = getDataset();
        if (!dataset) {{
          cards.innerHTML = '<div class="empty">No dataset selected.</div>';
          return;
        }}
        setNotice(`Loading ${dataset}...`);
        const data = await apiFetch(`${{API_BASE}}/datasets/${{encodeURIComponent(dataset)}}/pending`, {{
          method: "GET",
          headers: {{}}
        }});
        pendingCount.textContent = String(data.pending_chunks.length);
        summaryLine.textContent = `${{data.pending_chunks.length}} pending / ${{data.total_chunks}} total`;
        cards.innerHTML = "";
        if (!data.pending_chunks.length) {{
          cards.innerHTML = '<div class="empty">No pending chunks for this dataset.</div>';
        }} else {{
          for (const item of data.pending_chunks) {{
            cards.appendChild(chunkCard(item));
          }}
        }}
        setNotice(`Loaded ${dataset}.`);
      }} catch (err) {{
        setNotice(err.message, true);
        cards.innerHTML = `<div class="empty">${{err.message}}</div>`;
      }}
    }}

    document.getElementById("saveKeyBtn").addEventListener("click", () => {{
      setApiKey(keyInput.value.trim());
      setNotice("API key saved locally in this browser.");
    }});

    document.getElementById("clearKeyBtn").addEventListener("click", () => {{
      localStorage.removeItem("dharmagpt_manual_api_key");
      keyInput.value = "";
      setNotice("API key cleared.");
    }});

    document.getElementById("loadBtn").addEventListener("click", loadPending);

    buildDatasetUI();
    datasetField()?.addEventListener("change", loadPending);
    keyInput.value = getApiKey();
    if (DEFAULT_DATASET) {{
      const field = datasetField();
      if (field) field.value = DEFAULT_DATASET;
      loadPending();
    }} else {{
      setNotice("Add MANUAL_TRANSLATION_ALLOWED_DATASETS to enable a drop-down dataset selector.", true);
    }}
  </script>
</body>
</html>"""


@router.get("/admin/manual-translations", response_class=HTMLResponse)
async def manual_translation_admin() -> HTMLResponse:
    return HTMLResponse(_admin_page())
