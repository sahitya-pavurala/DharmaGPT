from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


def _chat_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DharmaGPT</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0f0f0f;
      --surface: #1a1a1a;
      --surface2: #222;
      --line: #2e2e2e;
      --text: #e8e8e8;
      --muted: #888;
      --accent: #c8a96e;
      --accent-dim: rgba(200,169,110,0.12);
      --user-bg: #1e2a1e;
      --user-border: #2d4a2d;
      --ans-bg: #1a1a2e;
      --ans-border: #2a2a4a;
      --red: #ef4444;
      --green: #22c55e;
      --radius: 12px;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 15px; height: 100dvh; display: flex; flex-direction: column; overflow: hidden; }

    /* ── Key gate ── */
    #key-gate {
      position: fixed; inset: 0; background: var(--bg); z-index: 100;
      display: flex; align-items: center; justify-content: center;
    }
    #key-gate .gate-card {
      background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
      padding: 36px 40px; width: min(440px, 92vw); display: flex; flex-direction: column; gap: 16px;
    }
    #key-gate h2 { font-size: 20px; font-weight: 600; color: var(--accent); }
    #key-gate p { font-size: 13px; color: var(--muted); line-height: 1.5; }
    #key-gate input { width: 100%; padding: 10px 14px; background: var(--surface2); border: 1px solid var(--line); border-radius: 8px; color: var(--text); font-size: 14px; outline: none; }
    #key-gate input:focus { border-color: var(--accent); }

    /* ── Layout ── */
    #app { display: flex; height: 100%; overflow: hidden; }

    /* ── Sidebar ── */
    #sidebar {
      width: 260px; flex-shrink: 0; background: var(--surface); border-right: 1px solid var(--line);
      display: flex; flex-direction: column; padding: 20px 16px; gap: 20px; overflow-y: auto;
    }
    .sidebar-logo { font-size: 18px; font-weight: 700; color: var(--accent); letter-spacing: 0.3px; }
    .sidebar-logo span { font-size: 12px; font-weight: 400; color: var(--muted); display: block; margin-top: 2px; }
    .sidebar-section { display: flex; flex-direction: column; gap: 8px; }
    .sidebar-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); font-weight: 600; }
    select, input[type="text"] {
      width: 100%; padding: 8px 10px; background: var(--surface2); border: 1px solid var(--line);
      border-radius: 8px; color: var(--text); font-size: 13px; outline: none;
    }
    select:focus, input[type="text"]:focus { border-color: var(--accent); }
    .sidebar-footer { margin-top: auto; font-size: 11px; color: var(--muted); line-height: 1.6; }
    .sidebar-footer a { color: var(--accent); text-decoration: none; }
    #change-key-btn { background: none; border: none; color: var(--muted); font-size: 11px; cursor: pointer; text-align: left; padding: 0; }
    #change-key-btn:hover { color: var(--text); }
    #new-chat-btn {
      width: 100%; padding: 9px; background: var(--accent-dim); border: 1px solid var(--accent);
      border-radius: 8px; color: var(--accent); font-size: 13px; font-weight: 500; cursor: pointer;
    }
    #new-chat-btn:hover { background: rgba(200,169,110,0.2); }

    /* ── Chat area ── */
    #chat-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    #thread { flex: 1; overflow-y: auto; padding: 24px 0; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
    #thread::-webkit-scrollbar { width: 4px; }
    #thread::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }

    /* ── Messages ── */
    .msg-row { display: flex; padding: 0 20px; }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.assistant { justify-content: flex-start; }
    .bubble {
      max-width: min(680px, 85%); padding: 14px 18px; border-radius: var(--radius);
      line-height: 1.65; font-size: 15px; white-space: pre-wrap; word-break: break-word;
    }
    .bubble.user { background: var(--user-bg); border: 1px solid var(--user-border); border-bottom-right-radius: 4px; }
    .bubble.assistant { background: var(--ans-bg); border: 1px solid var(--ans-border); border-bottom-left-radius: 4px; }

    /* ── Sources ── */
    .sources-row { padding: 0 20px; display: flex; flex-direction: column; gap: 6px; }
    .sources-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
    .source-chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .source-chip {
      display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
      background: var(--surface); border: 1px solid var(--line); border-radius: 999px;
      font-size: 12px; color: var(--muted); cursor: pointer; transition: border-color 0.15s, color 0.15s;
    }
    .source-chip:hover { border-color: var(--accent); color: var(--text); }
    .source-chip .score { font-size: 10px; color: var(--accent); }
    .source-detail {
      background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
      padding: 12px 14px; font-size: 13px; color: var(--muted); line-height: 1.6;
      display: none; margin-top: 4px;
    }
    .source-detail.open { display: block; }
    .source-detail .src-meta { font-size: 11px; color: var(--accent); margin-bottom: 6px; font-weight: 600; }

    /* ── Feedback + debug ── */
    .feedback-row { padding: 4px 20px; display: flex; align-items: center; gap: 10px; }
    .fb-btn {
      background: none; border: 1px solid var(--line); border-radius: 6px; color: var(--muted);
      font-size: 14px; padding: 3px 8px; cursor: pointer; transition: all 0.15s;
    }
    .fb-btn:hover { border-color: var(--accent); color: var(--accent); }
    .fb-btn.active-up { border-color: var(--green); color: var(--green); }
    .fb-btn.active-down { border-color: var(--red); color: var(--red); }
    .fb-label { font-size: 11px; color: var(--muted); }
    .debug-toggle { background: none; border: none; color: var(--muted); font-size: 11px; cursor: pointer; margin-left: auto; }
    .debug-toggle:hover { color: var(--text); }
    .debug-drawer {
      margin: 0 20px 4px; background: var(--surface); border: 1px solid var(--line);
      border-radius: 8px; padding: 10px 14px; font-size: 11px; color: var(--muted);
      font-family: monospace; display: none; line-height: 1.8;
    }
    .debug-drawer.open { display: block; }

    /* ── Typing bar ── */
    #input-bar {
      border-top: 1px solid var(--line); padding: 16px 20px;
      display: flex; gap: 10px; align-items: flex-end; background: var(--bg);
    }
    #input-bar textarea {
      flex: 1; background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
      color: var(--text); font-family: var(--font); font-size: 15px; padding: 12px 14px;
      resize: none; outline: none; max-height: 180px; overflow-y: auto; line-height: 1.5;
    }
    #input-bar textarea:focus { border-color: var(--accent); }
    #send-btn {
      background: var(--accent); border: none; border-radius: 10px; color: #000;
      font-weight: 600; font-size: 14px; padding: 12px 20px; cursor: pointer; white-space: nowrap;
      flex-shrink: 0; transition: opacity 0.15s;
    }
    #send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* ── Empty state ── */
    #empty-state {
      flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 8px; color: var(--muted); padding: 40px;
    }
    #empty-state h3 { font-size: 22px; color: var(--accent); font-weight: 600; }
    #empty-state p { font-size: 14px; text-align: center; max-width: 380px; line-height: 1.6; }
    .suggest-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px; }
    .suggest-chip {
      padding: 8px 14px; background: var(--surface); border: 1px solid var(--line);
      border-radius: 999px; font-size: 13px; color: var(--muted); cursor: pointer;
    }
    .suggest-chip:hover { border-color: var(--accent); color: var(--text); }

    /* ── Thinking indicator ── */
    .thinking { display: flex; gap: 5px; align-items: center; padding: 14px 18px; }
    .thinking span { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); animation: pulse 1.2s infinite; }
    .thinking span:nth-child(2) { animation-delay: 0.2s; }
    .thinking span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse { 0%,80%,100%{opacity:.2} 40%{opacity:1} }

    /* ── Mobile ── */
    @media(max-width: 640px) {
      #sidebar { display: none; }
      #thread { padding: 16px 0; gap: 14px; }
      .msg-row { padding: 0 12px; }
      .sources-row, .feedback-row, .debug-drawer { padding-left: 12px; padding-right: 12px; }
      .bubble { max-width: 92%; font-size: 14px; }
      #input-bar { padding: 12px; }
    }
  </style>
</head>
<body>

<!-- Key gate -->
<div id="key-gate">
  <div class="gate-card">
    <h2>DharmaGPT</h2>
    <p>Enter your API key to begin. It will be saved in your browser for this session.</p>
    <input id="gate-key-input" type="password" placeholder="API key" autocomplete="off"/>
    <button class="btn-primary" id="gate-submit" style="padding:10px;background:var(--accent);border:none;border-radius:8px;color:#000;font-weight:600;font-size:14px;cursor:pointer;">Enter</button>
    <div id="gate-error" style="font-size:13px;color:var(--red);display:none"></div>
  </div>
</div>

<div id="app" style="display:none">
  <!-- Sidebar -->
  <div id="sidebar">
    <div class="sidebar-logo">DharmaGPT <span>Sacred text assistant</span></div>

    <button id="new-chat-btn" onclick="newChat()">+ New conversation</button>

    <div class="sidebar-section">
      <div class="sidebar-label">Mode</div>
      <select id="mode">
        <option value="guidance">Guidance</option>
        <option value="story">Story</option>
        <option value="scholar">Scholar</option>
        <option value="children">Children</option>
      </select>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Language</div>
      <select id="lang">
        <option value="en">English</option>
        <option value="te">Telugu</option>
        <option value="hi">Hindi</option>
      </select>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">Section filter <span style="font-weight:400;text-transform:none;letter-spacing:0">(optional)</span></div>
      <input type="text" id="section-filter" placeholder="e.g. Bala Kanda"/>
    </div>

    <div class="sidebar-footer">
      <button id="change-key-btn" onclick="changeKey()">Change API key</button>
    </div>
  </div>

  <!-- Chat area -->
  <div id="chat-area">
    <div id="thread">
      <div id="empty-state">
        <h3>Ask the Ramayana</h3>
        <p>Ask about characters, stories, dharmic teachings, or any passage from the sacred texts.</p>
        <div class="suggest-chips">
          <div class="suggest-chip" onclick="suggest(this)">Who is Hanuman?</div>
          <div class="suggest-chip" onclick="suggest(this)">What is the significance of Rama's exile?</div>
          <div class="suggest-chip" onclick="suggest(this)">Describe the Sundara Kanda</div>
          <div class="suggest-chip" onclick="suggest(this)">What virtues does Rama embody?</div>
        </div>
      </div>
    </div>

    <div id="input-bar">
      <textarea id="query-input" rows="1" placeholder="Ask anything…"></textarea>
      <button id="send-btn" onclick="send()">Send</button>
    </div>
  </div>
</div>

<script>
const API = "/api/v1";
let history = [];
let apiKey = "";

// ── Key gate ──────────────────────────────────────────────────────────────────
function savedKey() { return localStorage.getItem("dharmagpt.apiKey") || ""; }

function initKey() {
  const k = savedKey();
  if (k) { apiKey = k; showApp(); }
}

function showApp() {
  document.getElementById("key-gate").style.display = "none";
  document.getElementById("app").style.display = "flex";
  document.getElementById("query-input").focus();
}

function changeKey() {
  document.getElementById("gate-key-input").value = "";
  document.getElementById("gate-error").style.display = "none";
  document.getElementById("key-gate").style.display = "flex";
  document.getElementById("app").style.display = "none";
}

document.getElementById("gate-submit").addEventListener("click", async () => {
  const k = document.getElementById("gate-key-input").value.trim();
  if (!k) return;
  // Validate key with a quick health-style probe
  const err = document.getElementById("gate-error");
  err.style.display = "none";
  try {
    const r = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": k },
      body: JSON.stringify({ query: "test", mode: "guidance", history: [], language: "en" }),
    });
    if (r.status === 401) { err.textContent = "Invalid API key."; err.style.display = "block"; return; }
  } catch(_) {}
  apiKey = k;
  localStorage.setItem("dharmagpt.apiKey", k);
  showApp();
});

document.getElementById("gate-key-input").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("gate-submit").click();
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function headers() { return { "Content-Type": "application/json", "X-API-Key": apiKey }; }

function autoResize(ta) {
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
}

// ── New chat ──────────────────────────────────────────────────────────────────
function newChat() {
  history = [];
  const thread = document.getElementById("thread");
  thread.innerHTML = `<div id="empty-state">
    <h3>Ask the Ramayana</h3>
    <p>Ask about characters, stories, dharmic teachings, or any passage from the sacred texts.</p>
    <div class="suggest-chips">
      <div class="suggest-chip" onclick="suggest(this)">Who is Hanuman?</div>
      <div class="suggest-chip" onclick="suggest(this)">What is the significance of Rama's exile?</div>
      <div class="suggest-chip" onclick="suggest(this)">Describe the Sundara Kanda</div>
      <div class="suggest-chip" onclick="suggest(this)">What virtues does Rama embody?</div>
    </div>
  </div>`;
}

function suggest(el) { document.getElementById("query-input").value = el.textContent; send(); }

// ── Render helpers ────────────────────────────────────────────────────────────
function appendUserBubble(text) {
  const thread = document.getElementById("thread");
  const empty = document.getElementById("empty-state");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="bubble user">${esc(text)}</div>`;
  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
  return row;
}

function appendThinking() {
  const thread = document.getElementById("thread");
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.id = "thinking-indicator";
  row.innerHTML = `<div class="bubble assistant thinking"><span></span><span></span><span></span></div>`;
  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
  return row;
}

function renderAnswer(data) {
  const thread = document.getElementById("thread");
  const thinking = document.getElementById("thinking-indicator");
  if (thinking) thinking.remove();

  // Answer bubble
  const ansRow = document.createElement("div");
  ansRow.className = "msg-row assistant";
  ansRow.innerHTML = `<div class="bubble assistant">${esc(data.answer)}</div>`;
  thread.appendChild(ansRow);

  // Sources
  if (data.sources && data.sources.length > 0) {
    const srcRow = document.createElement("div");
    srcRow.className = "sources-row";
    const chips = data.sources.map((s, i) => {
      const id = `src-${data.query_id}-${i}`;
      return `<span class="source-chip" onclick="toggleSource('${id}')">
        ${esc(s.citation || "Source")}${s.section ? " · " + esc(s.section) : ""}
        <span class="score">${s.score ? s.score.toFixed(2) : ""}</span>
      </span>
      <div class="source-detail" id="${id}">
        <div class="src-meta">${esc(s.citation || "")}${s.section ? " — " + esc(s.section) : ""}</div>
        ${esc(s.text || "")}
      </div>`;
    }).join("");
    srcRow.innerHTML = `<div class="sources-label">Sources</div><div class="source-chips">${chips}</div>`;
    thread.appendChild(srcRow);
  }

  // Feedback + debug
  const fbRow = document.createElement("div");
  fbRow.className = "feedback-row";
  const debugId = `debug-${data.query_id}`;
  fbRow.innerHTML = `
    <span class="fb-label">Helpful?</span>
    <button class="fb-btn" id="fb-up-${data.query_id}" onclick="sendFeedback('${data.query_id}', 'up', this)">&#128077;</button>
    <button class="fb-btn" id="fb-down-${data.query_id}" onclick="sendFeedback('${data.query_id}', 'down', this)">&#128078;</button>
    <button class="debug-toggle" onclick="toggleDebug('${debugId}')">debug ▾</button>`;
  thread.appendChild(fbRow);

  const debugDrawer = document.createElement("div");
  debugDrawer.className = "debug-drawer";
  debugDrawer.id = debugId;
  debugDrawer.innerHTML =
    `query_id: ${esc(data.query_id)}<br>` +
    `llm_backend: ${esc(data.llm_backend || "—")}<br>` +
    `llm_model: ${esc(data.llm_model || "—")}<br>` +
    `mode: ${esc(data.mode)}<br>` +
    `language: ${esc(data.language)}` +
    (data.llm_fallback_reason ? `<br>fallback_reason: ${esc(data.llm_fallback_reason)}` : "");
  thread.appendChild(debugDrawer);

  thread.scrollTop = thread.scrollHeight;
}

function toggleSource(id) {
  document.getElementById(id).classList.toggle("open");
}

function toggleDebug(id) {
  document.getElementById(id).classList.toggle("open");
}

// ── Feedback ──────────────────────────────────────────────────────────────────
async function sendFeedback(queryId, rating, btn) {
  const upBtn = document.getElementById(`fb-up-${queryId}`);
  const downBtn = document.getElementById(`fb-down-${queryId}`);
  upBtn.classList.remove("active-up"); downBtn.classList.remove("active-down");
  if (rating === "up") upBtn.classList.add("active-up");
  else downBtn.classList.add("active-down");

  // Find the query text and answer from history
  const lastUser = history.filter(m => m.role === "user").slice(-1)[0];
  const lastAsst = history.filter(m => m.role === "assistant").slice(-1)[0];
  try {
    await fetch(`${API}/feedback/${queryId}`, {
      method: "PATCH",
      headers: headers(),
      body: JSON.stringify({
        query_id: queryId,
        query: lastUser?.content || "",
        answer: lastAsst?.content || "",
        rating,
        mode: document.getElementById("mode").value,
      }),
    });
  } catch(_) {}
}

// ── Send ──────────────────────────────────────────────────────────────────────
async function send() {
  const ta = document.getElementById("query-input");
  const text = ta.value.trim();
  if (!text) return;

  const btn = document.getElementById("send-btn");
  btn.disabled = true;
  ta.value = "";
  ta.style.height = "auto";

  appendUserBubble(text);
  appendThinking();

  history.push({ role: "user", content: text });

  try {
    const r = await fetch(`${API}/query`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({
        query: text,
        mode: document.getElementById("mode").value,
        language: document.getElementById("lang").value,
        filter_section: document.getElementById("section-filter").value.trim() || null,
        history: history.slice(-10),
      }),
    });

    if (r.status === 401) { changeKey(); return; }
    if (!r.ok) { const b = await r.json(); throw new Error(b.detail || r.statusText); }

    const data = await r.json();
    history.push({ role: "assistant", content: data.answer });
    renderAnswer(data);
  } catch(e) {
    const thinking = document.getElementById("thinking-indicator");
    if (thinking) thinking.remove();
    const thread = document.getElementById("thread");
    const errRow = document.createElement("div");
    errRow.className = "msg-row assistant";
    errRow.innerHTML = `<div class="bubble assistant" style="color:var(--red)">${esc(e.message)}</div>`;
    thread.appendChild(errRow);
    thread.scrollTop = thread.scrollHeight;
    history.pop();
  } finally {
    btn.disabled = false;
    ta.focus();
  }
}

// ── Input auto-resize + keyboard shortcut ────────────────────────────────────
const ta = document.getElementById("query-input");
ta.addEventListener("input", () => autoResize(ta));
ta.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

initKey();
</script>
</body>
</html>"""


@router.get("/query", response_class=HTMLResponse)
async def query_page() -> HTMLResponse:
    return HTMLResponse(_chat_page())
