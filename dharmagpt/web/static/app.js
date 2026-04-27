const form = document.querySelector("#queryForm");
const messages = document.querySelector("#messages");
const queryInput = document.querySelector("#query");
const modeInput = document.querySelector("#mode");
const languageInput = document.querySelector("#language");
const apiKeyInput = document.querySelector("#apiKey");
const sendButton = document.querySelector("#sendButton");
const healthStatus = document.querySelector("#healthStatus");

const savedKey = localStorage.getItem("dharmagpt.betaKey");
if (savedKey) {
  apiKeyInput.value = savedKey;
}

function addMessage(kind, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${kind}`;

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.append(paragraph);

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    sources.slice(0, 5).forEach((source) => {
      const item = document.createElement("div");
      item.className = "source";
      const citation = source.citation || "Source";
      const detail = source.text ? ` ${source.text.slice(0, 260)}` : "";
      item.innerHTML = `<strong></strong><span></span>`;
      item.querySelector("strong").textContent = citation;
      item.querySelector("span").textContent = detail;
      sourceList.append(item);
    });
    article.append(sourceList);
  }

  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    healthStatus.textContent = health.status === "ok" ? "Healthy" : "Degraded";
    healthStatus.className = `status ${health.status === "ok" ? "ok" : "warn"}`;
  } catch {
    healthStatus.textContent = "Offline";
    healthStatus.className = "status warn";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  const apiKey = apiKeyInput.value.trim();
  if (!query) return;

  if (!apiKey) {
    addMessage("error", "Enter the beta API key before asking a question.");
    apiKeyInput.focus();
    return;
  }

  localStorage.setItem("dharmagpt.betaKey", apiKey);
  addMessage("user", query);
  queryInput.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "Asking";

  try {
    const response = await fetch("/api/v1/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify({
        query,
        mode: modeInput.value,
        language: languageInput.value,
        history: [],
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed with HTTP ${response.status}`);
    }

    const data = await response.json();
    addMessage("assistant", data.answer, data.sources || []);
  } catch (error) {
    addMessage("error", error.message || "The beta API could not answer right now.");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Ask";
    queryInput.focus();
  }
});

checkHealth();
