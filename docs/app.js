// Global dataset reference
let dashboardData = null;

// Map raw book names to beautiful generic Dharmic Section names
const sectionMapping = {
  "Bala Kanda": "Section I: Foundations",
  "Ayodhya Kanda": "Section II: Ethics & Duties",
  "Aranya Kanda": "Section III: Resilience",
  "Kishkindha Kanda": "Section IV: Trust & Alliances",
  "Sundara Kanda": "Section V: Devotion & Courage",
  "Yuddha Kanda": "Section VI: Crisis Leadership",
  "Unknown": "General Wisdom"
};

// Initialize Dashboard on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  fetchDashboardMetrics();
});

// Fetch JSON metrics from local static file
async function fetchDashboardMetrics() {
  try {
    const response = await fetch("data.json");
    if (!response.ok) throw new Error("Failed to load metrics JSON data");
    
    dashboardData = await response.json();
    populateDashboard(dashboardData);
  } catch (error) {
    console.error("Dashboard initialization error:", error);
    document.getElementById("playground-results").innerHTML = `
      <div class="loading text-red" style="color: #f87171;">
        <i class="fa-solid fa-triangle-exclamation"></i> Error loading database metrics: ${error.message}
      </div>
    `;
  }
}

// Populate UI Elements with loaded JSON data
function populateDashboard(data) {
  // Update generated time
  document.getElementById("gen-time").textContent = data.generated_at || "N/A";
  
  // Update total chunks counter
  const totalChunks = data.stats.total_chunks;
  document.getElementById("stat-chunks").textContent = Number(totalChunks).toLocaleString();

  // Render Kanda Distribution Chart (Chart.js)
  renderKandaChart(data.stats.kanda_distribution);
}

// Render the interactive bar chart using Chart.js
function renderKandaChart(distribution) {
  const ctx = document.getElementById("kandaChart").getContext("2d");
  
  const labels = Object.keys(distribution).map(k => sectionMapping[k] || k);
  const values = Object.values(distribution);

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Ingested Chunks",
        data: values,
        backgroundColor: [
          "rgba(99, 102, 241, 0.4)", // Indigo
          "rgba(139, 92, 246, 0.4)", // Violet
          "rgba(16, 185, 129, 0.4)", // Emerald
          "rgba(245, 158, 11, 0.4)",  // Amber
          "rgba(239, 68, 68, 0.4)",   // Red-rose
          "rgba(6, 182, 212, 0.4)"    // Cyan
        ],
        borderColor: [
          "#6366f1",
          "#8b5cf6",
          "#10b981",
          "#f59e0b",
          "#ef4444",
          "#06b6d4"
        ],
        borderWidth: 1.5,
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: "#111827",
          titleFont: { family: "'Outfit', sans-serif", size: 14 },
          bodyFont: { family: "'Inter', sans-serif", size: 13 },
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1
        }
      },
      scales: {
        y: {
          grid: {
            color: "rgba(255, 255, 255, 0.05)",
            drawBorder: false
          },
          ticks: {
            color: "#9ca3af",
            font: { family: "'Inter', sans-serif" }
          }
        },
        x: {
          grid: {
            display: false
          },
          ticks: {
            color: "#9ca3af",
            font: { family: "'Outfit', sans-serif", size: 12 }
          }
        }
      }
    }
  });
}

// Set query in input box and execute search
function setPlaygroundQuery(query) {
  const queryInput = document.getElementById("playground-query");
  queryInput.value = query;
  executePlaygroundSearch();
}

// Execute the semantic playground mock search
function executePlaygroundSearch() {
  const queryInput = document.getElementById("playground-query");
  const query = queryInput.value.trim();
  
  if (!query) return;

  const animationBox = document.getElementById("search-animation-box");
  const step1 = document.getElementById("sim-step-1");
  const step2 = document.getElementById("sim-step-2");
  const resultsContainer = document.getElementById("playground-results");

  // Show simulation steps
  resultsContainer.innerHTML = "";
  animationBox.classList.remove("hidden");
  step1.classList.remove("done");
  step2.classList.add("hidden");
  step2.classList.remove("done");

  // Step 1: Simulated Vectorization (800ms)
  setTimeout(() => {
    step1.classList.add("done");
    step1.querySelector(".sim-icon").innerHTML = '<i class="fa-solid fa-circle-check"></i>';
    step2.classList.remove("hidden");

    // Step 2: Simulated PostgreSQL Similarity Query (900ms)
    setTimeout(() => {
      step2.classList.add("done");
      step2.querySelector(".sim-icon").innerHTML = '<i class="fa-solid fa-circle-check"></i>';

      // Step 3: Render results
      setTimeout(() => {
        animationBox.classList.add("hidden");
        // Reset icons for next search
        step1.querySelector(".sim-icon").innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        step2.querySelector(".sim-icon").innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        
        renderMockResults(query, resultsContainer);
      }, 300);

    }, 900);

  }, 800);
}

// Map keywords to static samples and render them
function renderMockResults(query, container) {
  if (!dashboardData) {
    container.innerHTML = `<div class="loading">Dataset not loaded.</div>`;
    return;
  }

  const queryLower = query.toLowerCase();
  let matches = [];
  let generativeAnswer = "";

  // Match based on keywords
  if (queryLower.includes("anger") || queryLower.includes("greed") || queryLower.includes("abandon") || queryLower.includes("manage")) {
    generativeAnswer = "Handling overwhelming emotions like anger and greed requires active detachment. According to Sanatana Dharma, these emotions are destructive obstacles that cloud our judgment and inevitably bind us to constant misery (nityam duhkham). In your situation, practice patient self-reflection and turn your focus toward duty (dharma) rather than immediate desires. Peaceful compromise with those around you, as counselled in our classical guides, will strengthen your state of mind and protect your inner tranquility.";
    matches = [
      {
        section: "Section II: Ethics & Duties",
        citation: "Dharma Code - 2.28.24",
        score: 0.8924,
        text: "krodha lobhau = anger; greed; vimoktavyau = are to be abandoned; devotion is to be done on asceticism; what needs to be feared should not be feared, living in the forest is nityam duhkham (constant misery)."
      },
      {
        section: "Section VI: Crisis Leadership",
        citation: "Dharma Code - 6.35.8",
        score: 0.6124,
        text: "He who concludes peace even with enemies or wages war at a fitting time strengthens his own party, abandons avarice and anger, and attains a great power."
      }
    ];
  } else if (queryLower.includes("forest") || queryLower.includes("rama") || queryLower.includes("setback") || queryLower.includes("transition") || queryLower.includes("overcome")) {
    generativeAnswer = "Walking a path of duty often requires intense sacrifice, much like the classic path of exile. When faced with difficult life transitions, remember that devotion and resilience are your greatest virtues. The classic unwavering resolve of deep devotion exemplifies that true companionship and commitment stand firm even in the most pathless, challenging situations. Trust in your path, protect those who rely on you, and maintain your integrity in the face of adversity.";
    matches = [
      {
        section: "Section II: Ethics & Duties",
        citation: "Dharma Code - 2.118.3",
        score: 0.8247,
        text: "Sita did not accept his word, for her mind was set to follow Rama to the forest. She said: 'Oh Rama, a husband is a respectable person for a woman. I shall follow you wherever you go, even to the deep woods.'"
      },
      {
        section: "Section III: Resilience",
        citation: "Dharma Code - 3.1.12",
        score: 0.7491,
        text: "The hermits request Rama to protect them from demons in the forest. Rama obligingly starts this unilateral protection because it is the duty of a king to shield sages from distress."
      }
    ];
  } else if (queryLower.includes("devotion") || queryLower.includes("sita") || queryLower.includes("follow") || queryLower.includes("isolation") || queryLower.includes("strength")) {
    generativeAnswer = "Devotion is not passive submission, but an active, profound force of character that provides emotional stability during times of extreme sorrow. As demonstrated by unfaltering devotion during times of severe restriction, keeping your mind set on what is pure and noble will shield you from the threats and anxieties of your immediate environment. Draw strength from your inner commitments, hold onto hope, and let your devotion guide you through the darkness.";
    matches = [
      {
        section: "Section II: Ethics & Duties",
        citation: "Dharma Code - 2.118.4",
        score: 0.8654,
        text: "Sita manifests the tenderness of a mother and a father to Rama. She says: 'I know well that a husband is a respectable person. Even if he goes to a pathless forest, I shall walk before him.'"
      },
      {
        section: "Section V: Devotion & Courage",
        citation: "Dharma Code - 5.15.22",
        score: 0.7812,
        text: "Hanuman beholds Sita in the Ashoka grove, thin, distressed, but unfaltering in her devotion to Rama, reciting his name constantly and ignoring all threats from Ravana."
      }
    ];
  } else {
    generativeAnswer = "Welcome to DharmaCompass. When facing challenging life transitions, turn to the eternal wisdom of Dharma to restore emotional balance. By examining classical lessons of integrity, patient service, and compassionate action from scripture, you can navigate modern anxiety and organizational conflict with clear, tranquil leadership.";
    
    // Mixed default samples
    const bala = dashboardData.samples["Bala Kanda"] || [];
    const sundara = dashboardData.samples["Sundara Kanda"] || [];
    const combined = [...bala, ...sundara];
    
    // Pick first 2 default samples
    matches = combined.slice(0, 2).map((s, idx) => ({
      section: s.section ? (sectionMapping[s.section] || s.section) : "Sanatana Dharma Section",
      citation: s.citation ? s.citation.replace("Valmiki Ramayana", "Dharma Code") : "Dharma Code",
      score: 0.5476 - (idx * 0.05),
      text: s.text
    }));
  }

  // 1. Build Generative Answer Card
  let htmlContent = `
    <div class="generative-response-card">
      <div class="generative-header">
        <i class="fa-solid fa-sparkles"></i> DharmaCompass Synthesized Guidance
      </div>
      <div class="generative-text">${generativeAnswer}</div>
    </div>
  `;

  // 2. Build Context Database Divider
  htmlContent += `
    <div class="context-divider">
      <span><i class="fa-solid fa-database"></i> Retracted Context Chunks (Vector Store Similarity)</span>
    </div>
    <div class="samples-explorer">
  `;

  // 3. Render retrieved scripture context chunks
  matches.forEach(m => {
    htmlContent += `
      <div class="sample-card">
        <div class="sample-meta">
          <span><i class="fa-solid fa-book-open"></i> ${m.section}</span>
          <div>
            <span class="sample-citation">${m.citation}</span>
            <span class="score-badge">Similarity: ${(m.score).toFixed(4)}</span>
          </div>
        </div>
        <p class="sample-text">"${m.text}"</p>
      </div>
    `;
  });

  htmlContent += `</div>`; // Close samples-explorer div

  container.innerHTML = htmlContent;
}

// Switch main tabs (Search Playground vs Pipeline Metrics)
function switchMainTab(tabName) {
  // Update main tab button states
  const buttons = document.querySelectorAll(".main-tab-btn");
  buttons.forEach(btn => {
    if (btn.getAttribute("onclick").includes(tabName)) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  const playgroundTab = document.getElementById("playground-tab-content");
  const metricsTab = document.getElementById("metrics-tab-content");

  if (tabName === "playground") {
    playgroundTab.classList.remove("hidden");
    metricsTab.classList.add("hidden");
  } else {
    playgroundTab.classList.add("hidden");
    metricsTab.classList.remove("hidden");
  }
}
