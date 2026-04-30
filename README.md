# DharmaGPT

> AI-powered wisdom from the Valmiki Ramayana, Mahabharata, Bhagavad Gita, Upanishads, and Puranas — built from Bharat, accessible to the world.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made in India](https://img.shields.io/badge/Made%20in-India-orange)](https://github.com/dharmagpt)
[![Powered by Sarvam AI](https://img.shields.io/badge/Audio-Sarvam%20AI-blue)](https://sarvam.ai)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![CI](https://github.com/sahitya-pavurala/DharmaGPT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sahitya-pavurala/DharmaGPT/actions/workflows/ci.yml)

---

## What is DharmaGPT?

DharmaGPT is an open-source AI backend that powers natural-language access to the wisdom of Hindu sacred texts. Ask life questions, generate factual story retellings, find verse references, and process Sanskrit chantings — all grounded in real source texts with citations.

**This is not a chatbot that makes things up.** Every answer is retrieved from indexed source texts and cited to the finest provenance available — section (Kanda / Parva / Adhyaya / Skandha), chapter, and verse. The citation schema is text-agnostic: the same pipeline handles the Ramayana, Mahabharata, Upanishads, and Puranas without hardcoding any text's terminology.

---

## Features

| Feature | Description |
|---|---|
| **Life Guidance** | Ask dharmic questions, get wisdom grounded in actual verses |
| **Story Generation** | Factual retellings from source texts, chapter-accurate |
| **Children's Mode** | Age-appropriate stories with moral teachings |
| **Scholarly Lookup** | Search verses, themes, characters across all texts |
| **Audio Support** | Listen to Sanskrit chantings, search within pravachanams |
| **22 Indian Languages** | Powered by Sarvam AI's Saaras v3 for multilingual audio |
| **Pluggable Translation Backends** | Swap between API and local models such as Anthropic, Ollama, or IndicTrans2 |

---

## Tech Stack

```
Backend API     →  FastAPI (Python)
LLM             →  Claude (Anthropic) via API
Vector Search   →  Pinecone
Embeddings      →  OpenAI text-embedding-3-large
Audio STT/TTS   →  Sarvam AI (Saaras v3 + Bulbul v3)
Translation     →  Anthropic, Ollama, IndicTrans2
```

---

## Project Structure

```
dharmagpt/
├── api/routes/           # FastAPI route handlers
├── core/                 # RAG engine, retrieval, LLM
├── evaluation/           # Response quality validation pipeline
│   ├── metric_definitions.py     # MetricScore, ValidationResult data classes
│   ├── response_scorer.py        # LLM judge + rule-based scoring
│   ├── batch_runner.py           # Batch evaluation runner
│   └── sample_questions.jsonl    # 10 sample questions across all 4 modes
├── pipelines/            # Audio chunking, translation, indexing
├── models/               # Pydantic models
├── scripts/              # Integrated pipeline scripts (use dharmagpt package)
│   ├── normalize_raw_corpus.py
│   ├── translate_corpus.py
│   ├── ingest_to_pinecone.py
│   ├── transcribe_audio_batch.py
│   └── run_evaluation.py
├── tests/                # Unit tests
│   └── test_response_scorer.py
└── utils/                # Helpers, logging, canonical naming
scripts/
└── audio/                # Standalone audio tools (no server required)
    ├── audio_pipeline.py
    ├── sarvam_translate.py
    └── sarvam_translate_batch.py
data/                     # Local data (gitignored)
```

### Integrated scripts vs standalone scripts

**`dharmagpt/scripts/`** — the production data pipeline. These scripts import from the `dharmagpt` package (`core.config`, `core.translation`, `utils.naming`) and must be run from inside the `dharmagpt/` directory with a configured `.env`. They write output to `knowledge/processed/` using the [canonical file naming convention](#knowledge-file-naming) and feed data into Pinecone.

| Script | What it does |
|---|---|
| `normalize_raw_corpus.py` | Cleans scraped raw JSONL → flat corpus schema. Builds full citation strings: `"Valmiki Ramayana, Sundara Kanda, Sarga 15, Verse 22"`. Outputs `section`, `chapter`, `verse` fields. |
| `translate_corpus.py` | Batch-translates corpus records to English (Anthropic → Ollama → IndicTrans2) |
| `ingest_to_pinecone.py` | Embeds corpus records and upserts vectors to Pinecone. Stores `section`, `chapter`, `verse` metadata alongside legacy `kanda`/`sarga` keys for backward compatibility. |
| `transcribe_audio_batch.py` | Splits source audio into 29s clips and uploads each to the running API for transcription |
| `run_evaluation.py` | Evaluates RAG response quality across faithfulness, relevance, citation precision, and context use |

Run them in order for text corpus: `normalize_raw_corpus → translate_corpus → ingest_to_pinecone`. For audio: `transcribe_audio_batch` (requires the API server running). To evaluate quality: `run_evaluation`.

**`scripts/audio/`** — standalone offline tools. These call external APIs (Sarvam STT, Anthropic) directly without needing a running DharmaGPT server. Useful for quick local testing, processing audio on a machine without all credentials configured, or prototyping new language support. Output is JSONL written to local files rather than Pinecone.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Production API keys as needed: Anthropic/OpenAI, Pinecone, Sarvam AI
- Beta/staging corpus creation can run without Sarvam by using local Indic models:
  `STT_BACKEND=indicconformer` and `TRANSLATION_BACKEND=indictrans2`

### Run the API

```bash
cd dharmagpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn api.main:app --reload --port 8000
```

### Local Translation Backends

For Telugu to English translation, the audio pipeline now supports multiple backends:

- `anthropic` for Claude-based translation
- `ollama` for local models
- `indictrans2` for AI4Bharat IndicTrans2

For beta/staging corpus work, prefer:

```bash
STT_BACKEND=indicconformer
TRANSLATION_BACKEND=indictrans2
INDICTRANS2_MODEL=ai4bharat/indictrans2-indic-en-dist-200M
```

`indicconformer` uses a local ONNX export of AI4Bharat IndicConformer for Telugu ASR. `indictrans2` uses AI4Bharat IndicTrans2 for Telugu-English translation. This keeps the beta audio/corpus pipeline off Sarvam by default; Sarvam remains available as a production or fallback backend when configured.

### Knowledge File Naming

Generated knowledge files now follow a canonical stem:

`<source>[_<title>][_<author>]_<language>_<kind>_partNN.jsonl`

Examples:

- `valmiki_ramayanam_chaganti_te_audio_part01.jsonl`
- `valmiki_ramayanam_chaganti_te_transcript_part01.jsonl`
- `valmiki_ramayanam_valmiki_en_processed_part01.jsonl`

Use this pattern for processed corpus files, audio transcripts, review datasets, and future archive/database exports so the same identifier can survive migrations.

### Internal Admin UI

The repo includes a server-rendered response review console at:

- `GET /admin/feedback`

It supports the feedback review workflow used to improve answer quality over time, while keeping review data separate from live serving.

### Deployment

The recommended production setup is Docker Compose plus Nginx:

```bash
cp dharmagpt/.env.example dharmagpt/.env
# fill in the API keys and admin key
docker compose up --build -d
```

Required admin settings:

- `ADMIN_API_KEY`

The compose stack runs:

- `app` on port `8000` inside the Docker network
- `nginx` on port `80` for public access

### Query the API

```bash
# POST /api/v1/query
{
  "query": "How should I deal with anger?",
  "mode": "guidance",           # guidance | story | children | scholar
  "language": "en",             # answer language (routing in progress)
  "filter_section": "Sundara Kanda",  # optional: scope retrieval to one section
  "history": []                 # optional: last N conversation turns
}
```

Response includes `sources[]` — each with `citation`, `section`, `chapter`, `verse`, and `score`.

### Ingest Data

```bash
# Scrape and process text corpus (run from dharmagpt/)
python scripts/normalize_raw_corpus.py
python scripts/translate_corpus.py
python scripts/ingest_to_pinecone.py

# Process audio (requires API server running)
python scripts/transcribe_audio_batch.py --input-dir data/audio/ --language-code te-IN
```

### Run Tests

Two separate build steps — unit tests are always fast and offline; integration tests stay offline too, but they exercise the full pipeline end-to-end with local retrieval and a local Ollama model.

```bash
# Unit tests — no API keys required, runs in < 5s
make test-unit
# or directly:
cd dharmagpt && PYTHONPATH=. python -m pytest tests/unit/ -v

# Integration tests — no SaaS API keys required
# Tests the full path: query → local retrieve → local model generate → local model judge
make test-integration
# or directly:
cd dharmagpt && PYTHONPATH=. python -m pytest tests/integration/ -v --timeout=120

# Both steps together
make test-all
```

Integration tests are automatically **skipped** (not failed) when Ollama is unavailable, so they're safe to run locally and in CI without external API secrets.

| Suite | Location | Speed | Needs credentials |
|---|---|---|---|
| Unit | `tests/unit/` | < 5s | No |
| Integration | `tests/integration/` | 30–120s | Yes |

### Evaluate Response Quality

```bash
# Run the validation pipeline against sample questions (from dharmagpt/)
python scripts/run_evaluation.py

# Quick smoke test with 3 questions
python scripts/run_evaluation.py --limit 3

# Custom question set and output path
python scripts/run_evaluation.py \
  --questions evaluation/sample_questions.jsonl \
  --output evaluation/reports/run.jsonl
```

Scores each response across four dimensions:

| Metric | Weight | What it checks |
|---|---|---|
| Faithfulness | 35% | Claims are grounded in retrieved passages, not hallucinated |
| Answer relevance | 30% | Answer directly addresses the user's query |
| Context utilization | 20% | Answer draws from retrieved passages rather than ignoring them |
| Citation precision | 15% | Inline citations are accurate and traceable to sources |

A response **passes** when its weighted score ≥ 0.65. Results are written as JSONL to `evaluation/reports/`.

---

## Contributing

We welcome contributions — especially:
- **Sanskrit scholars** to verify citations and translations
- **Developers** to extend the text corpus (Mahabharata, Puranas)
- **Audio contributors** to add pravachanam recordings
- **Language contributors** for regional language support

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Sources & Acknowledgements




---

## License

MIT — free to use, study, and extend. Attribution appreciated.

---

*Satyam Vada. Dharmam Chara. — Speak truth. Walk the path of dharma.*
