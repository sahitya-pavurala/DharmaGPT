# DharmaGPT

> AI-powered wisdom from the Valmiki Ramayana, Mahabharata, Bhagavad Gita, Upanishads, and Puranas — built from Bharat, accessible to the world.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made in India](https://img.shields.io/badge/Made%20in-India-orange)](https://github.com/dharmagpt)
[![Powered by Sarvam AI](https://img.shields.io/badge/Audio-Sarvam%20AI-blue)](https://sarvam.ai)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## What is DharmaGPT?

DharmaGPT is an open-source AI backend that powers natural-language access to the wisdom of Hindu sacred texts. Ask life questions, generate factual story retellings, find verse references, and process Sanskrit chantings — all grounded in real source texts with citations.

**This is not a chatbot that makes things up.** Every answer is retrieved from indexed source texts (Valmiki Ramayana, Mahabharata, Bhagavad Gita, Upanishads, Puranas) and cited by kanda, parva, chapter, and verse.

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
| `normalize_raw_corpus.py` | Cleans scraped raw JSONL → flat corpus schema |
| `translate_corpus.py` | Batch-translates corpus records to English (Anthropic → Ollama → IndicTrans2) |
| `ingest_to_pinecone.py` | Embeds corpus records and upserts vectors to Pinecone |
| `transcribe_audio_batch.py` | Splits source audio into 29s clips and uploads each to the running API for transcription |
| `run_evaluation.py` | Evaluates RAG response quality across faithfulness, relevance, citation precision, and context use |

Run them in order for text corpus: `normalize_raw_corpus → translate_corpus → ingest_to_pinecone`. For audio: `transcribe_audio_batch` (requires the API server running). To evaluate quality: `run_evaluation`.

**`scripts/audio/`** — standalone offline tools. These call external APIs (Sarvam STT, Anthropic) directly without needing a running DharmaGPT server. Useful for quick local testing, processing audio on a machine without all credentials configured, or prototyping new language support. Output is JSONL written to local files rather than Pinecone.

---

## Getting Started

### Prerequisites
- Python 3.11+
- API keys: Anthropic, Pinecone, OpenAI (embeddings), Sarvam AI

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

The official AI4Bharat repository is cloned under `downloads/IndicTrans2`. Their top-level `install.sh` is Linux/conda-oriented, so on Windows we install the Python dependencies directly in the local runtime and use the Hugging Face model path through the shared translation layer.

### Manual Translation Review API

The manual translation endpoints now use dataset IDs instead of raw file paths and support an API key gate.

- `POST /api/v1/audio/manual-translations/chunk`
- `POST /api/v1/audio/manual-translations/bulk`
- `POST /api/v1/audio/manual-translations/review`
- `GET /api/v1/audio/manual-translations/datasets/{dataset_id}/pending`

Set `MANUAL_TRANSLATION_API_KEY` in `.env` before exposing the API to employees. Keep approved datasets under `MANUAL_TRANSLATION_DATASET_ROOT` and list any explicit allowlist in `MANUAL_TRANSLATION_ALLOWED_DATASETS`.

### Knowledge File Naming

Generated knowledge files now follow a canonical stem:

`<source>[_<title>][_<author>]_<language>_<kind>_partNN.jsonl`

Examples:

- `valmiki_ramayanam_chaganti_te_audio_part01.jsonl`
- `valmiki_ramayanam_chaganti_te_transcript_part01.jsonl`
- `valmiki_ramayanam_valmiki_en_processed_part01.jsonl`

Use this pattern for processed corpus files, audio transcripts, manual-review datasets, and future archive/database exports so the same identifier can survive migrations.

### Internal Admin UI

The repo also includes a simple server-rendered review console at:

- `GET /admin/manual-translations`

It reads and updates the same manual translation API used by the backend. The page stores the `X-API-Key` locally in the browser and lets reviewers:

- load a dataset by ID
- edit `text_en_manual`
- save draft changes
- mark chunks `approved`, `needs_work`, or `rejected`

### Team Server Deployment

For an internal team server, the recommended setup is Docker Compose plus Nginx:

```bash
cp dharmagpt/.env.example dharmagpt/.env
# fill in the API keys and manual translation settings
docker compose up --build -d
```

Required manual translation settings:

- `MANUAL_TRANSLATION_API_KEY`
- `MANUAL_TRANSLATION_DATASET_ROOT=knowledge/processed`
- `MANUAL_TRANSLATION_AUDIT_LOG=knowledge/audit/manual_translation_audit.jsonl`
- `MANUAL_TRANSLATION_ALLOWED_DATASETS=dataset_one,dataset_two`

The compose stack runs:

- `app` on port `8000` inside the Docker network
- `nginx` on port `80` for public access

Place your editable JSONL files under `dharmagpt/knowledge/processed/` and mount that directory so translations persist across restarts. If your team should only access the tool internally, put the server behind VPN, a private subnet, or an identity-aware proxy.

### Ingest Data

```bash
# Scrape and process text corpus (run from dharmagpt/)
python scripts/normalize_raw_corpus.py
python scripts/translate_corpus.py
python scripts/ingest_to_pinecone.py

# Process audio (requires API server running)
python scripts/transcribe_audio_batch.py --input-dir data/audio/ --language-code te-IN
```

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

- **Valmiki Ramayana** — [valmikiramayan.net](https://www.valmikiramayan.net/) (K. M. K. Murthy translation)


---

## License

MIT — free to use, study, and extend. Attribution appreciated.

---

*Satyam Vada. Dharmam Chara. — Speak truth. Walk the path of dharma.*
