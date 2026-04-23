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
├── pipelines/            # Data ingestion pipelines
├── models/               # Pydantic models
└── utils/                # Helpers, logging
scripts/
├── audio/                # Sarvam audio pipeline
└── embed/                # Pinecone embedding pipeline
data/                     # Local data (gitignored)
tests/                    # Tests
```

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
# 2. Process your audio files
python scripts/audio/audio_pipeline.py --input data/audio/ --batch

# 3. Embed and index everything
python scripts/embed/embed_and_index.py --input data/chunks/ --index dharma-gpt
```

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
