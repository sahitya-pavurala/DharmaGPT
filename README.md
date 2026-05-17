# DharmaGPT

[![Powered by Sarvam AI](https://img.shields.io/badge/Audio-Sarvam%20AI-blue)](https://sarvam.ai)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![CI](https://github.com/ShambaviLabs/DharmaGPT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ShambaviLabs/DharmaGPT/actions/workflows/ci.yml)

AI-powered access to Hindu sacred texts, built around cited answers from source material.

DharmaGPT is a FastAPI backend for asking questions, generating grounded retellings, looking up passages, and processing devotional audio. It uses retrieval-augmented generation so responses can point back to citations instead of relying on unsupported model output.

## Features

- Question answering with source citations
- Story and children's modes
- Scholarly lookup across indexed texts
- Audio transcription and translation support
- Evaluation tools for response quality

## Stack

- FastAPI
- Anthropic Claude
- LangChain provider adapters
- pgvector
- OpenAI embeddings
- Sarvam AI audio
- Optional local translation backends

## Setup

```bash
cd dharmagpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the required API keys in `.env`, then start the API:

```bash
uvicorn api.main:app --reload --port 8000
```

From the repo root, you can also run:

```bash
make serve
```

## API

```http
POST /api/v1/query
```

```json
{
  "query": "How should I deal with anger?",
  "mode": "guidance",
  "language": "en",
  "history": []
}
```

Responses include cited `sources` with citation metadata.

## Common Commands

```bash
make test-unit
make test-integration
make evaluate-smoke
make pipeline
```

## Docs

- [Architecture](ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Audio translation pipeline](docs/blog_01_audio_translation_pipeline.md)
- [RAG validation pipeline](docs/blog_02_rag_validation_pipeline.md)
- [Offline-first testing](docs/blog_03_offline_first_testing.md)
- [Local-first translation](docs/blog_04_local_first_translation.md)

## License

ShambaviLabs - free to use for all dharmic purposes.
