# DharmaGPT — Architecture

## Overview

DharmaGPT is a standalone AI engine that exposes a REST API. Other projects (mobile apps, web frontends, bots) call it — it does not contain any frontend code.

```
Caller (any client)
    │  REST (JSON)
    ▼
FastAPI  (dharmagpt/api/)
    ├── Query → RAG Engine → Pinecone → Claude → Response
    └── Audio → Sarvam STT → Chunker → Pinecone
```

## Data Flow: Query

1. Caller sends `POST /api/v1/query` with `{query, mode, history}`
2. Backend embeds query via OpenAI `text-embedding-3-large`
3. Pinecone returns top-5 most similar chunks from the corpus
4. Retrieved passages are injected into Claude's system prompt
5. Claude generates an answer grounded in those passages, with citations
6. Response returned with `answer` + `sources[]` (citation, sarga, score)

## Data Flow: Audio Ingestion

1. Caller uploads audio file (chanting, pravachanam, discourse) via `POST /api/v1/audio/transcribe`
2. Backend sends to Sarvam Saaras v3 with `with_timestamps=true`
3. Word-level timestamps used to chunk at natural pause boundaries (>0.8s)
4. Each chunk tagged: speaker type (chanting vs commentary), shloka detection, kanda
5. Chunks embedded and upserted to Pinecone
6. Audio chunks become searchable alongside text chunks

## Chunking Strategy

### Text (valmikiramayan.net)
- Primary: shloka-level (one verse = one chunk, ~20-80 words)
- Secondary: overlapping windows of 3 shlokas for context continuity
- Metadata: kanda, sarga, verse index, characters, themes, URL

### Audio (Sarvam STT output)
- Primary: pause-boundary chunking (gap > 0.8s = natural chunk break)
- Fallback: sentence-boundary via danda (।) detection
- Metadata: start/end timestamps, speaker type, has_shloka flag

## Pinecone Index Schema

```
Vector: float[3072]  (text-embedding-3-large)
Metadata:
  source_type:    "text" | "audio"
  kanda:          "Sundara Kanda" | ...
  sarga:          int
  citation:       "Valmiki Ramayana, Sundara Kanda, Sarga 15, ~Verse 3"
  url:            "https://www.valmikiramayan.net/..."
  characters:     ["Hanuman", "Sita"]
  themes:         ["devotion", "courage"]
  word_count:     int
  text_preview:   first 300 chars (for display)
  # Audio-only:
  start_time_sec: float
  end_time_sec:   float
  speaker_type:   "chanting" | "commentary_hindi" | "commentary_english"
  has_shloka:     bool
```

## RAG Modes

| Mode | Retrieval | Prompt | Response |
|------|-----------|--------|----------|
| `guidance` | General semantic | Wise elder, apply wisdom | 150-250w + reflection |
| `story` | Episode-focused | Literary narrator | 200-350w + citation |
| `children` | Narrative chunks | Simple, warm | 150-200w + moral |
| `scholar` | High-precision | Academic, structured | Headers + IAST citations |

## Sarvam AI Integration

- **STT**: Saaras v3 — 22 Indian languages, code-mixed, word timestamps, diarization
- **TTS**: Bulbul v3 — 11 languages, 35+ voices (future: read answers aloud)
- **Model**: `saaras:v3` for transcription, `bulbul:v3` for synthesis

## Scaling Considerations

- Pinecone serverless scales automatically; no ops needed
- FastAPI workers can be added behind a load balancer (Railway, Render, AWS)
- Audio transcription is the bottleneck — queue long files with Celery + Redis
