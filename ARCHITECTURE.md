# DharmaGPT — Architecture

## Overview

DharmaGPT is a standalone AI engine that exposes a REST API. Other projects (mobile apps, web frontends, bots) call it — it does not contain any frontend code.

```
Caller (any client)
    │  REST (JSON)
    ▼
FastAPI  (dharmagpt/api/)
    ├── Query → RAG Engine → pgvector → Claude → Response
    └── Audio → Sarvam STT → Chunker → pgvector
```

## Data Flow: Query

1. Caller sends `POST /api/v1/query` with `{query, mode, history, language, filter_section}`
2. Backend embeds query via OpenAI `text-embedding-3-large`
3. pgvector returns top-5 most similar chunks; filtered by `section` if `filter_section` is set
4. Each chunk's passage header is enriched with `Ch. N, V. N` when not already in the citation string
5. Passages are injected into Claude's system prompt with citation rules: cite to the finest level shown, never fabricate
6. Claude generates an answer grounded in those passages, with inline verse-level citations
7. Response returned with `answer` + `sources[]` (citation, section, chapter, verse, score)

LangChain is used only as a provider adapter at the chat-model boundary. The
retrieval flow, context formatting, citation enrichment, prompts, and API
response shape are owned by DharmaGPT code so grounding behavior remains easy
to inspect and test.

## Data Flow: Audio Ingestion

1. Caller uploads audio file (chanting, pravachanam, discourse) via `POST /api/v1/audio/transcribe` with optional `section` metadata
2. Backend sends to Sarvam Saaras v3 with `with_timestamps=true`
3. Word-level timestamps used to chunk at natural pause boundaries (>0.8s)
4. Each chunk tagged: speaker type (chanting vs commentary), shloka detection, section
5. Chunks embedded and upserted to Pinecone
6. Audio chunks become searchable alongside text chunks

## Chunking Strategy

### Text (valmikiramayan.net)
- Primary: shloka-level (one verse = one chunk, ~20-80 words)
- Secondary: overlapping windows of 3 shlokas for context continuity
- Metadata: section (Kanda/Parva/etc.), chapter, verse, characters, themes, URL

### Audio (Sarvam STT output)
- Primary: pause-boundary chunking (gap > 0.8s = natural chunk break)
- Fallback: sentence-boundary via danda (।) detection
- Metadata: start/end timestamps, speaker type, has_shloka flag

## Vector Schema (pgvector)

```
Vector: float[3072]  (text-embedding-3-large)
Metadata:
  source_type:    "text" | "audio"
  section:        "Sundara Kanda" | "Adi Parva" | ...   ← text-agnostic
  chapter:        int                                    ← Sarga / Adhyaya / Chapter
  verse:          int                                    ← shloka / verse number
  kanda:          str   (= section, kept for backward compat)
  sarga:          int   (= chapter, kept for backward compat)
  citation:       "Valmiki Ramayana, Sundara Kanda, Sarga 15, Verse 22"
  url:            "https://www.valmikiramayan.net/..."
  characters:     ["Hanuman", "Sita"]
  tags:           ["devotion", "courage"]
  is_shloka:      bool
  text:           full chunk text
  text_preview:   first 500 chars
  text_en:        English translation (if available)
  has_english:    bool
  dataset_id:     str
  # Audio-only:
  start_time_sec: float
  end_time_sec:   float
  speaker_type:   "chanting" | "commentary_hindi" | "commentary_english"
  has_shloka:     bool
```

Citation strings are built at normalization time and include all available levels. At query time, the retrieval layer additionally enriches any sparse citation with `Ch. N, V. N` if the chapter or verse fields are populated but not yet in the citation string.

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

- pgvector runs on PostgreSQL; scale vertically or add read replicas as query volume grows
- FastAPI workers can be added behind a load balancer (Railway, Render, AWS)
- Audio transcription is the bottleneck — queue long files with Celery + Redis
