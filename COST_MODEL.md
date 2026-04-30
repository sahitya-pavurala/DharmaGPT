# DharmaGPT Cost Model

This document captures the current staging-vs-production model usage and a
first-pass pay-per-use estimate for production. Prices change, so use this as a
planning model and refresh provider pricing before committing to product prices.

Currency assumption: **$1 = ₹90**.

## Current Backend Policy

### Beta / Staging

Goal: build and test corpus pipelines with minimal API spend.

| Step | Backend | Model |
| --- | --- | --- |
| Audio speech-to-text | `indicconformer` | `sulabhkatiyar/indicconformer-120m-onnx`, Telugu model |
| Telugu to English translation | `indictrans2` | `ai4bharat/indictrans2-indic-en-dist-200M` |
| Embeddings | `ollama` | `nomic-embed-text` |
| Answer generation | `ollama` | `qwen2.5:7b` |
| Vector DB | `pinecone` currently | `dharma-gpt` index |
| Validation | explicit only | Anthropic judge if manually run |

Normal beta audio/corpus creation should not call Sarvam.

### Production

Goal: hosted reliability and higher quality where it matters.

| Step | Backend | Model/API |
| --- | --- | --- |
| Audio speech-to-text | `sarvam` | Saaras v3 speech-to-text |
| Audio translation | `sarvam` if enabled | speech-to-text-translate |
| Text translation | `sarvam` or `openai` | Sarvam Translate / OpenAI |
| Embeddings | `openai` | `text-embedding-3-large` |
| Answer generation | `openai` | `gpt-4.1-mini` baseline |
| Vector DB | `pinecone` | `dharma-gpt` index |
| Validation | not in normal query flow | Anthropic Sonnet judge only for eval/admin/CI |

## Observed Sarvam Spend

On April 29, the audio batch created **221 transcript records** before quota
stopped the run. The Sarvam dashboard showed:

| Metric | Value |
| --- | ---: |
| Requests | 449 |
| Cost/credits used | ₹108.23 |
| USD equivalent | $1.20 |

Why this lines up:

- Each successful 29-second chunk used two Sarvam audio calls:
  `speech-to-text` and `speech-to-text-translate`.
- `221 chunks * 29 sec = 6,409 sec = 1.78 audio hours`.
- Sarvam published pricing is ₹30/hour for speech-to-text and ₹30/hour for
  speech-to-text-translate, so two calls are about `₹60/hour`.
- `1.78 hours * ₹60/hour = ₹106.80`, very close to the observed ₹108.23.

Working production estimate:

| Audio path | INR/hour | USD/hour |
| --- | ---: | ---: |
| STT only | ₹30 | $0.33 |
| STT + audio translate | ₹60 | $0.67 |
| STT with diarization | ₹45 | $0.50 |
| STT + translate + diarization | ₹45 | $0.50 |

For the full 282-chunk Day 01 file:

- `282 * 29 sec = 8,178 sec = 2.27 hours`
- STT only: `2.27 * ₹30 = ₹68`
- STT + audio translate: `2.27 * ₹60 = ₹136`

## Production Query Cost

Assume baseline answer generation with OpenAI `gpt-4.1-mini`.

Published OpenAI pricing used here:

- Input: **$0.40 / 1M tokens**
- Output: **$1.60 / 1M tokens**
- Embeddings `text-embedding-3-large`: **$0.13 / 1M tokens**

Rupee conversion:

- GPT input: `₹36 / 1M tokens`
- GPT output: `₹144 / 1M tokens`
- Embeddings: `₹11.70 / 1M tokens`

Example per-query costs:

| Query shape | Input tokens | Output tokens | Cost USD | Cost INR |
| --- | ---: | ---: | ---: | ---: |
| Small answer | 2,000 | 500 | $0.0016 | ₹0.14 |
| Normal RAG answer | 6,000 | 1,000 | $0.0040 | ₹0.36 |
| Long answer | 12,000 | 2,000 | $0.0080 | ₹0.72 |

Embedding a user query is tiny. A 100-token query with
`text-embedding-3-large` costs about:

- `$0.000013`
- `₹0.00117`

So query-time cost is dominated by answer generation, not embeddings.

## Corpus Build Cost

For production corpus ingestion using hosted APIs:

### Audio Corpus

| Item | Cost |
| --- | ---: |
| Sarvam STT only | ₹30/hour |
| Sarvam STT + audio translate | ₹60/hour |
| OpenAI embedding after transcript creation | usually small, about ₹11.70 / 1M tokens |

Rule of thumb:

- 10 hours audio with STT only: **₹300**
- 10 hours audio with STT + translate: **₹600**
- 100 hours audio with STT + translate: **₹6,000**

### Text Translation

Sarvam text translate is priced at **₹20 / 10,000 characters**.

Rule of thumb:

- 100k characters: **₹200**
- 1M characters: **₹2,000**
- 10M characters: **₹20,000**

For audio, prefer the audio translate endpoint when using Sarvam because it is
billed per audio hour and was much cheaper for the April 29 run than translating
all generated text separately.

## Validation / Evaluation Cost

Normal production queries should not call validators.

When evaluation is explicitly run with Anthropic Sonnet:

- Input: **$3 / 1M tokens** = `₹270 / 1M tokens`
- Output: **$15 / 1M tokens** = `₹1,350 / 1M tokens`

Example one-call judge:

| Eval shape | Input tokens | Output tokens | Cost USD | Cost INR |
| --- | ---: | ---: | ---: | ---: |
| Standard eval | 4,000 | 800 | $0.024 | ₹2.16 |
| Large eval | 8,000 | 1,200 | $0.042 | ₹3.78 |

This is fine for CI/admin sampling, but too expensive to run on every user query.

## Pay-Per-Usage Pricing Suggestions

These are pricing-product ideas, not final business pricing.

### Query Packs

If a normal RAG answer costs about `₹0.36`, then:

| Pack | Included queries | Estimated API cost | Suggested floor |
| --- | ---: | ---: | ---: |
| Trial | 100 | ₹36 | free/₹99 |
| Starter | 1,000 | ₹360 | ₹999+ |
| Pro | 10,000 | ₹3,600 | ₹7,999+ |

This leaves margin for Pinecone, server, retries, support, and failed calls.

### Corpus Upload Pricing

Audio processing can be priced separately because it has a clear per-hour cost.

| Product action | Estimated API cost | Suggested floor |
| --- | ---: | ---: |
| Audio STT only | ₹30/hour | ₹99/hour |
| Audio STT + translation | ₹60/hour | ₹199/hour |
| Premium reviewed corpus | API cost + human/admin time | custom |

For beta/staging, use local Indic models and treat cost as server compute.

## Sources

- Sarvam pricing: https://www.sarvam.ai/api-pricing
- Sarvam credits/rate limits: https://docs.sarvam.ai/api-reference-docs/getting-started/pricing
- OpenAI pricing: https://platform.openai.com/docs/pricing/
- OpenAI GPT-4.1 mini model pricing: https://platform.openai.com/docs/models/gpt-4.1-mini
- Anthropic pricing: https://www.anthropic.com/pricing
