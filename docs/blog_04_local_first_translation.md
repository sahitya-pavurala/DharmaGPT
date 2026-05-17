# Local-First Translation for Indic Languages: A Three-Backend Fallback Chain

> How to build a translation pipeline that works without cloud API credits — and degrades gracefully when backends fail.

**Suggested tags:** AI, NLP, Indic Languages, LLM, Ollama, IndicTrans2, Open Source, Translation

---

Building a corpus pipeline for Indic languages has a practical constraint that most ML tutorials ignore: you cannot depend on cloud API availability or billing limits when doing iterative corpus work. A translation job that fails halfway through a 200-file batch because you hit a rate limit — or because you are building offline — is a pipeline that cannot be trusted.

DharmaGPT's translation layer solves this with a three-backend fallback chain: try Anthropic first for quality, fall back to a local Ollama model if that fails, fall back to a local IndicTrans2 model if Ollama fails too.

---

## The Three Backends

### 1. Anthropic (`claude-sonnet-4-20250514`)
Highest quality for Telugu → English translation of dharmic content. Handles Sanskrit names, honorifics, and context-dependent meaning well. Requires `ANTHROPIC_API_KEY` and internet.

### 2. Ollama (`qwen2.5:7b`)
Local inference, no API key, works fully offline. Qwen 2.5 is a strong multilingual model with solid Telugu comprehension. Quality is below Anthropic but acceptable for corpus building.

### 3. IndicTrans2 (`ai4bharat/indictrans2-indic-en-dist-200M`)
Fully offline, specialized for Indic languages. The distilled 200M parameter model runs on CPU. Translation is more literal than Qwen but highly reliable at the sentence level.

### skip
Explicit no-op. Useful for processing runs where you only want to normalize and ingest without translating.

---

## Auto Mode: Local-First by Default

In `auto` mode, the pipeline tries backends in priority order and stops at the first success. Every outcome — success or failure — is recorded:

```json
{
  "translation_backend": "ollama",
  "translation_version": "qwen2.5:7b",
  "translation_fallback_reason": "failed:anthropic",
  "translation_attempted_backends": ["anthropic", "ollama"]
}
```

This is real output from processing 189 Telugu transcript records. 30 records were translated by Anthropic. 159 fell back to Ollama. The fallback is not a failure state — it is a first-class outcome with a full audit trail.

---

## Lazy Imports for Optional Dependencies

IndicTrans2 requires `torch`, `transformers`, `sentencepiece`, and `indicnlp` — a heavy dependency set that takes seconds to import. Loading them at module import time means every script that touches the translation module pays this cost, even scripts that never use IndicTrans2.

Fix: imports move inside the function that uses them:

```python
def _translate_with_indictrans2(text, config, source_lang, target_lang):
    import torch  # only loaded when this backend is actually called
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    ...
```

`from core.translation import translate_text` is now fast regardless of what is installed.

---

## Model Caching for IndicTrans2

Loading a 200M parameter model takes a few seconds. For a 200-record batch using IndicTrans2 as fallback, loading it 200 times would be unusably slow. The model is cached after the first load:

```python
@lru_cache(maxsize=4)
def _load_indictrans2_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device), device
```

A repeated translation run within the same Python process reuses the already-loaded model.

---

## Both Field Names Written for Compatibility

New translations go into `text_en_model`. Earlier records used `text_en`. Rather than break compatibility, both are written:

```python
record["text_en_model"] = outcome.text
record["text_en"] = outcome.text   # compatibility alias
```

Downstream code — ingestion, retrieval, manual review — reads whichever field it knows about. The ingestion script handles both explicitly:

```python
text_en = record.get("text_en") or record.get("text_en_model") or ""
```

---

## Human Translation as a Separate Field

`text_en_manual` is separate from `text_en_model`. Manual corrections do not overwrite model output. This enables:
- Quality measurement by comparing model vs. human output over time
- Audit trail with reviewer ID, timestamp, and review note
- Model-human translation pairs as future supervised fine-tuning data

---

## Recovering Silent Failures

The `translation_fallback_reason` field makes silent failures findable. After a batch run that produced 71 records with empty `text_en`, the cause was immediately visible:

```
translation_fallback_reason: "failed:ollama:No module named 'torch'"
```

PyTorch was not installed in that environment. Recovery was a targeted re-run using the existing `process_file()` function — 71 files recovered, 0 re-translations of already-complete records.

---

## Windows-Specific Fixes

Two encoding problems appeared during Windows development:

**Batch subprocess output** failed with `UnicodeEncodeError: 'charmap' codec` on Telugu characters in stdout. Fixed by setting `PYTHONIOENCODING=utf-8` for child processes.

**File I/O** consistently uses `encoding="utf-8"`. Python 3 defaults to the system locale on Windows (cp1252 on most machines) — incompatible with Telugu Unicode.

---

## Running the Translation Pipeline

```bash
# Auto mode: tries Anthropic → Ollama → IndicTrans2
cd dharmagpt && PYTHONPATH=. python scripts/translate_corpus.py

# Specific file
python scripts/translate_corpus.py \
  --file audio_transcript/01_01_sampoorna_ramayanam.../part18.jsonl

# Force re-translation
python scripts/translate_corpus.py --force

# Local-only (no Anthropic key needed)
# Ollama must be running with qwen2.5:7b pulled
python scripts/translate_corpus.py  # auto skips Anthropic, uses Ollama
```

---

## Closing

A translation pipeline for a dharmic AI corpus cannot depend on a single cloud API being available. The three-backend fallback chain means the pipeline can run in any environment and produces auditable, recoverable output regardless of which backends succeed. The audit fields turn what would be silent failures into inspectable state — which makes large-batch corpus work practical.

*Code is open source at [github.com/ShambaviLabs/DharmaGPT](https://github.com/ShambaviLabs/DharmaGPT)*
