PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
PYTHONPATH_SET := PYTHONPATH=.
PYTEST_INTEGRATION_ARGS ?=
AUDIO_INPUT_DIR ?= knowledge/uploads/audio_sources
AUDIO_OUTPUT_DIR ?= downloads/clips_29s_full
AUDIO_LANGUAGE_CODE ?= hi-IN
AUDIO_LANGUAGE_TAG ?= hi
AUDIO_API_URL ?= http://127.0.0.1:8000/api/v1/audio/transcribe

# ─── Test build steps ─────────────────────────────────────────────────────────

## Run fast offline unit tests (no API keys required)
.PHONY: test-unit
test-unit:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTEST) tests/unit/ -v

## Run full end-to-end integration tests (requires local Ollama)
.PHONY: test-integration
test-integration:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTEST) tests/integration/ -v $(PYTEST_INTEGRATION_ARGS)

## Run both unit and integration tests
.PHONY: test-all
test-all: test-unit test-integration

# ─── Evaluation ───────────────────────────────────────────────────────────────

## Score RAG response quality against sample questions
.PHONY: evaluate
evaluate:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTHON) scripts/run_evaluation.py

## Quick 3-question smoke test of the evaluation pipeline
.PHONY: evaluate-smoke
evaluate-smoke:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTHON) scripts/run_evaluation.py --limit 3

# ─── Data pipeline ────────────────────────────────────────────────────────────

## Normalize raw scraped corpus → flat schema
.PHONY: normalize
normalize:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTHON) scripts/normalize_raw_corpus.py

## Translate corpus records to English
.PHONY: translate
translate:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTHON) scripts/translate_corpus.py

## Embed and upsert corpus to Pinecone
.PHONY: ingest
ingest:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTHON) scripts/ingest_to_pinecone.py

## Full data pipeline: normalize → translate → ingest
.PHONY: pipeline
pipeline: normalize translate ingest

## Transcribe audio files and upsert vector chunks via API
.PHONY: audio-vectorize
audio-vectorize:
	cd dharmagpt && $(PYTHONPATH_SET) $(PYTHON) scripts/transcribe_audio_batch.py \
		--input-dir $(AUDIO_INPUT_DIR) \
		--output-dir $(AUDIO_OUTPUT_DIR) \
		--language-code $(AUDIO_LANGUAGE_CODE) \
		--language-tag $(AUDIO_LANGUAGE_TAG) \
		--api-url $(AUDIO_API_URL) \
		--recursive

# ─── Dev server ───────────────────────────────────────────────────────────────

## Start the FastAPI dev server
.PHONY: serve
serve:
	cd dharmagpt && $(PYTHONPATH_SET) uvicorn api.main:app --reload --port 8000

# ─── Beta Docker server ───────────────────────────────────────────────────────

## Start beta Docker stack
.PHONY: beta-up
beta-up:
	./scripts/docker_beta_up.sh

## Start/rebuild beta Docker stack
.PHONY: beta-build
beta-build:
	./scripts/docker_beta_up.sh --build

## Start beta Docker stack with Cloudflare tunnel
.PHONY: beta-tunnel
beta-tunnel:
	./scripts/docker_beta_up.sh --tunnel

## Stop beta Docker stack
.PHONY: beta-down
beta-down:
	./scripts/docker_beta_down.sh

## Back up beta Docker Postgres and knowledge files
.PHONY: beta-backup
beta-backup:
	./scripts/docker_beta_backup.sh

.PHONY: help
help:
	@grep -E '^##' Makefile | sed 's/## //'
