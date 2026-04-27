PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
PYTHONPATH_SET := PYTHONPATH=.
PYTEST_INTEGRATION_ARGS ?=

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

# ─── Dev server ───────────────────────────────────────────────────────────────

## Start the FastAPI dev server
.PHONY: serve
serve:
	cd dharmagpt && $(PYTHONPATH_SET) uvicorn api.main:app --reload --port 8000

.PHONY: help
help:
	@grep -E '^##' Makefile | sed 's/## //'
