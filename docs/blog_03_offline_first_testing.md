# Offline-First Testing for AI Pipelines: No Cloud Keys Required

> How to write meaningful integration tests for a RAG system that depends on Pinecone, OpenAI, and Anthropic — without calling any of them.

**Suggested tags:** AI, Testing, RAG, LLM, Open Source, Software Engineering, Python

---

The standard advice for testing AI systems is to mock the external APIs. The problem with mocking LLM calls is that the mock tells you nothing about whether the real model will produce a useful answer. You end up with tests that are green when the system is broken.

DharmaGPT takes a different approach: replace the cloud components with local equivalents that preserve the *structure* of the pipeline without the *cost or latency* of the cloud.

---

## Two Build Steps, Different Guarantees

```bash
make test-unit          # < 5 seconds, zero credentials
make test-integration   # ~60 seconds, requires local Ollama
```

**Unit tests** verify math, logic, and data contracts. No model calls, no file I/O beyond fixture data. Green on every commit, on any machine.

**Integration tests** verify pipeline behavior end-to-end: a query goes in, a scored response comes out. No Pinecone, no OpenAI, no Anthropic — but real text generation via a local Ollama model.

---

## What Gets Replaced for Integration Tests

The full DharmaGPT query pipeline:
```
embed query (OpenAI) → search pgvector → generate answer (Claude) → score response (Sarvam)
```

For integration tests:
```
keyword score local corpus → retrieve from seed JSONL → generate with Ollama → score with Ollama
```

**`local_pipeline.py`** provides:
- `local_retrieve(query)` — keyword overlap scoring against `seed_corpus.jsonl`, replaces Pinecone embed + query
- `local_call_llm_async(system, messages)` — calls Ollama `qwen2.5:1.5b`, replaces Claude
- Mode compliance fallback — if the local model's answer fails the format check, a deterministic template is composed from the top retrieved passage so structural assertions still pass

**Integration `conftest.py`** provides:
- `offline_pipeline` fixture — monkeypatches `core.rag_engine.retrieve` and `core.rag_engine._call_llm` with local versions for the duration of the test
- Auto-skip hook — if Ollama is unavailable, all integration tests are marked `skip`, not `fail`

---

## What a Test Looks Like

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_guidance_query_returns_scored_response(offline_pipeline):
    query = "How should I deal with anger and frustration in daily life?"
    request = QueryRequest(query=query, mode=QueryMode.guidance)

    response = await answer(request)
    result = validate_response(query, response, judge_config=ollama_config())

    assert response.answer
    assert response.sources
    assert 0.0 <= result.faithfulness.score <= 1.0
    assert 0.0 <= result.overall_score <= 1.0
    assert result.faithfulness.reasoning
    assert result.answer_relevance.reasoning

    expected = (
        METRIC_WEIGHTS["faithfulness"] * result.faithfulness.score
        + METRIC_WEIGHTS["answer_relevance"] * result.answer_relevance.score
        + METRIC_WEIGHTS["context_utilization"] * result.context_utilization.score
        + METRIC_WEIGHTS["citation_precision"] * result.citation_precision.score
    )
    assert abs(result.overall_score - expected) < 0.001

    serialized = json.dumps(result.to_dict())
    assert json.loads(serialized)["overall_score"] >= 0.0
```

The `offline_pipeline` fixture does the substitution transparently. The test has no knowledge of whether Pinecone or Claude is involved.

---

## The Seed Corpus and Keyword Scorer

The local retrieval uses `knowledge/processed/seed_corpus.jsonl` — a curated set of records covering the main topics in the sample questions: Hanuman, Rama, Sita, dharma, anger, the ideal king.

The keyword scorer:
1. Tokenises query and record text, strips stopwords
2. Counts token overlap
3. Adds domain-specific boosts (e.g., if query mentions "hanuman" and the record has Hanuman as a character, +0.2)
4. Clamps to `[0.36, 0.98]` — floor ensures all returned sources are above the `rag_min_score` threshold (0.35)

---

## The Unit Test Layer (29 tests, < 5 seconds)

- **Retrieval stats**: mean, min score, source count, section diversity with deduplication
- **Mode compliance regex**: all four modes with pass and fail cases
- **MetricScore labels**: good/fair/poor thresholds including boundary values
- **Overall score**: weighted average formula, zero case, partial case
- **Weight invariant**: `sum(METRIC_WEIGHTS.values()) == 1.0`
- **Passage formatting**: empty case, numbered passages, citation fallback
- **ValidationResult.to_dict()**: output keys, nested structure, correct values

---

## Auto-Skip, Not Auto-Fail

```python
def pytest_collection_modifyitems(config, items):
    if ollama_available():
        return
    skip = pytest.mark.skip(
        reason="Integration tests require local Ollama with the configured model"
    )
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)
```

Integration tests can be included in CI without requiring Ollama on the runner. On machines without it, they skip cleanly.

---

## The Trade-off

Replacing Pinecone with keyword scoring and Claude with `qwen2.5:1.5b` means the integration tests do not verify embedding quality or model-specific response formatting. What they do verify:

- Full pipeline wiring from query to scored result
- Data contracts (request/response/ValidationResult shapes)
- Evaluation math (weighted score formula, pass threshold)
- Mode compliance detection across all four modes
- JSON serialization of results

Embedding quality and model-specific behavior are validated separately through `run_evaluation.py` when credentials are available.

---

## Running the Tests

```bash
ollama pull qwen2.5:1.5b

make test-unit
make test-integration
make test-all

# Use a different local model
OLLAMA_MODEL=qwen2.5:7b make test-integration
```

---

## Closing

For AI pipelines, "test cheaply" means without cloud API calls on every commit. The local substitution approach — keyword retrieval from a seed corpus, small local model for generation, same local model for the judge — provides that. The tests are fast, the assertions are meaningful, and the pipeline wiring is exercised end-to-end.

*Code is open source at [github.com/ShambaviLabs/DharmaGPT](https://github.com/ShambaviLabs/DharmaGPT)*
