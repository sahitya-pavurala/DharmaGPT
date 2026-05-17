# Validating RAG Responses for Dharmic AI: Four Metrics, Two Local Judges, Zero Cloud Calls

> How to know whether your AI actually used the sources it retrieved — or just made something up that sounds right.

**Suggested tags:** AI, RAG, LLM, Evaluation, NLP, India, Open Source

---

The hardest problem in RAG is not retrieval. Retrieval is measurable — cosine scores, source counts, metadata filters. The hard problem is what happens after retrieval: did the language model actually *use* the passages it was given, or did it generate a confident-sounding answer from its training data while ignoring the retrieved context entirely?

For a dharmic AI, this matters acutely. A hallucinated verse reference or a fabricated character action is not just a technical failure — it actively misleads someone asking about source texts they trust.

This post explains the validation pipeline built for DharmaGPT.

---

## The Four Metrics

### 1. Faithfulness (35% weight)
Are the factual claims in the answer directly supported by the retrieved passages? Claims about events, characters, teachings, or verse contents that come only from the model's training data — not from the retrieved context — are flagged as unsupported. Score 1.0 means every specific claim traces back to a passage.

### 2. Answer Relevance (30% weight)
Does the answer actually address the user's query? A retrieved context about Hanuman's journey is not useful if the question was about Rama's exile. This catches cases where the model produces a technically accurate passage summary that misses the question entirely.

### 3. Context Utilization (20% weight)
Did the answer draw from the retrieved passages, or ignore them? Subtly different from faithfulness: a model can be faithful to the passages it *chose* to use while effectively ignoring most of the retrieved context. This measures how tightly the answer is built from the retrieved material.

### 4. Citation Precision (15% weight)
Are the inline citations accurate and traceable? DharmaGPT's prompts require citations in the format `[Valmiki Ramayana, Sundara Kanda, Sarga 15]`. This checks whether cited sources actually match the retrieved passages.

---

## Overall Score and Pass Threshold

```
overall = 0.35 × faithfulness
        + 0.30 × answer_relevance
        + 0.20 × context_utilization
        + 0.15 × citation_precision
```

A response **passes** when `overall_score ≥ 0.65`. The weights reflect priority: faithfulness matters most for a source-grounded system; missing citations are a quality signal but not catastrophic if the answer is otherwise grounded.

---

## Two Judge Calls, Split by Concern

**Primary judge** (`sarvamai/sarvam-m`):
- answer_relevance + context_utilization
- Holistic reading comprehension — a smaller, faster model handles this well

**Secondary judge** (`sarvamai/sarvam-30b`):
- faithfulness + citation_precision
- Requires careful claim-by-claim comparison against source passages — a larger model is appropriate

Both run via a local OpenAI-compatible API (`localhost:8000/v1`). No cloud evaluation call.

---

## What the Judge Returns

**Primary output:**
```json
{
  "answer_relevance": {
    "score": 0.9,
    "reasoning": "The answer directly addresses anger management through the Gita's teaching on equanimity."
  },
  "context_utilization": {
    "score": 0.75,
    "reasoning": "Draws from passages 1 and 3 but ignores passage 2 which is most relevant."
  }
}
```

**Secondary output:**
```json
{
  "faithfulness": {
    "score": 0.6,
    "unsupported_claims": ["Arjuna wept for three days before the battle"],
    "reasoning": "One duration claim is not present in any retrieved passage."
  },
  "citation_precision": {
    "score": 0.85,
    "invalid_citations": [],
    "reasoning": "All cited sections match retrieved passages."
  }
}
```

The `unsupported_claims` and `invalid_citations` arrays are directly actionable for debugging.

---

## Switching to a Local Ollama Judge

For development without a Sarvam server, any Ollama model overrides both judge roles:

```python
from core.llm import LLMBackend, LLMConfig
from evaluation.response_scorer import validate_response

local_judge = LLMConfig(
    backend=LLMBackend.ollama,
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
)
result = validate_response(query, response, judge_config=local_judge)
```

---

## Rule-Based Metrics (No LLM)

**Retrieval stats** (computed from chunk metadata):
- `score_mean`, `score_min` — cosine similarity distribution
- `source_count` — how many chunks were retrieved
- `section_diversity` — distinct text sections in the retrieved set (kanda/parva/adhyaya/skandha — generic across all Indic texts)

**Mode compliance** (regex checks, no model):
- `guidance` → must contain a reflection question (`?`)
- `story` → must contain `SOURCE:` tag
- `children` → must contain a moral lesson phrase
- `scholar` → must reference a section and number

---

## Running the Evaluation

```bash
cd dharmagpt && PYTHONPATH=. python scripts/run_evaluation.py

# Quick smoke test
python scripts/run_evaluation.py --limit 3
```

Sample output:
```
==================================================
  Total evaluated:    10
  Passed (>= 0.65):   8  (80%)
  Mode compliance:    90%
──────────────────────────────────────────────────
  Overall score:          0.741
  Faithfulness (35%):     0.782
  Answer relevance (30%): 0.810
  Context utilization:    0.694
  Citation precision:     0.651
  Retrieval score (avg):  0.823
==================================================
```

Results are written as JSONL to `evaluation/reports/` with full per-question breakdowns.

---

## What This Validates — and What It Does Not

**Validates:** whether the model used its retrieved context well.

**Does not validate:**
- Whether retrieval itself found the right chunks (covered by cosine scores and section diversity)
- Whether the source texts themselves are accurate (requires human scholarship review)
- Whether the answer is spiritually appropriate for the seeker's situation (requires domain expert judgment)

The `unsupported_claims` list from the faithfulness judge is the most directly actionable output for catching specific hallucinations.

---

## Closing

Grounded AI requires measurable grounding. The four metrics give concrete, actionable numbers for whether a RAG system used what it retrieved. The split judge design keeps each evaluation concern focused. The local-model override makes the pipeline usable without a cloud evaluation API.

*Code is open source at [github.com/ShambaviLabs/DharmaGPT](https://github.com/ShambaviLabs/DharmaGPT)*
