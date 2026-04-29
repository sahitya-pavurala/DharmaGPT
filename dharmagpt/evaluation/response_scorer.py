"""
response_scorer.py - score a DharmaGPT RAG response across quality dimensions.

Entry point: validate_response(query, response) -> ValidationResult

Default judge stack (configurable via .env):
  sarvamai/sarvam-m   (primary)   -> answer_relevance, context_utilization
  sarvamai/sarvam-30b (secondary) -> faithfulness, citation_precision

Both run via an OpenAI-compatible API (default: http://localhost:8000/v1).
Override by passing judge_config to validate_response() — used in local model tests.

Two rule-based metrics (free, no API call):
  retrieval stats -> cosine score mean/min, source count, section diversity
  mode_compliance -> structural format check per query mode

Overall score is a weighted average of the four judge metrics.
A response passes when overall_score >= PASS_THRESHOLD (0.65).
"""

from __future__ import annotations

import json
import re
import statistics

import structlog

from core.config import get_settings
from core.llm import LLMBackend, LLMConfig, generate_text_sync
from evaluation.metric_definitions import MetricScore, RetrievalStats, ValidationResult
from models.schemas import QueryResponse, SourceChunk

log = structlog.get_logger()
settings = get_settings()

PASS_THRESHOLD = 0.65

# Weighted contribution of each judge metric to overall_score (must sum to 1.0)
METRIC_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.35,
    "answer_relevance": 0.30,
    "context_utilization": 0.20,
    "citation_precision": 0.15,
}

_PRIMARY_METRICS = ("answer_relevance", "context_utilization")
_SECONDARY_METRICS = ("faithfulness", "citation_precision")

# Regex patterns used to check whether an answer follows the expected mode format
_MODE_COMPLIANCE_PATTERNS: dict[str, re.Pattern] = {
    "guidance": re.compile(r"\?"),
    "story": re.compile(r"SOURCE\s*:", re.IGNORECASE),
    "children": re.compile(r"(what this story teaches|teaches us|moral)", re.IGNORECASE),
    "scholar": re.compile(r"(Kanda|Parva|Sarga|Chapter)\s*\d+", re.IGNORECASE),
}

_JUDGE_SYSTEM = (
    "You are a precise evaluator for a Hindu sacred texts Q&A system. "
    "Return only valid JSON - no markdown fences, no extra text."
)

_JUDGE_PROMPTS: dict[str, str] = {
    "primary": """\
Evaluate this response from DharmaGPT, a Q&A system grounded in Hindu sacred texts.

QUERY:
{query}

RETRIEVED PASSAGES (the only factual source the system should draw from):
{passages}

SYSTEM RESPONSE:
{answer}

Return ONLY this JSON object. Do not wrap it in markdown or add any explanation:
{{
  "answer_relevance": {{
    "score": <0.0-1.0>,
    "reasoning": "<one sentence>"
  }},
  "context_utilization": {{
    "score": <0.0-1.0>,
    "reasoning": "<one sentence>"
  }}
}}

SCORING CRITERIA:

answer_relevance (0-1): How directly and completely does the answer address the user's
query? 1.0 = fully on-topic and complete, 0.0 = completely off-topic or empty.

context_utilization (0-1): To what degree is the answer grounded in the retrieved passages
versus drawn from the model's general training data? 1.0 = answer is tightly built from
the passages, 0.0 = passages are ignored entirely.
""",
    "secondary": """\
Evaluate this response from DharmaGPT, a Q&A system grounded in Hindu sacred texts.

QUERY:
{query}

RETRIEVED PASSAGES (the only factual source the system should draw from):
{passages}

SYSTEM RESPONSE:
{answer}

Return ONLY this JSON object. Do not wrap it in markdown or add any explanation:
{{
  "faithfulness": {{
    "score": <0.0-1.0>,
    "unsupported_claims": ["<claim not found in passages>"],
    "reasoning": "<one sentence>"
  }},
  "citation_precision": {{
    "score": <0.0-1.0>,
    "invalid_citations": ["<citation that cannot be verified against the passages>"],
    "reasoning": "<one sentence>"
  }}
}}

SCORING CRITERIA:

faithfulness (0-1): What fraction of specific factual claims (events, characters, teachings,
verse contents) in the answer are directly supported by the retrieved passages above?
Claims that come only from model training data - not from the passages - must be flagged
as unsupported. Score 1.0 only if every factual claim is traceable to a passage.

citation_precision (0-1): What fraction of inline citations of the form [Text, Section,
Chapter/Verse X] are accurate and traceable to the retrieved passages? If the answer
makes specific claims without any citations at all, score 0.0.
""",
}


def _llm_config(role: str) -> LLMConfig:
    backend_name, model, api_key, base_url, timeout_sec = settings.evaluation_model_for(role)
    return LLMConfig(
        backend=LLMBackend(backend_name),
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        max_tokens=1024,
    )


def _format_passages_for_judge(sources: list[SourceChunk]) -> str:
    if not sources:
        return "[No passages were retrieved for this query]"
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"[{i}] {s.citation} (retrieval score={s.score})\n{s.text}")
    return "\n\n".join(parts)


def _call_judge(role: str, query: str, answer: str, sources: list[SourceChunk], config: LLMConfig) -> dict:
    prompt = _JUDGE_PROMPTS[role].format(
        query=query,
        passages=_format_passages_for_judge(sources),
        answer=answer,
    )
    raw = generate_text_sync(
        _JUDGE_SYSTEM,
        [{"role": "user", "content": prompt}],
        config,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"```\s*$", "", raw.strip())
    return json.loads(raw)


def _build_metric(name: str, judge_data: dict, detail_key: str | None = None) -> MetricScore:
    details: dict = {}
    if detail_key and judge_data.get(detail_key):
        details[detail_key] = judge_data[detail_key]
    return MetricScore(
        name=name,
        score=float(judge_data.get("score", 0.0)),
        reasoning=judge_data.get("reasoning", ""),
        details=details,
    )


def _compute_retrieval_stats(sources: list[SourceChunk]) -> RetrievalStats:
    scores = [s.score for s in sources]
    unique_sections = {s.section for s in sources if s.section}
    return RetrievalStats(
        score_mean=round(statistics.mean(scores), 4) if scores else 0.0,
        score_min=round(min(scores), 4) if scores else 0.0,
        source_count=len(sources),
        section_diversity=len(unique_sections),
    )


def _check_mode_compliance(answer: str, mode: str) -> bool:
    pattern = _MODE_COMPLIANCE_PATTERNS.get(mode)
    return bool(pattern and pattern.search(answer))


def _compute_overall_score(metrics: dict[str, MetricScore]) -> float:
    return sum(METRIC_WEIGHTS[k] * m.score for k, m in metrics.items())


def validate_response(
    query: str,
    response: QueryResponse,
    judge_config: LLMConfig | None = None,
) -> ValidationResult:
    """
    Score a RAG response across faithfulness, relevance, context use, and citation quality.

    Makes two judge LLM calls:
      - primary   (sarvamai/sarvam-m):   answer_relevance + context_utilization
      - secondary (sarvamai/sarvam-30b): faithfulness + citation_precision

    Both calls use judge_config when provided (overrides both roles).
    Pass an Ollama LLMConfig to score with a local model instead — no Sarvam server needed.

    Rule-based metrics (retrieval stats, mode compliance) are computed locally, no LLM.

    Returns a ValidationResult with per-metric scores and an overall pass/fail.
    """
    config_primary = judge_config or _llm_config("primary")
    config_secondary = judge_config or _llm_config("secondary")
    log.info(
        "scoring_response",
        query_id=response.query_id,
        mode=response.mode,
        primary_judge=config_primary.model,
        secondary_judge=config_secondary.model,
    )

    primary_data = _call_judge("primary", query, response.answer, response.sources, config_primary)
    secondary_data = _call_judge("secondary", query, response.answer, response.sources, config_secondary)

    faithfulness = _build_metric("faithfulness", secondary_data["faithfulness"], "unsupported_claims")
    answer_relevance = _build_metric("answer_relevance", primary_data["answer_relevance"])
    context_utilization = _build_metric("context_utilization", primary_data["context_utilization"])
    citation_precision = _build_metric("citation_precision", secondary_data["citation_precision"], "invalid_citations")

    llm_metrics = {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_utilization": context_utilization,
        "citation_precision": citation_precision,
    }
    overall = _compute_overall_score(llm_metrics)

    result = ValidationResult(
        query_id=response.query_id,
        query=query,
        mode=response.mode.value,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_utilization=context_utilization,
        citation_precision=citation_precision,
        retrieval=_compute_retrieval_stats(response.sources),
        mode_compliance=_check_mode_compliance(response.answer, response.mode.value),
        overall_score=overall,
        passed=overall >= PASS_THRESHOLD,
    )

    log.info(
        "scoring_done",
        query_id=response.query_id,
        overall=round(overall, 3),
        passed=result.passed,
    )
    return result
