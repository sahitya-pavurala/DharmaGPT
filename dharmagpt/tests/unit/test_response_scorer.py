"""
test_response_scorer.py — unit tests for the response validation pipeline.

Tests cover:
  - RetrievalStats computation (mean, min, section diversity)
  - Mode compliance regex checks for all four query modes
  - MetricScore label thresholds (good / fair / poor)
  - Overall score weighted average and pass/fail threshold
  - Passage formatting for the LLM judge prompt
  - ValidationResult.to_dict() output shape
  - METRIC_WEIGHTS sum to 1.0

All tests are offline — no API calls are made.
"""

import pytest

from evaluation.metric_definitions import MetricScore, RetrievalStats, ValidationResult
from evaluation.response_scorer import (
    METRIC_WEIGHTS,
    PASS_THRESHOLD,
    _check_mode_compliance,
    _compute_overall_score,
    _compute_retrieval_stats,
    _format_passages_for_judge,
    validate_response,
)
from core.llm import LLMBackend, LLMConfig
from models.schemas import QueryMode, QueryResponse, SourceChunk


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _source(
    score: float = 0.85,
    section: str | None = "Sundara Kanda",
    chapter: int | None = 15,
    text: str = "Hanuman leapt across the ocean.",
) -> SourceChunk:
    citation = "Valmiki Ramayana"
    if section and chapter:
        citation = f"Valmiki Ramayana, {section}, Sarga {chapter}"
    elif section:
        citation = f"Valmiki Ramayana, {section}"
    return SourceChunk(
        text=text,
        citation=citation,
        section=section,
        chapter=chapter,
        score=score,
    )


def _response(
    answer: str = "Rama acted with dharma. [Valmiki Ramayana, Ayodhya Kanda, Sarga 2]",
    sources: list[SourceChunk] | None = None,
    mode: QueryMode = QueryMode.guidance,
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        sources=sources or [_source()],
        mode=mode,
        language="en",
        query_id="test-001",
    )


def _all_zero_metrics() -> dict[str, MetricScore]:
    return {k: MetricScore(k, 0.0) for k in METRIC_WEIGHTS}


def _all_one_metrics() -> dict[str, MetricScore]:
    return {k: MetricScore(k, 1.0) for k in METRIC_WEIGHTS}


# ─── RetrievalStats ───────────────────────────────────────────────────────────


def test_retrieval_stats_mean_and_min():
    sources = [_source(score=0.9), _source(score=0.7), _source(score=0.8)]
    stats = _compute_retrieval_stats(sources)
    assert stats.score_mean == pytest.approx(0.8, abs=0.01)
    assert stats.score_min == 0.7


def test_retrieval_stats_source_count():
    sources = [_source(), _source(), _source()]
    assert _compute_retrieval_stats(sources).source_count == 3


def test_retrieval_stats_section_diversity_deduplicates():
    sources = [
        _source(section="Sundara Kanda"),
        _source(section="Yuddha Kanda"),
        _source(section="Sundara Kanda"),
    ]
    assert _compute_retrieval_stats(sources).section_diversity == 2


def test_retrieval_stats_no_section():
    sources = [_source(section=None), _source(section=None)]
    assert _compute_retrieval_stats(sources).section_diversity == 0


def test_retrieval_stats_empty_sources():
    stats = _compute_retrieval_stats([])
    assert stats.score_mean == 0.0
    assert stats.score_min == 0.0
    assert stats.source_count == 0
    assert stats.section_diversity == 0


# ─── Mode compliance ──────────────────────────────────────────────────────────


def test_mode_compliance_guidance_passes_with_question():
    assert _check_mode_compliance("Is this the path of dharma?", "guidance") is True


def test_mode_compliance_guidance_fails_without_question():
    assert _check_mode_compliance("Rama acted with dharma.", "guidance") is False


def test_mode_compliance_story_passes_with_source_tag():
    assert _check_mode_compliance("He leapt across.\nSOURCE: Valmiki Ramayana.", "story") is True


def test_mode_compliance_story_fails_without_source_tag():
    assert _check_mode_compliance("He leapt across the ocean.", "story") is False


def test_mode_compliance_children_passes_with_moral():
    assert _check_mode_compliance("What this story teaches us: be brave.", "children") is True


def test_mode_compliance_children_alternate_phrasing():
    assert _check_mode_compliance("This teaches us to always speak the truth.", "children") is True


def test_mode_compliance_scholar_passes_with_kanda_reference():
    assert _check_mode_compliance("See Sundara Kanda 15 for this verse.", "scholar") is True


def test_mode_compliance_scholar_passes_with_sarga():
    assert _check_mode_compliance("Valmiki Ramayana, Sarga 5 states...", "scholar") is True


def test_mode_compliance_unknown_mode_returns_false():
    assert _check_mode_compliance("some text", "unknown_mode") is False


# ─── MetricScore label ────────────────────────────────────────────────────────


def test_metric_score_label_good():
    assert MetricScore("x", 0.9).label == "good"


def test_metric_score_label_fair():
    assert MetricScore("x", 0.6).label == "fair"


def test_metric_score_label_poor():
    assert MetricScore("x", 0.3).label == "poor"


def test_metric_score_label_boundary_good():
    assert MetricScore("x", 0.8).label == "good"


def test_metric_score_label_boundary_fair():
    assert MetricScore("x", 0.5).label == "fair"


# ─── Overall score ────────────────────────────────────────────────────────────


def test_overall_score_all_ones():
    assert _compute_overall_score(_all_one_metrics()) == pytest.approx(1.0)


def test_overall_score_all_zeros():
    assert _compute_overall_score(_all_zero_metrics()) == pytest.approx(0.0)


def test_overall_score_mixed():
    metrics = {
        "faithfulness": MetricScore("faithfulness", 1.0),
        "answer_relevance": MetricScore("answer_relevance", 0.0),
        "context_utilization": MetricScore("context_utilization", 0.0),
        "citation_precision": MetricScore("citation_precision", 0.0),
    }
    # Only faithfulness contributes: 0.35 * 1.0 = 0.35
    assert _compute_overall_score(metrics) == pytest.approx(0.35)


def test_weights_sum_to_one():
    assert sum(METRIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_pass_threshold_value():
    assert PASS_THRESHOLD == pytest.approx(0.65)


# ─── Passage formatting ───────────────────────────────────────────────────────


def test_format_passages_empty():
    result = _format_passages_for_judge([])
    assert "No passages" in result


def test_format_passages_includes_text_and_location():
    sources = [_source(text="Sita was found.", section="Sundara Kanda", chapter=15)]
    formatted = _format_passages_for_judge(sources)
    assert "Sita was found." in formatted
    assert "Sundara Kanda" in formatted
    assert "Sarga 15" in formatted


def test_format_passages_fallback_to_citation_when_no_section():
    sources = [_source(section=None, chapter=None)]
    formatted = _format_passages_for_judge(sources)
    assert "Valmiki Ramayana" in formatted


def test_format_passages_multiple_numbered():
    sources = [_source(text=f"text {i}") for i in range(3)]
    formatted = _format_passages_for_judge(sources)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "[3]" in formatted


# ─── ValidationResult.to_dict ─────────────────────────────────────────────────


def test_validation_result_to_dict_shape():
    result = ValidationResult(
        query_id="q1",
        query="What is dharma?",
        mode="guidance",
        faithfulness=MetricScore("faithfulness", 0.9, "All claims supported."),
        answer_relevance=MetricScore("answer_relevance", 0.8, "On topic."),
        context_utilization=MetricScore("context_utilization", 0.7, "Good use."),
        citation_precision=MetricScore("citation_precision", 0.6, "Some issues.", {"invalid_citations": ["fake ref"]}),
        retrieval=RetrievalStats(score_mean=0.82, score_min=0.71, source_count=3, section_diversity=2),
        mode_compliance=True,
        overall_score=0.79,
        passed=True,
    )
    d = result.to_dict()
    assert d["query_id"] == "q1"
    assert d["overall_score"] == 0.79
    assert d["passed"] is True
    assert d["mode_compliance"] is True
    assert "faithfulness" in d["metrics"]
    assert "retrieval" in d
    assert d["retrieval"]["source_count"] == 3
    assert d["metrics"]["citation_precision"]["details"] == {"invalid_citations": ["fake ref"]}


def test_validate_response_uses_one_combined_judge_call(monkeypatch):
    calls = []

    def fake_call_judge(role, query, answer, sources, config):
        calls.append(role)
        return {
            "faithfulness": {
                "score": 0.9,
                "unsupported_claims": [],
                "reasoning": "All factual claims are supported.",
            },
            "answer_relevance": {"score": 0.8, "reasoning": "The answer is relevant."},
            "context_utilization": {"score": 0.7, "reasoning": "The answer uses context."},
            "citation_precision": {
                "score": 0.6,
                "invalid_citations": [],
                "reasoning": "Citations are mostly precise.",
            },
        }

    monkeypatch.setattr("evaluation.response_scorer._call_judge", fake_call_judge)

    result = validate_response(
        "What is dharma?",
        _response(),
        judge_config=LLMConfig(
            backend=LLMBackend.ollama,
            model="qwen2.5:7b",
        ),
    )

    assert calls == ["combined"]
    assert result.overall_score == pytest.approx(0.785)
    assert result.passed is True
