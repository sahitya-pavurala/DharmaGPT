"""
metric_definitions.py — data classes for DharmaGPT response validation results.

MetricScore holds a single scored dimension (faithfulness, relevance, etc.).
ValidationResult bundles all scores for one query/response pair and is the
output type of response_scorer.validate_response().
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricScore:
    """Score for a single evaluation dimension, 0.0 (worst) to 1.0 (best)."""

    name: str
    score: float
    reasoning: str = ""
    details: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.score >= 0.8:
            return "good"
        if self.score >= 0.5:
            return "fair"
        return "poor"


@dataclass
class RetrievalStats:
    """Derived statistics about the chunks retrieved for a query — no LLM call needed."""

    score_mean: float
    score_min: float
    source_count: int
    kanda_diversity: int


@dataclass
class ValidationResult:
    """
    Full evaluation result for one query/response pair.

    LLM-judged metrics (require an Anthropic API call):
      faithfulness         — claims in the answer are grounded in retrieved passages
      answer_relevance     — answer directly addresses the query
      context_utilization  — answer uses retrieved passages rather than ignoring them
      citation_precision   — inline citations are accurate and traceable to sources

    Rule-based metrics (free, no LLM call):
      retrieval            — cosine score stats and source diversity for the retrieved chunks
      mode_compliance      — structural check that the answer follows the requested mode format
    """

    query_id: str
    query: str
    mode: str

    # LLM-judged
    faithfulness: MetricScore
    answer_relevance: MetricScore
    context_utilization: MetricScore
    citation_precision: MetricScore

    # Rule-based
    retrieval: RetrievalStats
    mode_compliance: bool

    # Summary
    overall_score: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "mode": self.mode,
            "overall_score": round(self.overall_score, 3),
            "passed": self.passed,
            "metrics": {
                "faithfulness": {
                    "score": self.faithfulness.score,
                    "label": self.faithfulness.label,
                    "reasoning": self.faithfulness.reasoning,
                    "details": self.faithfulness.details,
                },
                "answer_relevance": {
                    "score": self.answer_relevance.score,
                    "label": self.answer_relevance.label,
                    "reasoning": self.answer_relevance.reasoning,
                },
                "context_utilization": {
                    "score": self.context_utilization.score,
                    "label": self.context_utilization.label,
                    "reasoning": self.context_utilization.reasoning,
                },
                "citation_precision": {
                    "score": self.citation_precision.score,
                    "label": self.citation_precision.label,
                    "reasoning": self.citation_precision.reasoning,
                    "details": self.citation_precision.details,
                },
            },
            "retrieval": {
                "score_mean": self.retrieval.score_mean,
                "score_min": self.retrieval.score_min,
                "source_count": self.retrieval.source_count,
                "kanda_diversity": self.retrieval.kanda_diversity,
            },
            "mode_compliance": self.mode_compliance,
        }
