"""
batch_runner.py - run the validation pipeline over a set of sample questions.

load_questions(path)   - read a JSONL file of {query, mode} objects
run_batch(questions)   - generate a live RAG response for each question and score it
summarize(results)     - aggregate ValidationResult list into a human-readable summary dict

Typical flow (called by scripts/run_evaluation.py):
    questions = load_questions(Path("evaluation/sample_questions.jsonl"))
    results   = asyncio.run(run_batch(questions))
    summary   = summarize(results)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog

from core.rag_engine import answer
from evaluation.metric_definitions import ValidationResult
from evaluation.response_scorer import validate_response
from models.schemas import QueryMode, QueryRequest

log = structlog.get_logger()


def load_questions(path: Path) -> list[dict]:
    """Read a JSONL file of evaluation questions.

    Each non-blank line must be a JSON object with at minimum a "query" key.
    Optional keys: "mode" (defaults to "guidance"), "filter_kanda".
    Lines beginning with "//" are treated as comments and skipped.
    """
    questions: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                questions.append(json.loads(line))
    return questions


async def _evaluate_one(q: dict) -> ValidationResult:
    request = QueryRequest(
        query=q["query"],
        mode=QueryMode(q.get("mode", "guidance")),
        filter_kanda=q.get("filter_kanda"),
    )
    response = await answer(request)
    # validate_response is synchronous (two blocking local judge calls by default) - run in thread
    return await asyncio.to_thread(validate_response, q["query"], response)


async def run_batch(questions: list[dict]) -> list[ValidationResult]:
    """Generate and score a RAG response for each question sequentially.

    Sequential (not concurrent) to avoid hammering the judge stack with
    parallel calls on top of parallel generation calls.
    """
    results: list[ValidationResult] = []
    for i, q in enumerate(questions, 1):
        log.info("batch_progress", current=i, total=len(questions), query=q["query"][:60])
        try:
            result = await _evaluate_one(q)
            results.append(result)
            log.info(
                "question_scored",
                overall=round(result.overall_score, 3),
                passed=result.passed,
                faithfulness=round(result.faithfulness.score, 3),
            )
        except Exception as exc:
            log.error("question_failed", query=q["query"][:60], error=str(exc))
    return results


def summarize(results: list[ValidationResult]) -> dict:
    """Aggregate a list of ValidationResults into a flat summary dict."""
    if not results:
        return {"total": 0}

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    compliant = sum(1 for r in results if r.mode_compliance)

    def _mean(attr: str) -> float:
        return round(sum(getattr(r, attr).score for r in results) / n, 3)

    return {
        "total": n,
        "passed": passed,
        "pass_rate": round(passed / n, 3),
        "mode_compliance_rate": round(compliant / n, 3),
        "mean_overall": round(sum(r.overall_score for r in results) / n, 3),
        "mean_faithfulness": _mean("faithfulness"),
        "mean_answer_relevance": _mean("answer_relevance"),
        "mean_context_utilization": _mean("context_utilization"),
        "mean_citation_precision": _mean("citation_precision"),
        "mean_retrieval_score": round(
            sum(r.retrieval.score_mean for r in results) / n, 3
        ),
    }
