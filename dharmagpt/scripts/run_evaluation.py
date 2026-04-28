"""
run_evaluation.py — evaluate DharmaGPT RAG response quality end-to-end.

For each question in the input JSONL file, this script:
  1. Runs the full RAG pipeline (embed → Pinecone retrieve → Claude generate).
  2. Scores the response with an LLM judge (faithfulness, relevance, context use, citations).
  3. Writes a per-question JSONL report and prints an aggregate summary.

Scoring dimensions:
  faithfulness (35%)       — claims grounded in retrieved passages, not hallucinated
  answer_relevance (30%)   — answer directly addresses the query
  context_utilization (20%)— answer draws from retrieved passages rather than ignoring them
  citation_precision (15%) — inline citations are accurate and traceable

A response passes when its weighted overall score >= 0.65.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --questions evaluation/sample_questions.jsonl
    python scripts/run_evaluation.py --output evaluation/reports/run.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = REPO_ROOT / "evaluation" / "sample_questions.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "reports" / "latest.jsonl"


def _print_summary(summary: dict) -> None:
    width = 50
    print(f"\n{'=' * width}")
    print(f"  Total evaluated:    {summary.get('total', 0)}")
    print(f"  Passed (>= 0.65):   {summary.get('passed', 0)}  ({summary.get('pass_rate', 0):.0%})")
    print(f"  Mode compliance:    {summary.get('mode_compliance_rate', 0):.0%}")
    print(f"{'─' * width}")
    print(f"  Overall score:          {summary.get('mean_overall', 0):.3f}")
    print(f"  Faithfulness (35%):     {summary.get('mean_faithfulness', 0):.3f}")
    print(f"  Answer relevance (30%): {summary.get('mean_answer_relevance', 0):.3f}")
    print(f"  Context utilization:    {summary.get('mean_context_utilization', 0):.3f}")
    print(f"  Citation precision:     {summary.get('mean_citation_precision', 0):.3f}")
    print(f"  Retrieval score (avg):  {summary.get('mean_retrieval_score', 0):.3f}")
    print(f"{'=' * width}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DharmaGPT RAG response quality"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS),
        help="JSONL file with evaluation questions (default: evaluation/sample_questions.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="JSONL file to write per-question results (default: evaluation/reports/latest.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit with code 1 if pass rate is below RATE (0.0-1.0). Used in CI.",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"Questions file not found: {questions_path}")

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.batch_runner import load_questions, run_batch, summarize

    questions = load_questions(questions_path)
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        sys.exit("No questions found in the input file.")

    print(f"Evaluating {len(questions)} question(s) from {questions_path.name} ...")

    results = asyncio.run(run_batch(questions))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    summary = summarize(results)
    _print_summary(summary)
    print(f"Full results written to: {output_path}\n")

    if args.fail_below is not None:
        pass_rate = summary.get("pass_rate", 0.0)
        if pass_rate < args.fail_below:
            print(
                f"FAIL: pass rate {pass_rate:.0%} is below threshold {args.fail_below:.0%}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
