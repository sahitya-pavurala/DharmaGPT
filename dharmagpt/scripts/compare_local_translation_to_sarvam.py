"""
compare_local_translation_to_sarvam.py

Generate local Ollama translations for Sarvam-transcribed audio records and
compare the local English output against Sarvam's saved English translation.

The script is intentionally side-effect light: it does not mutate corpus JSONL
files. It writes a resumable JSONL detail report plus a compact summary JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "dharmagpt"
    / "knowledge"
    / "processed"
    / "audio_transcript"
    / "20260429t024410z_ramayana_avashyaktamu_day01"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dharmagpt" / "knowledge" / "reports"


load_dotenv(REPO_ROOT / "dharmagpt" / ".env")


def _read_record(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty JSONL file: {path}")
    return json.loads(lines[0])


def _record_part(path: Path) -> int:
    match = re.search(r"_part(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _load_records(dataset_dir: Path) -> list[tuple[Path, dict]]:
    paths = sorted(dataset_dir.glob("*.jsonl"), key=_record_part)
    records: list[tuple[Path, dict]] = []
    for path in paths:
        record = _read_record(path)
        if (record.get("text") or "").strip() and (record.get("text_en") or "").strip():
            records.append((path, record))
    return records


def _load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = row.get("id")
        if record_id:
            done[record_id] = row
    return done


def _ollama_generate(text: str, *, model: str, base_url: str, timeout: int) -> str:
    resp = requests.post(
        base_url.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Telugu to English translator for Hindu scripture lectures. "
                        "Return only fluent English translation. Do not transliterate into Latin script. "
                        "Do not explain your process. Do not repeat these instructions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Translate this Telugu transcript to faithful English. "
                        "Preserve names like Rama, Valmiki, Vishwamitra, Ahalya, and Sanskrit terms, "
                        "but translate the meaning into English sentences.\n\n"
                        f"{text}"
                    ),
                },
            ],
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return ((resp.json().get("message") or {}).get("content") or "").strip()


def _local_translate(
    text: str,
    *,
    backend: str,
    model: str,
    base_url: str,
    timeout: int,
) -> tuple[str, str]:
    if backend == "ollama":
        return (
            _ollama_generate(text, model=model, base_url=base_url, timeout=timeout),
            model,
        )
    if backend == "indictrans2":
        from core.translation import TranslationBackend, TranslationConfig, translate_text

        outcome = translate_text(
            text,
            config=TranslationConfig(
                backend=TranslationBackend.indictrans2,
                indictrans2_model=model,
            ),
            source_lang="te",
            target_lang="en",
        )
        return outcome.text, outcome.version
    raise ValueError(f"Unsupported local backend: {backend}")


def _ollama_embeddings(texts: list[str], *, model: str, base_url: str, timeout: int) -> list[list[float]]:
    resp = requests.post(
        base_url.rstrip("/") + "/api/embed",
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    vectors = data.get("embeddings") or []
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _flags(local_text: str, sarvam_text: str) -> list[str]:
    flags: list[str] = []
    local = (local_text or "").strip()
    sarvam = (sarvam_text or "").strip()
    if not local:
        flags.append("empty_local")
        return flags
    if "translation:" in local.lower() or "telugu:" in local.lower():
        flags.append("prompt_leak")
    if (
        "preserve names" in local.lower()
        or "do not explain" in local.lower()
        or "sacred terms" in local.lower()
        or "return only" in local.lower()
    ):
        flags.append("prompt_leak")
    if re.search(r"[\u0C00-\u0C7F]", local):
        flags.append("contains_telugu")
    transliteration_tokens = re.findall(r"\b[a-z]+[āīūṛṝḷḹṅñṭḍṇśṣḥṁṃ][a-zāīūṛṝḷḹṅñṭḍṇśṣḥṁṃ]*\b", local.lower())
    if len(transliteration_tokens) >= 8:
        flags.append("likely_transliteration")
    if len(local) < max(40, len(sarvam) * 0.35):
        flags.append("too_short")
    if len(local) > max(300, len(sarvam) * 2.5):
        flags.append("too_long")
    if len(set(local.lower().split())) <= 4 and len(local.split()) > 8:
        flags.append("repetitive")
    return flags


def _bucket(semantic_similarity: float, flags: list[str]) -> str:
    if "empty_local" in flags or "prompt_leak" in flags:
        return "fail"
    if semantic_similarity >= 0.88 and not flags:
        return "strong"
    if semantic_similarity >= 0.80 and not {"too_short", "too_long"} & set(flags):
        return "usable"
    if semantic_similarity >= 0.72:
        return "review"
    return "fail"


def _summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    similarities = [row["semantic_similarity"] for row in rows]
    sequence_ratios = [row["sequence_ratio"] for row in rows]
    buckets: dict[str, int] = {}
    flags: dict[str, int] = {}
    for row in rows:
        buckets[row["quality_bucket"]] = buckets.get(row["quality_bucket"], 0) + 1
        for flag in row.get("flags", []):
            flags[flag] = flags.get(flag, 0) + 1
    worst = sorted(rows, key=lambda row: row["semantic_similarity"])[:10]
    return {
        "records": len(rows),
        "semantic_similarity": {
            "mean": statistics.mean(similarities) if similarities else 0.0,
            "median": statistics.median(similarities) if similarities else 0.0,
            "min": min(similarities) if similarities else 0.0,
            "max": max(similarities) if similarities else 0.0,
        },
        "sequence_ratio": {
            "mean": statistics.mean(sequence_ratios) if sequence_ratios else 0.0,
            "median": statistics.median(sequence_ratios) if sequence_ratios else 0.0,
        },
        "quality_buckets": buckets,
        "flags": flags,
        "worst_examples": [
            {
                "id": row["id"],
                "source_file": row["source_file"],
                "semantic_similarity": row["semantic_similarity"],
                "quality_bucket": row["quality_bucket"],
                "flags": row["flags"],
                "sarvam_preview": row["sarvam_text"][:220],
                "local_preview": row["local_text"][:220],
            }
            for row in worst
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local Ollama translations against saved Sarvam translations")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--translation-model", default="qwen2.5:7b")
    parser.add_argument("--local-backend", default="ollama", choices=["ollama", "indictrans2"])
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="Optional max records for a sample run")
    parser.add_argument("--force", action="store_true", help="Recompute rows already present in the JSONL report")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / f"{args.local_backend}_translation_vs_sarvam_{dataset_dir.name}.jsonl"
    summary_path = output_dir / f"{args.local_backend}_translation_vs_sarvam_{dataset_dir.name}_summary.json"

    records = _load_records(dataset_dir)
    if args.limit:
        records = records[: args.limit]

    done = {} if args.force else _load_done(detail_path)
    rows = [] if args.force else list(done.values())

    mode = "w" if args.force else "a"
    started = time.time()
    with detail_path.open(mode, encoding="utf-8") as out:
        for index, (path, record) in enumerate(records, start=1):
            record_id = record.get("id") or path.stem
            if record_id in done:
                print(f"[{index}/{len(records)}] skip {record_id}")
                continue

            te_text = (record.get("text") or "").strip()
            sarvam_text = (record.get("text_en") or "").strip()
            print(f"[{index}/{len(records)}] translate {record_id}")

            local_text, local_version = _local_translate(
                te_text,
                backend=args.local_backend,
                model=args.translation_model,
                base_url=args.ollama_url,
                timeout=args.timeout,
            )
            emb_sarvam, emb_local = _ollama_embeddings(
                [sarvam_text, local_text],
                model=args.embedding_model,
                base_url=args.ollama_url,
                timeout=args.timeout,
            )
            semantic = _cosine(emb_sarvam, emb_local)
            flags = _flags(local_text, sarvam_text)
            row = {
                "id": record_id,
                "record_file": str(path),
                "source": record.get("source"),
                "source_file": record.get("source_file"),
                "language": record.get("language"),
                "sarvam_backend": record.get("translation_backend"),
                "sarvam_version": record.get("translation_version"),
                "local_backend": args.local_backend,
                "local_model": local_version,
                "embedding_model": args.embedding_model,
                "sarvam_text": sarvam_text,
                "local_text": local_text,
                "semantic_similarity": round(semantic, 4),
                "sequence_ratio": round(_ratio(sarvam_text, local_text), 4),
                "length_ratio": round(len(local_text) / max(1, len(sarvam_text)), 4),
                "flags": flags,
                "quality_bucket": _bucket(semantic, flags),
            }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    summary = _summarize(rows)
    summary.update(
        {
            "dataset_dir": str(dataset_dir),
            "detail_report": str(detail_path),
            "translation_model": args.translation_model,
            "local_backend": args.local_backend,
            "embedding_model": args.embedding_model,
            "elapsed_sec": round(time.time() - started, 2),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nComplete")
    print(f"  Records: {summary['records']}")
    print(f"  Detail:  {detail_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Buckets: {summary['quality_buckets']}")
    print(f"  Mean semantic similarity: {summary['semantic_similarity']['mean']:.4f}")


if __name__ == "__main__":
    main()
