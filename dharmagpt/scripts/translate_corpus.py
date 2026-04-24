"""
translate_corpus.py — batch-translate processed JSONL corpus records to English.

Reads JSONL files under knowledge/processed/ and fills the text_en_model field
for records that are not yet translated (non-English, no existing text_en_model).
Tries Anthropic first, falls back to Ollama, then IndicTrans2.

Usage:
    python scripts/translate_corpus.py
    python scripts/translate_corpus.py --file audio_transcript/my_dataset/part01.jsonl
    python scripts/translate_corpus.py --max-workers 6 --force
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import structlog

from core.config import get_settings
from core.translation import TranslationBackend, TranslationConfig, translate_text

log = structlog.get_logger()
settings = get_settings()

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "knowledge" / "processed"


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _translation_config() -> TranslationConfig:
    return TranslationConfig(
        backend=TranslationBackend.auto,
        anthropic_model=settings.anthropic_model,
        anthropic_api_key=settings.anthropic_api_key,
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        indictrans2_model=settings.indictrans2_model,
    )


def _source_text(record: dict) -> str:
    for field in ("text_te", "text", "text_en"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_lang(record: dict) -> str:
    language = (record.get("language") or "").strip().lower()
    if language in {"", "en", "eng", "english"}:
        return "en"
    if language in {"te", "tel", "telugu"}:
        return "te"
    if language in {"hi", "hin", "hindi"}:
        return "hi"
    if language in {"sa", "san", "sanskrit"}:
        return "sa"
    return language or "en"


def _needs_translation(record: dict, force: bool) -> bool:
    if force:
        return True
    if record.get("text_en_model"):
        return False
    return _source_lang(record) != "en"


def _translate_record(record: dict, *, config: TranslationConfig, force: bool) -> tuple[dict, bool]:
    if not _needs_translation(record, force):
        return record, False

    source_text = _source_text(record)
    if not source_text:
        return record, False

    try:
        outcome = translate_text(
            source_text,
            config=config,
            source_lang=_source_lang(record),
            target_lang="en",
        )
    except Exception as exc:
        record["translation_mode"] = config.backend.value
        record["translation_backend"] = ""
        record["translation_version"] = ""
        record["translation_fallback_reason"] = f"error:{exc}"
        record["translation_attempted_backends"] = []
        return record, False

    record["text_en_model"] = outcome.text
    record["translation_mode"] = outcome.requested_mode
    record["translation_backend"] = outcome.backend
    record["translation_version"] = outcome.version
    record["translation_fallback_reason"] = outcome.fallback_reason
    record["translation_attempted_backends"] = list(outcome.attempted_backends)
    return record, True


def process_file(path: Path, *, max_workers: int, force: bool) -> dict:
    records = _load_records(path)
    config = _translation_config()
    updated = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_translate_record, record.copy(), config=config, force=force): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            translated_record, changed = future.result()
            records[idx] = translated_record
            if changed:
                updated += 1

    if updated:
        _write_records(path, records)

    return {
        "file": path.name,
        "records": len(records),
        "updated": updated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch auto-translate processed JSONL datasets")
    parser.add_argument("--file", type=str, default=None, help="Translate a single JSONL file from knowledge/processed/")
    parser.add_argument("--force", action="store_true", help="Re-translate even if text_en_model already exists")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel translation workers")
    args = parser.parse_args()

    if args.file:
        files = [PROCESSED_DIR / args.file]
        if not files[0].exists():
            raise SystemExit(f"File not found: {files[0]}")
    else:
        files = sorted(PROCESSED_DIR.glob("*.jsonl"))

    if not files:
        raise SystemExit(f"No JSONL files found in {PROCESSED_DIR}")

    for path in files:
        result = process_file(path, max_workers=max(1, args.max_workers), force=args.force)
        log.info("auto_translation_complete", **result)
        print(f"{result['file']}: updated {result['updated']} of {result['records']} records")


if __name__ == "__main__":
    main()
