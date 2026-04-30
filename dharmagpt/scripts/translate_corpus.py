"""
translate_corpus.py — batch-translate processed JSONL corpus records to English.

Reads JSONL files under knowledge/processed/ and fills the text_en_model field
for records that are not yet translated (non-English, no existing text_en_model).

Uses the configured TRANSLATION_BACKEND from .env (default: sarvam).
No fallbacks — if translation fails, the exception propagates and kills the run.

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

from core.backends.translation import get_translator

log = structlog.get_logger()

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


def _translate_record(record: dict, *, force: bool) -> tuple[dict, bool]:
    if not _needs_translation(record, force):
        return record, False

    source_text = _source_text(record)
    if not source_text:
        return record, False

    translator = get_translator()
    result = translator.translate(
        source_text,
        source_lang=_source_lang(record),
        target_lang="en",
    )

    if result.skipped:
        return record, False

    # Keep both field names for compatibility with ingestion and existing corpora.
    record["text_en_model"] = result.text
    record["text_en"] = result.text
    record["translation_backend"] = result.backend
    return record, True


def process_file(path: Path, *, max_workers: int, force: bool) -> dict:
    records = _load_records(path)
    updated = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_translate_record, record.copy(), force=force): idx
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
    parser = argparse.ArgumentParser(
        description="Pass 1 of the two-pass translation pipeline.\n"
                    "Translates missing records using the configured backend (default: ollama for bulk).\n"
                    "Run fix_bad_translations.py afterwards for pass 2 (Sarvam quality fix).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Translate a single JSONL file (relative to knowledge/processed/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-translate even if text_en_model already exists",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Parallel translation workers (default: 4)",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="Override TRANSLATION_BACKEND for this run (indictrans2 | ollama | sarvam | anthropic | openai | skip)",
    )
    args = parser.parse_args()

    # Backend override — set before get_translator() is called so lru_cache picks it up
    if args.backend:
        import os
        os.environ["TRANSLATION_BACKEND"] = args.backend.lower()
        from core.backends import translation as _tb
        _tb.get_translator.cache_clear()

    if args.file:
        files = [PROCESSED_DIR / args.file]
        if not files[0].exists():
            raise SystemExit(f"File not found: {files[0]}")
    else:
        files = sorted(PROCESSED_DIR.glob("**/*.jsonl"))

    if not files:
        raise SystemExit(f"No JSONL files found in {PROCESSED_DIR}")

    from core.backends.translation import get_translator
    backend_name = get_translator().backend_name
    print(f"Backend: {backend_name}  |  Files: {len(files)}")

    total_updated = 0
    for path in files:
        result = process_file(path, max_workers=max(1, args.max_workers), force=args.force)
        log.info("translation_complete", **result)
        if result["updated"]:
            print(f"  {result['file']}: +{result['updated']} of {result['records']}")
            total_updated += result["updated"]
    print(f"\nDone. Total translated: {total_updated}")


if __name__ == "__main__":
    main()
