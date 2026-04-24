"""
normalize_raw_corpus.py — convert scraped raw JSONL into the flat DharmaGPT corpus schema.

Reads scraped files (nested metadata structure) from the data/chunks/ directory,
cleans text, fixes encoding issues, deduplicates records, and writes flat partitioned
JSONL files to knowledge/processed/ ready for translation and Pinecone ingestion.

Usage:
    python scripts/normalize_raw_corpus.py                         # normalize all scraped files
    python scripts/normalize_raw_corpus.py --file sundara_chunks.jsonl
    python scripts/normalize_raw_corpus.py --dry-run               # report issues, don't write
    python scripts/normalize.py --author chaganti --language te --kind audio
"""

import argparse
import json
import re
import sys
from pathlib import Path
from math import ceil

from utils.naming import canonical_jsonl_filename, normalize_language_tag, slugify

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
RAW_DIR     = REPO_ROOT / "knowledge" / "raw"
PROCESSED_DIR = REPO_ROOT / "knowledge" / "processed"

# Where scraped files live (sibling of the repo root)
SCRAPED_DIR = REPO_ROOT.parent / "data" / "chunks" / "valmiki_ramayanam"

# ─── Theme → Tag mapping ──────────────────────────────────────────────────────

THEME_TO_TAG = {
    "dharma":      "dharma",
    "karma":       "karma",
    "devotion":    "devotion",
    "bhakti":      "bhakti",
    "love":        "devotion",
    "war":         "war",
    "duty":        "duty",
    "ethics":      "ethics",
    "courage":     "courage",
    "wisdom":      "jnana",
    "sadhana":     "dharma",
    "grief":       "grief",
    "family":      "family",
    "sacrifice":   "duty",
    "moksha":      "moksha",
    "surrender":   "surrender",
    "story":       "story",
    "pilgrimage":  "pilgrimage",
    "ritual":      "ritual",
}

# Characters to always tag with devotion if present
DEVOTION_CHARACTERS = {"Rama", "Hanuman", "Sita", "Seetha", "Seeta"}

# ─── Text Cleaners ────────────────────────────────────────────────────────────

# Patterns to strip from text
NOISE_PATTERNS = [
    # "Book III : Aranya Kanda - Forest Treck Chapter [Sarga] 1 Verses converted to UTF-8, Sep, 09"
    re.compile(
        r"Book\s+[IVX]+\s*:.*?(?:converted to UTF-8[^\n.]*\.?)",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Introduction" keyword at start of sentence
    re.compile(r"\bIntroduction\b\s*", re.IGNORECASE),
    # "Verse Locator" + the romanized Sanskrit verse that follows (up to verse number like 6-1-1)
    re.compile(r"Verse Locator\s+.{0,500}?\d+-\d+-\d+", re.DOTALL),
    re.compile(r"Verse Locator\s*", re.IGNORECASE),
    # Trailing verse number like "6-1-1" or "2-1-1"
    re.compile(r"\b\d+-\d+-\d+\b"),
    # Verse separator pipes "||"
    re.compile(r"\|\|[\d\-\.]*"),
    re.compile(r"\s*\|\s*"),
    # "Chapter [Sarga] N" headers
    re.compile(r"Chapter\s*\[Sarga\]\s*\d+", re.IGNORECASE),
    # "~Verse N" artifacts
    re.compile(r"~Verse\s+\d+", re.IGNORECASE),
    # Consecutive whitespace
    re.compile(r"\n{2,}"),
    re.compile(r"[ \t]{2,}"),
]


def fix_encoding(text: str) -> str:
    """
    Fix double-encoded UTF-8 Sanskrit text.
    Scraped strings have latin1-range codepoints (U+0080–U+00FF) that are
    actually mis-read UTF-8 bytes — e.g., 'Ä\x81' should be 'ā'.
    Fixes each mojibake run independently so normal Unicode (curly quotes,
    em-dashes, etc.) in the same string is left untouched.
    """
    def _fix_run(m: re.Match) -> str:
        chunk = m.group(0)
        try:
            return chunk.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return chunk
    return re.sub(r"[\x80-\xff]+", _fix_run, text)


_ANALYSIS_START = re.compile(
    r"\s+\d+\.\s+[^=\n]{1,60}=",  # "1. shatrughnaH =" or "13. saMvR^itaH ="
)


def clean_text(raw: str) -> str:
    """Apply all noise patterns and encoding fix to a raw text chunk."""
    text = fix_encoding(raw)
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    # Truncate at start of word-by-word analysis block
    m = _ANALYSIS_START.search(text)
    if m:
        text = text[:m.start()]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_mostly_noise(text: str) -> bool:
    """Return True if the cleaned text is too short or is just a header."""
    if len(text) < 40:
        return True
    # Looks like it's only an intro/header, no actual dharmic content
    noise_indicators = [
        "converted to utf-8",
        "this canto is named",
        "this first chapter",
        "this chapter describes",
    ]
    lower = text.lower()
    if any(ind in lower for ind in noise_indicators) and len(text) < 200:
        return True
    return False


# ─── Schema Converter ─────────────────────────────────────────────────────────

def scraped_to_schema(raw: dict, filename: str) -> dict | None:
    """
    Convert a scraped record (nested metadata) to the flat DharmaGPT schema.
    Returns None if the record should be skipped.
    """
    meta = raw.get("metadata", {})
    text = clean_text(raw.get("text", ""))

    if is_mostly_noise(text):
        return None

    # Build tags from themes + character heuristics
    themes = meta.get("themes", []) or []
    tags = list({THEME_TO_TAG.get(t.lower(), t.lower()) for t in themes if t})
    characters = meta.get("characters", []) or []
    if any(c in DEVOTION_CHARACTERS for c in characters):
        if "devotion" not in tags:
            tags.append("devotion")
    if not tags:
        tags = ["story"]  # fallback

    # Detect if chunk contains Sanskrit verse markers
    has_shloka = bool(re.search(r"[।॥|ā ī ū ṛ ṭ ḍ ṇ ṅ ñ ṃ ḥ]", text))

    kanda = meta.get("kanda", "")
    sarga = meta.get("sarga")
    verse_index = meta.get("verse_index")

    # Build deterministic id from kanda + sarga + verse
    kanda_slug = kanda.lower().replace(" ", "_").replace("kanda", "kanda")
    chunk_id = raw.get("id") or f"{kanda_slug}_s{sarga or 0:03d}_v{verse_index or 0:03d}"

    return {
        "id": chunk_id,
        "text": text,
        "source": "valmiki_ramayana",
        "kanda": kanda,
        "sarga": sarga,
        "verse_start": verse_index,
        "verse_end": verse_index,
        "citation": meta.get("citation", f"Valmiki Ramayana, {kanda}"),
        "language": "en",           # scraped site is English translation + some Sanskrit
        "source_type": meta.get("source_type", "text"),
        "tags": tags,
        "topics": [],
        "characters": characters,
        "is_shloka": has_shloka,
        "url": meta.get("url", ""),
        "notes": f"Normalized from {filename}",
    }


def slugify_kanda(name: str) -> str:
    slug = (name or "unknown_kanda").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown_kanda"


def partition_dir_for(record: dict) -> Path:
    source_type = (record.get("source_type") or "text").strip().lower()
    source = (record.get("source") or "unknown_source").strip().lower().replace(" ", "_")
    kanda_slug = slugify_kanda(str(record.get("kanda") or "unknown_kanda"))
    return PROCESSED_DIR / source_type / source / kanda_slug


def write_partitioned_records(records: list[dict], partition_size: int, naming: dict[str, str]) -> list[Path]:
    """
    Write records as partitioned files:
    knowledge/processed/<source_type>/<source>/<kanda>/<canonical>_partNN.jsonl
    """
    grouped: dict[Path, list[dict]] = {}
    for record in records:
        base = partition_dir_for(record)
        grouped.setdefault(base, []).append(record)

    written_files: list[Path] = []
    for base, items in grouped.items():
        base.mkdir(parents=True, exist_ok=True)
        total_parts = max(1, ceil(len(items) / partition_size))
        for idx in range(total_parts):
            start = idx * partition_size
            end = start + partition_size
            part_items = items[start:end]
            out_path = base / canonical_jsonl_filename(
                naming["source"],
                title=naming["title"],
                author=naming["author"],
                language=naming["language"],
                kind=naming["kind"],
                part=idx + 1,
            )
            with out_path.open("w", encoding="utf-8") as fh:
                for record in part_items:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_files.append(out_path)

    return written_files


# ─── Main ─────────────────────────────────────────────────────────────────────

def _naming_context(src: Path, records: list[dict], source: str | None, title: str | None, author: str | None, language: str | None, kind: str | None) -> dict[str, str]:
    first = records[0] if records else {}
    resolved_source = source or (first.get("source") if isinstance(first.get("source"), str) else None) or src.stem
    resolved_title = title or (str(first.get("kanda")) if first.get("kanda") else None) or src.stem
    resolved_author = author or (str(first.get("author")) if first.get("author") else None)
    resolved_language = language or (str(first.get("language")) if first.get("language") else None) or "en"
    resolved_kind = kind or (str(first.get("source_type")) if first.get("source_type") else None) or "processed"
    if resolved_kind == "text":
        resolved_kind = "processed"
    elif resolved_kind == "audio_transcript":
        resolved_kind = "transcript"
    return {
        "source": slugify(resolved_source, default="unknown_source"),
        "title": slugify(resolved_title, default="unknown_title"),
        "author": slugify(resolved_author, default="unknown_author") if resolved_author else "unknown_author",
        "language": normalize_language_tag(resolved_language),
        "kind": slugify(resolved_kind, default="processed"),
    }


def normalize_file(
    src: Path,
    dry_run: bool,
    layout: str,
    partition_size: int,
    source: str | None,
    title: str | None,
    author: str | None,
    language: str | None,
    kind: str | None,
) -> dict:
    stats = {"read": 0, "written": 0, "skipped_noise": 0, "skipped_invalid": 0}

    lines_out = []
    seen_ids: set[str] = set()

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error in {src.name}: {e}")
                stats["skipped_invalid"] += 1
                continue

            record = scraped_to_schema(raw, src.name)
            if record is None:
                stats["skipped_noise"] += 1
                continue

            # Deduplicate
            if record["id"] in seen_ids:
                record["id"] += f"_dup{stats['written']}"
            seen_ids.add(record["id"])

            lines_out.append(json.dumps(record, ensure_ascii=False))
            stats["written"] += 1

    if not dry_run and lines_out:
        records = [json.loads(line) for line in lines_out]
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        naming = _naming_context(src, records, source, title, author, language, kind)
        if layout in {"flat", "both"}:
            out_path = PROCESSED_DIR / canonical_jsonl_filename(
                naming["source"],
                title=naming["title"],
                author=naming["author"],
                language=naming["language"],
                kind=naming["kind"],
                part=1,
            )
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(lines_out) + "\n")
            print(f"  ✅  flat: {src.name} → {out_path.name}  ({stats['written']} records)")

        if layout in {"partitioned", "both"}:
            written = write_partitioned_records(records, partition_size=partition_size, naming=naming)
            print(f"  ✅  partitioned: {src.name} → {len(written)} part file(s)")
    else:
        print(f"  🔍  {src.name}: {stats['written']} would be written, "
              f"{stats['skipped_noise']} noise skipped")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize scraped Valmiki JSONL files")
    parser.add_argument("--file", type=str, default=None, help="Single file to normalize")
    parser.add_argument("--dry-run", action="store_true", help="Report stats without writing")
    parser.add_argument(
        "--layout",
        choices=["flat", "partitioned", "both"],
        default="both",
        help="Output layout under knowledge/processed",
    )
    parser.add_argument("--source", type=str, default=None, help="Override canonical source slug")
    parser.add_argument("--title", type=str, default=None, help="Override canonical work/title slug")
    parser.add_argument("--author", type=str, default=None, help="Override canonical author slug")
    parser.add_argument("--language", type=str, default=None, help="Override canonical language slug")
    parser.add_argument("--kind", type=str, default=None, help="Override canonical artifact kind slug")
    parser.add_argument(
        "--partition-size",
        type=int,
        default=1000,
        help="Max records per partitioned output file",
    )
    args = parser.parse_args()

    if not SCRAPED_DIR.exists():
        sys.exit(f"❌  Scraped dir not found: {SCRAPED_DIR}\n"
                 "    Run this script from the repo root or adjust SCRAPED_DIR.")

    if args.file:
        files = [SCRAPED_DIR / args.file]
        if not files[0].exists():
            sys.exit(f"❌  File not found: {files[0]}")
    else:
        files = sorted(SCRAPED_DIR.glob("*_chunks.jsonl"))

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Normalizing {len(files)} file(s)...\n")

    totals = {"read": 0, "written": 0, "skipped_noise": 0, "skipped_invalid": 0}
    for f in files:
        if f.stat().st_size == 0:
            print(f"  ⏭️   {f.name}: empty — skipping")
            continue
        stats = normalize_file(
            f,
            dry_run=args.dry_run,
            layout=args.layout,
            partition_size=max(1, args.partition_size),
            source=args.source,
            title=args.title,
            author=args.author,
            language=args.language,
            kind=args.kind,
        )
        for k in totals:
            totals[k] += stats[k]

    print(f"\n{'─'*50}")
    print(f"Total read:          {totals['read']:,}")
    print(f"Total written:       {totals['written']:,}")

    print(f"Noise skipped:       {totals['skipped_noise']:,}")
    print(f"Invalid skipped:     {totals['skipped_invalid']:,}")
    retention = (totals['written'] / totals['read'] * 100) if totals['read'] else 0
    print(f"Retention rate:      {retention:.1f}%")


if __name__ == "__main__":
    main()
