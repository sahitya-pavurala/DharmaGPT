"""
normalize.py — DharmaGPT Corpus Normalizer
===========================================
Converts scraped JSONL files (nested metadata schema) into the flat
DharmaGPT corpus schema, cleaning text and fixing encoding issues.

Usage:
    python scripts/normalize.py                         # normalize all scraped files
    python scripts/normalize.py --file sundara_chunks.jsonl
    python scripts/normalize.py --dry-run               # report issues, don't write
"""

import argparse
import json
import re
import sys
from pathlib import Path

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def normalize_file(src: Path, dry_run: bool) -> dict:
    stats = {"read": 0, "written": 0, "skipped_noise": 0, "skipped_invalid": 0}

    out_name = src.stem.replace("_chunks", "") + "_normalized.jsonl"
    out_path = PROCESSED_DIR / out_name

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
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines_out) + "\n")
        print(f"  ✅  {src.name} → {out_path.name}  ({stats['written']} records written)")
    else:
        print(f"  🔍  {src.name}: {stats['written']} would be written, "
              f"{stats['skipped_noise']} noise skipped")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize scraped Valmiki JSONL files")
    parser.add_argument("--file", type=str, default=None, help="Single file to normalize")
    parser.add_argument("--dry-run", action="store_true", help="Report stats without writing")
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
        stats = normalize_file(f, dry_run=args.dry_run)
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
