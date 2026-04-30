from __future__ import annotations

import re
from pathlib import Path


_PART_SUFFIX = re.compile(r"_part\d+$")
_ARTIFACT_SUFFIX = re.compile(r"_(?:audio|transcript|processed|manual)_part\d+$")
_PART_NUMBER = re.compile(r"_part(\d+)$")
_LEGACY_PART_FILE = re.compile(r"^part-\d{4}$")


def slugify(value: str | None, *, default: str = "unknown") -> str:
    text = (value or "").strip().lower()
    if not text:
        return default
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def normalize_language_tag(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "en"
    if text in {"te", "tel", "telugu", "te-in"}:
        return "te"
    if text in {"en", "eng", "english", "en-us", "en-gb"}:
        return "en"
    if text in {"hi", "hin", "hindi", "hi-in"}:
        return "hi"
    if text in {"sa", "san", "sanskrit"}:
        return "sa"
    return slugify(text, default="en")


def base_stem_from_filename(value: str | None, *, default: str = "unknown") -> str:
    """
    Derive a reusable artifact stem from an uploaded filename or existing stem.

    This strips a trailing canonical artifact suffix such as `_audio_part01`
    or `_transcript_part01` so derived files can keep the same stable base.
    """
    if not value:
        return default
    stem = slugify(Path(value).stem, default=default)
    stem = _ARTIFACT_SUFFIX.sub("", stem)
    stem = _PART_SUFFIX.sub("", stem)
    return stem or default


def source_stem_from_audio_filename(
    value: str | None,
    *,
    language: str | None = None,
    default: str = "unknown",
) -> str:
    """
    Derive the base source stem from an audio filename.

    This strips an optional trailing language token plus artifact suffix so
    `valmiki_ramayanam_chaganti_te_audio_part01.mp3` becomes
    `valmiki_ramayanam_chaganti`.
    """
    if not value:
        return default

    stem = slugify(Path(value).stem, default=default)
    lang = normalize_language_tag(language) if language else None
    if lang:
        stem = re.sub(rf"_{re.escape(lang)}_(?:audio|transcript|processed|manual)_part\d+$", "", stem)
        stem = re.sub(rf"_{re.escape(lang)}_part\d+$", "", stem)
        stem = re.sub(rf"_{re.escape(lang)}$", "", stem)
    stem = _ARTIFACT_SUFFIX.sub("", stem)
    stem = _PART_SUFFIX.sub("", stem)
    return stem or default


def part_number_from_filename(value: str | None, *, default: int = 1) -> int:
    if not value:
        return default

    stem = Path(value).stem
    match = _PART_NUMBER.search(slugify(stem, default=""))
    if not match:
        return default
    try:
        raw_part = match.group(1)
        part = int(raw_part)
        if part == 0 or len(raw_part) >= 4:
            return part + 1
        return max(1, part)
    except ValueError:
        return default


def canonical_dataset_stem(
    source: str,
    *,
    language: str,
    kind: str,
    title: str | None = None,
    author: str | None = None,
    part: int = 1,
) -> str:
    pieces = [
        slugify(source),
        slugify(title) if title else "",
        slugify(author) if author else "",
        normalize_language_tag(language),
        slugify(kind),
        f"part{part:02d}",
    ]
    return "_".join(piece for piece in pieces if piece)


def canonical_jsonl_filename(
    source: str,
    *,
    language: str,
    kind: str,
    title: str | None = None,
    author: str | None = None,
    part: int = 1,
) -> str:
    return canonical_dataset_stem(
        source,
        language=language,
        kind=kind,
        title=title,
        author=author,
        part=part,
    ) + ".jsonl"


def dataset_id_from_path(path: Path, root: Path | None = None) -> str:
    """
    Map a processed-file path to the canonical dataset id.

    Prefer canonical stems like `valmiki_ramayanam_chaganti_te_audio_part01`.
    Fall back to legacy nested layouts when we encounter old `part-0001.jsonl`
    files inside directory trees.
    """
    stem = path.stem
    if stem and not _LEGACY_PART_FILE.match(stem):
        return stem

    if root is None:
        return stem

    try:
        rel = path.relative_to(root)
    except ValueError:
        return stem

    parent_parts = rel.parts[:-1]
    if not parent_parts:
        return stem
    return ".".join(parent_parts)


def is_canonical_part_file(path: Path) -> bool:
    return bool(_PART_SUFFIX.search(path.stem))
