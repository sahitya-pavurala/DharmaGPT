"""
scripts/audio/audio_pipeline.py

Processes Sanskrit/Hindi audio files via Sarvam Saaras v3 STT,
chunks by pause boundaries, outputs JSONL ready for embed_and_index.py.

Usage:
    pip install requests python-dotenv tqdm
    python scripts/audio/audio_pipeline.py --input data/audio/hanuman_chalisa.mp3
    python scripts/audio/audio_pipeline.py --input data/audio/ --batch --lang hi-IN
"""

import os
import re
import sys
import json
import time
import uuid
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
from dotenv import load_dotenv

load_dotenv("../../dharmagpt/.env")

SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL  = "https://api.sarvam.ai/speech-to-text"
SUPPORTED       = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma"}

SACRED_MARKERS  = [
    "shri ram", "jai ram", "jai hanuman", "namah shivaya", "om namo",
    "sita ram", "jai siya ram", "pavan putra", "anjaneya", "bajrangbali",
    "om namah shivaya", "hare krishna", "jai shri krishna",
]
SHLOKA_PAT = re.compile(r"[।॥|]{1,2}")

THEMES = {
    "devotion": ["bhakti", "devotion", "worship", "pray", "surrender"],
    "dharma":   ["dharma", "duty", "righteous", "virtue", "truth"],
    "courage":  ["courage", "brave", "fearless", "strength"],
    "wisdom":   ["wisdom", "wise", "knowledge", "intellect"],
    "sadhana":  ["sadhana", "tapas", "penance", "meditation", "yogi"],
}


# ── Sarvam STT ─────────────────────────────────────────────────────────────────

def transcribe(path: Path, lang: str) -> dict:
    """Call Saaras v3. Cache result as .transcript.json next to audio."""
    cache = path.with_suffix(".transcript.json")
    if cache.exists():
        print(f"    [cache] {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    mime_map = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
        ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
        ".opus": "audio/opus", ".wma": "audio/x-ms-wma",
    }
    mime = mime_map.get(path.suffix.lower(), "audio/mpeg")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    resp = requests.post(
        SARVAM_STT_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        files={"file": (path.name, audio_bytes, mime)},
        data={
            "model": "saaras:v3",
            "language_code": lang,
            "with_timestamps": "true",
            "with_diarization": "true",
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# ── Chunking ───────────────────────────────────────────────────────────────────

def _speaker(text: str) -> str:
    has_danda = bool(SHLOKA_PAT.search(text))
    en_ratio  = len(re.findall(r"\b[a-zA-Z]{3,}\b", text)) / max(len(text.split()), 1)
    sacred    = any(m in text.lower() for m in SACRED_MARKERS)
    if has_danda or sacred:
        return "chanting"
    return "commentary_english" if en_ratio > 0.5 else "commentary_hindi"


def chunk_by_pause(words: list[dict], min_w: int = 12, max_w: int = 70) -> list[dict]:
    chunks, buf, t0 = [], [], 0.0
    for i, w in enumerate(words):
        buf.append(w)
        last = i == len(words) - 1
        gap  = (words[i+1].get("start", 0) - w.get("end", 0)) if not last else 999
        text = " ".join(x.get("word","") for x in buf)
        cut  = (
            (gap > 0.8 and len(buf) >= min_w)
            or (bool(SHLOKA_PAT.search(text)) and len(buf) >= min_w)
            or len(buf) >= max_w
            or last
        )
        if cut and buf:
            t = re.sub(r"\s+", " ", text).strip()
            chunks.append({"text": t, "start": t0, "end": w.get("end", 0), "speaker": _speaker(t), "has_shloka": bool(SHLOKA_PAT.search(t))})
            buf = []
            t0  = words[i+1].get("start", 0) if not last else 0
    return chunks


def chunk_fallback(text: str) -> list[dict]:
    segs, buf = re.split(r"[।॥|]{1,2}|\.(?=\s)", text), []
    chunks = []
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        buf.append(seg)
        if len(" ".join(buf).split()) >= 20:
            t = " ".join(buf)
            chunks.append({"text": t, "start": None, "end": None, "speaker": _speaker(t), "has_shloka": bool(SHLOKA_PAT.search(t))})
            buf = []
    if buf:
        t = " ".join(buf)
        chunks.append({"text": t, "start": None, "end": None, "speaker": _speaker(t), "has_shloka": bool(SHLOKA_PAT.search(t))})
    return chunks


# ── Build output chunk ─────────────────────────────────────────────────────────

def build_chunk(raw: dict, audio_name: str, idx: int, meta: dict) -> dict:
    text = raw["text"]
    text_lower = text.lower()
    return {
        "id": f"audio_{audio_name}_{uuid.uuid4().hex[:6]}_{idx:04d}",
        "text": text,
        "metadata": {
            "source_type":     "audio",
            "source_file":     audio_name,
            "kanda":           meta.get("kanda", ""),
            "sarga":           meta.get("sarga", ""),
            "language":        meta.get("lang", "hi-IN"),
            "description":     meta.get("description", audio_name),
            "citation":        f"Audio: {meta.get('description', audio_name)}",
            "start_time_sec":  raw.get("start") or "",
            "end_time_sec":    raw.get("end") or "",
            "speaker_type":    raw.get("speaker", "unknown"),
            "has_shloka":      raw.get("has_shloka", False),
            "word_count":      len(text.split()),
            "themes":          [t for t, ws in THEMES.items() if any(w in text_lower for w in ws)],
            "characters":      [],
        },
    }


# ── Process one file ───────────────────────────────────────────────────────────

def process_file(path: Path, out_path: Path, lang: str, file_meta: dict):
    print(f"\n  {path.name}")
    data = transcribe(path, lang)
    words = data.get("words", [])
    raw_chunks = chunk_by_pause(words) if words else chunk_fallback(data.get("transcript", ""))

    stem = path.stem
    final = [build_chunk(rc, stem, i, file_meta) for i, rc in enumerate(raw_chunks)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for ch in final:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"    → {len(final)} chunks appended to {out_path.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", default="../../data/chunks/audio_chunks.jsonl")
    parser.add_argument("--lang",   default="hi-IN")
    parser.add_argument("--batch",  action="store_true")
    parser.add_argument("--kanda",  default="")
    parser.add_argument("--desc",   default="")
    args = parser.parse_args()

    inp  = Path(args.input)
    out  = Path(args.output)
    meta = {"lang": args.lang, "kanda": args.kanda, "description": args.desc}

    if args.batch or inp.is_dir():
        files = [f for f in inp.iterdir() if f.suffix.lower() in SUPPORTED]
        print(f"Found {len(files)} audio files")
        # Optional: load per-file metadata from metadata.json in the dir
        dir_meta_path = inp / "metadata.json"
        dir_meta = json.loads(dir_meta_path.read_text()) if dir_meta_path.exists() else {}
        for f in tqdm(files):
            fm = {**meta, **dir_meta.get(f.name, {})}
            process_file(f, out, args.lang, fm)
            time.sleep(0.3)
    elif inp.is_file():
        process_file(inp, out, args.lang, meta)
    else:
        print(f"Error: {inp} not found"); sys.exit(1)
