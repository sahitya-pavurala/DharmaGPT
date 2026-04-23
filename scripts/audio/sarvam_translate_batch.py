"""
Batch runner for Telugu audio translation pipeline.

Processes all audio files in a directory by invoking scripts/audio/sarvam_translate.py
for each file, with skip/resume behavior.

Examples:
    python scripts/audio/sarvam_translate_batch.py --input-dir downloads --output-dir data/chunks/all
    python scripts/audio/sarvam_translate_batch.py --input-dir downloads --chunks 0 --translator ollama
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma"}


def discover_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(files)


def build_output_name(audio_path: Path) -> str:
    safe_stem = audio_path.stem.replace("/", "_").replace("\\", "_")
    return f"telugu_chunks_{safe_stem}.jsonl"


def run_one(
    python_exe: str,
    audio_file: Path,
    output_file: Path,
    chunks: int,
    translator: str,
    ollama_model: str,
    ollama_url: str,
    indictrans2_model: str,
    manual_translations_dir: Path | None,
) -> tuple[bool, str]:
    script_path = REPO_ROOT / "scripts" / "audio" / "sarvam_translate.py"
    cmd = [
        python_exe,
        str(script_path),
        "--input",
        str(audio_file),
        "--output",
        str(output_file),
        "--chunks",
        str(chunks),
        "--translator",
        translator,
        "--ollama-model",
        ollama_model,
        "--ollama-url",
        ollama_url,
        "--indictrans2-model",
        indictrans2_model,
    ]

    if manual_translations_dir is not None:
        candidate = manual_translations_dir / f"{audio_file.stem}.jsonl"
        if candidate.exists():
            cmd.extend(["--manual-translations", str(candidate)])

    child_env = dict(os.environ)
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    run = subprocess.run(cmd, text=True, capture_output=True, env=child_env)
    if run.returncode == 0:
        return True, run.stdout.strip()
    err = (run.stderr or run.stdout or "").strip()
    return False, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Telugu audio translation runner")
    parser.add_argument("--input-dir", required=True, help="Directory with audio files")
    parser.add_argument(
        "--output-dir",
        default="data/chunks/batch",
        help="Directory for per-audio output JSONL files",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,
        help="Chunks per file. Use 0 or negative for all chunks in each file",
    )
    parser.add_argument(
        "--translator",
        choices=["auto", "skip", "anthropic", "ollama", "indictrans2"],
        default="auto",
        help="Translation backend",
    )
    parser.add_argument("--ollama-model", default="qwen2.5:7b", help="Ollama model")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument(
        "--indictrans2-model",
        default="ai4bharat/indictrans2-indic-en-dist-200M",
        help="IndicTrans2 model path or HF id",
    )
    parser.add_argument(
        "--manual-translations-dir",
        default=None,
        help="Optional directory containing <audio_stem>.jsonl manual translation files",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess files even if output exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_dir = Path(args.manual_translations_dir) if args.manual_translations_dir else None

    files = discover_audio_files(input_dir=input_dir, recursive=args.recursive)
    if not files:
        print(f"No supported audio files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} audio files")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Chunks:     {args.chunks} ({'all' if args.chunks <= 0 else 'limited'})")
    print(f"Translator: {args.translator}\n")

    python_exe = sys.executable
    results: list[dict] = []
    success = 0
    skipped = 0

    for i, audio_file in enumerate(files, start=1):
        out_file = output_dir / build_output_name(audio_file)

        if out_file.exists() and not args.overwrite:
            print(f"[{i}/{len(files)}] SKIP  {audio_file.name} (output exists)")
            skipped += 1
            results.append(
                {
                    "audio": str(audio_file),
                    "output": str(out_file),
                    "status": "skipped_existing",
                }
            )
            continue

        print(f"[{i}/{len(files)}] RUN   {audio_file.name}")
        ok, details = run_one(
            python_exe=python_exe,
            audio_file=audio_file,
            output_file=out_file,
            chunks=args.chunks,
            translator=args.translator,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            indictrans2_model=args.indictrans2_model,
            manual_translations_dir=manual_dir,
        )

        if ok:
            success += 1
            print(f"         OK   -> {out_file.name}")
            results.append(
                {
                    "audio": str(audio_file),
                    "output": str(out_file),
                    "status": "ok",
                }
            )
        else:
            print("         FAIL")
            print(f"         {details.splitlines()[-1] if details else 'Unknown error'}")
            results.append(
                {
                    "audio": str(audio_file),
                    "output": str(out_file),
                    "status": "failed",
                    "error": details,
                }
            )

    summary = {
        "total": len(files),
        "ok": success,
        "skipped_existing": skipped,
        "failed": len(files) - success - skipped,
        "translator": args.translator,
        "chunks": args.chunks,
        "results": results,
    }

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nBatch complete")
    print(f"  OK:      {summary['ok']}")
    print(f"  Skipped: {summary['skipped_existing']}")
    print(f"  Failed:  {summary['failed']}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
