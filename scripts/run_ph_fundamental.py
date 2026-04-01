#!/usr/bin/env python3
"""
scripts/run_ph_fundamental.py
===============================
CLI orchestrator for the PH Macro Fundamental System.

Commands
--------
transcribe   Download and transcribe the 15 PH macro YouTube videos.
             Outputs: transcriptions/PH_macro/audio/  + transcripts/

train        Parse transcripts and train the directional ML model.
             Outputs: data/ph_macro_corpus.json
                      data/ph_fundamental_model.pkl

report       Fetch today's news, compute alignment, write daily report.
             Outputs: reports/ph_macro/YYYY-MM-DD_ph_fundamental_report.md
                      reports/ph_macro/YYYY-MM-DD_ph_fundamental_report.json

all          Run transcribe → train → report in sequence.

Usage
-----
    python scripts/run_ph_fundamental.py transcribe
    python scripts/run_ph_fundamental.py train
    python scripts/run_ph_fundamental.py report
    python scripts/run_ph_fundamental.py all

    python scripts/run_ph_fundamental.py report --days 3
    python scripts/run_ph_fundamental.py train   --refresh
    python scripts/run_ph_fundamental.py all     --model large
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

VENV_PYTHON  = _ROOT / ".venv" / "bin" / "python"
_PY          = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

TRANSCRIPT_DIR = _ROOT / "transcriptions" / "PH_macro" / "transcripts"
REPORT_DIR     = _ROOT / "reports" / "ph_macro"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(script: str, *extra_args: str) -> int:
    """Run a Python script in a subprocess. Returns exit code."""
    cmd = [_PY, script, *extra_args]
    print(f"\n  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def _section(title: str) -> None:
    print()
    print("─" * 70)
    print(f"  {title}")
    print("─" * 70)


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_transcribe(model: str = "medium", skip_download: bool = False) -> int:
    _section("STEP 1 / 3 — Transcribe PH Macro Videos")
    args = ["scripts/transcribe_ph_macro.py", "--model", model]
    if skip_download:
        args.append("--skip-download")
    return _run(*args)


def cmd_train(refresh: bool = False) -> int:
    _section("STEP 2 / 3 — Parse Transcripts & Train Directional Model")
    n_txt = len(list(TRANSCRIPT_DIR.glob("*.txt"))) if TRANSCRIPT_DIR.exists() else 0
    if n_txt == 0:
        print(f"  ⚠️  No transcripts found in {TRANSCRIPT_DIR.relative_to(_ROOT)}.")
        print("  Run 'transcribe' first, or pass --skip-download if audio exists.")
        return 1

    print(f"  Found {n_txt} transcript(s) in {TRANSCRIPT_DIR.relative_to(_ROOT)}")

    # Import inline so the venv interpreter is guaranteed
    from fundamental.ph_transcript_parser import load_corpus
    from fundamental.ph_fundamental_model import PHFundamentalModel

    print("  Parsing corpus …")
    corpus = load_corpus(refresh=refresh)
    print(f"  ✅ Corpus: {corpus.source_count} transcripts, "
          f"{corpus.total_words:,} words, "
          f"baseline={corpus.baseline_direction}")

    print("  Training ML model …")
    model_obj = PHFundamentalModel()
    meta = model_obj.train(refresh=refresh)

    if meta.n_chunks == 0:
        print("  ⚠️  No training chunks — rule-based fallback will be used.")
    else:
        print(f"  ✅ Model: {meta.n_chunks} chunks, CV acc={meta.accuracy:.1%}, "
              f"vocab={meta.vocab_size:,}")

    # Print directional baseline
    print()
    print("  ┌─────────────────────────────────────┐")
    print(f"  │  US Market Baseline: {corpus.baseline_direction:<12}       │")
    print("  ├──────────────┬──────────┬───────────┤")
    print("  │ Topic        │ Dir.     │ Score     │")
    print("  ├──────────────┼──────────┼───────────┤")
    for topic, direction in corpus.topic_directions.items():
        score = corpus.topic_scores.get(topic, 0)
        print(f"  │ {topic:<12} │ {direction:<8} │ {score:>+6}    │")
    print("  └──────────────┴──────────┴───────────┘")
    return 0


def cmd_report(days: int = 1, refresh_corpus: bool = False, refresh_model: bool = False) -> int:
    _section("STEP 3 / 3 — Generate Daily Alignment Report")
    from fundamental.ph_daily_report import generate_report
    report = generate_report(
        days_ahead=days,
        refresh_corpus=refresh_corpus,
        refresh_model=refresh_model,
    )
    return 0


def cmd_all(
    model: str = "medium",
    skip_download: bool = False,
    refresh: bool = False,
    days: int = 1,
) -> int:
    """Run the full pipeline: transcribe → train → report."""
    print()
    print("=" * 70)
    print("  PH MACRO FUNDAMENTAL SYSTEM — FULL PIPELINE")
    print("=" * 70)
    t0 = time.time()

    rc = cmd_transcribe(model=model, skip_download=skip_download)
    if rc != 0:
        print(f"  ⚠️  Transcription step returned exit code {rc} — continuing …")

    rc = cmd_train(refresh=refresh)
    if rc != 0:
        print(f"  ❌ Training step failed (exit {rc}). Aborting report generation.")
        return rc

    rc = cmd_report(days=days, refresh_corpus=refresh, refresh_model=False)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"  PIPELINE COMPLETE  ({elapsed:.0f}s)")
    print(f"  Reports: {REPORT_DIR.relative_to(_ROOT)}/")
    print("=" * 70)
    return rc


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PH Macro Fundamental System — transcribe | train | report | all",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ph_fundamental.py transcribe
  python scripts/run_ph_fundamental.py train --refresh
  python scripts/run_ph_fundamental.py report --days 3
  python scripts/run_ph_fundamental.py all --model large
        """,
    )
    parser.add_argument(
        "command",
        choices=["transcribe", "train", "report", "all"],
        help="Which stage to run",
    )
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model (used by 'transcribe' and 'all', default: medium)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip yt-dlp; only transcribe existing audio files",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-parse corpus and re-train model even if cached",
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Days ahead of news to include in the alignment report (default: 1)",
    )
    args = parser.parse_args()

    if args.command == "transcribe":
        rc = cmd_transcribe(model=args.model, skip_download=args.skip_download)
    elif args.command == "train":
        rc = cmd_train(refresh=args.refresh)
    elif args.command == "report":
        rc = cmd_report(days=args.days, refresh_corpus=args.refresh)
    elif args.command == "all":
        rc = cmd_all(
            model=args.model,
            skip_download=args.skip_download,
            refresh=args.refresh,
            days=args.days,
        )
    else:
        parser.print_help()
        rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
