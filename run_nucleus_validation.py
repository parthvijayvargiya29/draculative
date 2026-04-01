#!/usr/bin/env python3
"""
run_nucleus_validation.py — Entry point for nucleus engine validation.

Run this file to execute all 5 validation checks against 2 years of
real market data and produce a gate pass/fail result.

EXIT CODES:
  0 = Gate PASS (all 5 checks passed)
  1 = Gate FAIL (one or more checks failed — see recalibration output)

USAGE:
  # First run — downloads 2 years of data (~5 min):
  python run_nucleus_validation.py --plot

  # Subsequent runs — uses disk cache (~30 sec):
  python run_nucleus_validation.py --report-only --plot

  # With new ICT transcripts:
  python run_nucleus_validation.py --transcripts ./new_transcripts/ --report-only

  # With a Notion export:
  python run_nucleus_validation.py --notion ./my_notes.md --report-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nucleus_validation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run nucleus engine validation against 2 years of Alpaca daily bars."
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Skip data fetch (use disk cache) and go straight to report generation.",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Show matplotlib charts after validation (distribution + score timeseries).",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force-refresh all cached data from the API.",
    )
    p.add_argument(
        "--transcripts", type=str, default=None, metavar="DIR",
        help="Path to a directory of new .txt transcript files to parse and log.",
    )
    p.add_argument(
        "--notion", type=str, default=None, metavar="FILE",
        help="Path to a Notion markdown export to parse for ICT concepts.",
    )
    p.add_argument(
        "--out-dir", type=str, default="reports/nucleus",
        help="Directory to write JSON reports (default: reports/nucleus/).",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    return p.parse_args()


def load_data(report_only: bool, refresh: bool) -> dict:
    """Load universe data. Raises on fatal failure."""
    from simulation.alpaca_data_fetcher import AlpacaDataFetcher
    fetcher = AlpacaDataFetcher()

    if report_only:
        # Force-load from cache only (no API call)
        logger.info("--report-only: loading from disk cache (no API call)")
        import os
        os.environ["_ALPACA_CACHE_ONLY"] = "1"   # signal to fetcher

    logger.info("Loading universe data…")
    data = fetcher.load_universe(refresh=refresh)
    logger.info("Universe loaded: %d symbols", len(data))
    return data


def process_transcripts(transcript_dir: str) -> None:
    """Parse new ICT transcripts and append to concept_library.json."""
    path = Path(transcript_dir)
    if not path.exists():
        logger.warning("--transcripts path not found: %s", transcript_dir)
        return

    txt_files = list(path.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", transcript_dir)
        return

    logger.info("Processing %d new transcript(s) from %s", len(txt_files), transcript_dir)
    try:
        from fundamental.transcript_parser import process_transcript, deduplicate
        new_concepts = []
        for f in txt_files:
            concepts = process_transcript(str(f))
            new_concepts.extend(concepts)
            logger.info("  Parsed: %s — %d concepts", f.name, len(concepts))

        deduped = deduplicate(new_concepts)
        logger.info("New unique concepts: %d", len(deduped))

        # Append to existing concept_library.json
        lib_path = Path("transcripts/processed/concept_library.json")
        existing = []
        if lib_path.exists():
            import json as _json
            with open(lib_path) as f:
                existing = _json.load(f)

        from dataclasses import asdict
        merged = existing + [asdict(c) for c in deduped]
        import json as _json
        with open(lib_path, "w") as f:
            _json.dump(merged, f, indent=2)
        logger.info("Concept library updated: %d total concepts", len(merged))

    except Exception as exc:
        logger.error("Transcript processing failed: %s", exc)


def process_notion(notion_file: str) -> None:
    """Parse a Notion markdown export and log extracted ICT concepts."""
    path = Path(notion_file)
    if not path.exists():
        logger.warning("--notion file not found: %s", notion_file)
        return

    logger.info("Parsing Notion export: %s", path.name)
    try:
        from fundamental.transcript_parser import process_transcript, deduplicate
        concepts = process_transcript(str(path))
        deduped  = deduplicate(concepts)
        logger.info("Notion export — %d unique concepts extracted", len(deduped))

        # Save to a separate notion concepts file
        out_path = Path("transcripts/processed/notion_concepts.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        import json as _json
        existing = []
        if out_path.exists():
            with open(out_path) as f:
                existing = _json.load(f)
        merged = existing + [asdict(c) for c in deduped]
        with open(out_path, "w") as f:
            _json.dump(merged, f, indent=2)
        logger.info("Notion concepts saved: %s (%d total)", out_path, len(merged))

    except Exception as exc:
        logger.error("Notion processing failed: %s", exc)


def plot_results(report_dict: dict) -> None:
    """Generate and display validation charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend; swap to "TkAgg" for window
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np

        dist   = report_dict.get("distribution", {})
        checks = {c["name"]: c for c in report_dict.get("checks", [])}

        # ── Figure layout ───────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Nucleus Engine Validation", fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        # Panel 1: Distribution bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        if dist:
            labels = list(dist.keys())
            values = [v * 100 for v in dist.values()]
            colors = ["#e74c3c" if v > 35 else "#2ecc71" for v in values]
            bars   = ax1.barh(labels, values, color=colors)
            ax1.axvline(35, color="red", linestyle="--", linewidth=1, label="35% gate")
            ax1.set_xlabel("% of Days as Dominant Nucleus")
            ax1.set_title("Nucleus Distribution")
            ax1.legend(fontsize=8)
            for bar, val in zip(bars, values):
                ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         f"{val:.1f}%", va="center", fontsize=8)

        # Panel 2: Check summary (scorecard)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis("off")
        gate_pass = report_dict.get("gate_pass", False)
        gate_text = "✅ GATE PASS" if gate_pass else "❌ GATE FAIL"
        gate_color = "#2ecc71" if gate_pass else "#e74c3c"

        row_labels = []
        row_values = []
        row_colors = []
        for c in report_dict.get("checks", []):
            row_labels.append(c["name"])
            row_values.append(f"{c['value']:.3f} / gate={c['gate']:.3f}")
            row_colors.append("#2ecc71" if c["passed"] else "#e74c3c")

        table_data = [[l, v] for l, v in zip(row_labels, row_values)]
        tbl = ax2.table(
            cellText=table_data,
            colLabels=["Check", "Value / Gate"],
            cellLoc="left",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for i, color in enumerate(row_colors, 1):
            tbl[(i, 0)].set_facecolor(color + "44")
            tbl[(i, 1)].set_facecolor(color + "44")

        ax2.set_title(gate_text, fontsize=12, color=gate_color, fontweight="bold")

        # Panel 3: Per-nucleus persistence
        pers_detail = {}
        for c in report_dict.get("checks", []):
            if c["name"] == "Persistence":
                pers_detail = c.get("detail", {})
                break
        ax3 = fig.add_subplot(gs[1, 0])
        if pers_detail:
            ax3.bar(pers_detail.keys(), pers_detail.values(), color="#3498db")
            ax3.axhline(2,  color="green", linestyle="--", linewidth=1, label="Min (2)")
            ax3.axhline(20, color="red",   linestyle="--", linewidth=1, label="Max (20)")
            ax3.set_ylabel("Avg Persistence (bars)")
            ax3.set_title("Persistence by Nucleus Type")
            ax3.tick_params(axis="x", rotation=45)
            ax3.legend(fontsize=8)

        # Panel 4: Score spread histogram
        ax4 = fig.add_subplot(gs[1, 1])
        raw_results = report_dict.get("raw_results", [])
        if raw_results:
            top_scores = [r["score"] for r in raw_results]
            ax4.hist(top_scores, bins=30, color="#9b59b6", edgecolor="white", alpha=0.8)
            ax4.axvline(0.65, color="orange", linestyle="--", linewidth=1, label="Strong signal (0.65)")
            ax4.set_xlabel("Winning Nucleus Score")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Score Distribution (winning nucleus)")
            ax4.legend(fontsize=8)

        # Save chart
        out_dir = Path("reports/nucleus")
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        chart_path = out_dir / f"nucleus_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved: %s", chart_path)
        print(f"\n  📊  Chart saved → {chart_path}")

    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)


def main() -> int:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── Optional: process new ICT transcripts ─────────────────────────────
    if args.transcripts:
        process_transcripts(args.transcripts)

    # ── Optional: process Notion export ───────────────────────────────────
    if args.notion:
        process_notion(args.notion)

    # ── Load market data ──────────────────────────────────────────────────
    try:
        data = load_data(report_only=args.report_only, refresh=args.refresh)
    except RuntimeError as exc:
        logger.error("Data load failed: %s", exc)
        logger.error(
            "If you don't have Alpaca API keys, install yfinance as fallback:\n"
            "  pip install yfinance"
        )
        return 1

    # ── Run validation ─────────────────────────────────────────────────────
    try:
        from core.nucleus_validator import NucleusValidator
        validator = NucleusValidator(data)
        report    = validator.run()
    except Exception as exc:
        logger.exception("Validation failed with unexpected error: %s", exc)
        return 1

    # ── Plot ────────────────────────────────────────────────────────────────
    if args.plot:
        # Load the saved JSON to pass to plotter (avoids serialisation issues)
        report_dir = Path(args.out_dir)
        json_files = sorted(report_dir.glob("nucleus_validation_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if json_files:
            with open(json_files[0]) as f:
                report_dict = json.load(f)
            plot_results(report_dict)

    # ── Exit code ────────────────────────────────────────────────────────────
    return 0 if report.gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
