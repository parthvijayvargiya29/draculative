"""
fundamental/ph_daily_report.py
================================
Generates a clean daily report combining:
  1. Transcript directional baseline (from corpus)
  2. Today's news alignment score
  3. Per-topic breakdown
  4. Overall US market directional call

Output files
------------
  reports/ph_macro/YYYY-MM-DD_ph_fundamental_report.md   ← human-readable
  reports/ph_macro/YYYY-MM-DD_ph_fundamental_report.json ← machine-readable

Also prints a compact console summary.

Usage
-----
    # From repo root
    python -m fundamental.ph_daily_report

    # Programmatic
    from fundamental.ph_daily_report import generate_report
    generate_report()
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT      = Path(__file__).parent.parent
REPORT_DIR = _ROOT / "reports" / "ph_macro"
sys.path.insert(0, str(_ROOT))

from fundamental.ph_transcript_parser import load_corpus, TranscriptCorpus, TOPIC_SEEDS
from fundamental.ph_fundamental_model import PHFundamentalModel, get_model, ModelMeta
from fundamental.ph_news_alignment import (
    compute_alignment,
    AlignmentReport,
    AlignedItem,
    ALIGN_MATCH,
    ALIGN_PARTIAL,
    ALIGN_MISS,
)


# ── ANSI / emoji helpers ──────────────────────────────────────────────────────

_DIR_EMOJI = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️ "}
_ALIGN_EMOJI = {ALIGN_MATCH: "✅", ALIGN_PARTIAL: "⚠️ ", ALIGN_MISS: "❌"}
_SIM_GRADE = {
    (0.70, 1.01): ("STRONG ALIGNMENT",   "🟢"),
    (0.45, 0.70): ("MODERATE ALIGNMENT", "🟡"),
    (0.00, 0.45): ("LOW ALIGNMENT",      "🔴"),
}


def _sim_grade(rate: float):
    for (lo, hi), (label, dot) in _SIM_GRADE.items():
        if lo <= rate < hi:
            return label, dot
    return "N/A", "⚪"


def _dir_emoji(d: str) -> str:
    return _DIR_EMOJI.get(d, "")


# ── Markdown generator ────────────────────────────────────────────────────────

def _build_markdown(
    date_str:  str,
    corpus:    TranscriptCorpus,
    report:    AlignmentReport,
    meta:      Optional[ModelMeta],
) -> str:
    grade_label, grade_dot = _sim_grade(report.overall_similarity_rate)

    lines = [
        f"# PH Macro Daily Report — {date_str}",
        f"",
        f"> **{report.headline}**",
        f"",
        f"---",
        f"",
        f"## 1. Transcript Directional Baseline",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Source transcripts | {corpus.source_count} |",
        f"| Total words analysed | {corpus.total_words:,} |",
        f"| **Overall US market call** | {_dir_emoji(corpus.baseline_direction)} **{corpus.baseline_direction}** |",
        f"| Baseline score (signed) | {corpus.baseline_score:+d} |",
        f"",
        f"### Per-Topic Directional Baseline",
        f"",
        f"| Topic | Direction | Score |",
        f"|-------|-----------|-------|",
    ]
    for topic in TOPIC_SEEDS:
        t_dir   = corpus.topic_directions.get(topic, "NEUTRAL")
        t_score = corpus.topic_scores.get(topic, 0)
        lines.append(f"| {topic} | {_dir_emoji(t_dir)} {t_dir} | {t_score:+d} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## 2. News Alignment Score",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Date range | {report.date_range} |",
        f"| News items fetched | {report.total_news_items} |",
        f"| Items scored | {report.scored_items} |",
        f"| ✅ MATCH | {report.match_count} |",
        f"| ⚠️  PARTIAL | {report.partial_count} |",
        f"| ❌ MISMATCH | {report.mismatch_count} |",
        f"| **Similarity rate** | **{report.overall_similarity_rate:.1%}** |",
        f"| **Alignment grade** | {grade_dot} **{grade_label}** |",
        f"",
    ]

    if report.topic_alignments:
        lines += [
            f"### Per-Topic Alignment",
            f"",
            f"| Topic | Baseline | Scored | Matches | Partials | Misses | Similarity |",
            f"|-------|----------|--------|---------|----------|--------|------------|",
        ]
        for ta in sorted(report.topic_alignments.values(), key=lambda x: -x.similarity_rate):
            lines.append(
                f"| {ta.topic} | {_dir_emoji(ta.baseline_dir)} {ta.baseline_dir} "
                f"| {ta.items_scored} | {ta.match_count} | {ta.partial_count} "
                f"| {ta.mismatch_count} | {ta.similarity_rate:.0%} |"
            )
        lines.append("")

    if report.aligned_items:
        # Show top-10 news items (sorted by score desc)
        top_items = sorted(report.aligned_items, key=lambda x: -x.alignment_score)[:10]
        lines += [
            f"---",
            f"",
            f"## 3. Top Aligned News Items",
            f"",
            f"| # | Date | Title | Predicted | Baseline | Alignment |",
            f"|---|------|-------|-----------|----------|-----------|",
        ]
        for i, item in enumerate(top_items, 1):
            title_trunc = item.news_title[:55] + "…" if len(item.news_title) > 55 else item.news_title
            a_emoji     = _ALIGN_EMOJI.get(item.alignment, "")
            lines.append(
                f"| {i} | {item.news_date} | {title_trunc} "
                f"| {item.predicted_dir} | {item.baseline_dir} "
                f"| {a_emoji} {item.alignment} |"
            )
        lines.append("")

    # ML model metadata
    if meta and meta.n_chunks > 0:
        lines += [
            f"---",
            f"",
            f"## 4. Model Metadata",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Trained at | {meta.trained_at} |",
            f"| Training chunks | {meta.n_chunks:,} |",
            f"| Source transcripts | {meta.n_transcripts} |",
            f"| CV accuracy | {meta.accuracy:.1%} |",
            f"| Vocabulary size | {meta.vocab_size:,} |",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## 5. Directional Summary",
        f"",
        f"| Component | Signal |",
        f"|-----------|--------|",
        f"| Transcript baseline | {_dir_emoji(corpus.baseline_direction)} {corpus.baseline_direction} |",
        f"| News flow today | {grade_dot} {grade_label} ({report.overall_similarity_rate:.0%}) |",
        f"| **Final US market call** | {_dir_emoji(corpus.baseline_direction)} **{corpus.baseline_direction}** |",
        f"",
        f"---",
        f"*Generated by Draculative Alpha Engine · PH Macro Module · {date_str}*",
    ]

    return "\n".join(lines)


# ── Console summary ───────────────────────────────────────────────────────────

def _print_console_summary(corpus: TranscriptCorpus, report: AlignmentReport) -> None:
    grade_label, grade_dot = _sim_grade(report.overall_similarity_rate)

    print()
    print("=" * 70)
    print("  PH MACRO DAILY REPORT")
    print("=" * 70)
    print(f"  Corpus : {corpus.source_count} transcripts, {corpus.total_words:,} words")
    print(f"  Baseline direction : {_dir_emoji(corpus.baseline_direction)} {corpus.baseline_direction}  "
          f"(score={corpus.baseline_score:+d})")
    print()
    print(f"  News scored : {report.scored_items} items")
    print(f"  ✅ Match    : {report.match_count}")
    print(f"  ⚠️  Partial  : {report.partial_count}")
    print(f"  ❌ Mismatch : {report.mismatch_count}")
    print(f"  Similarity  : {report.overall_similarity_rate:.1%}  {grade_dot} {grade_label}")
    print()
    print(f"  Headline: {report.headline}")
    print("=" * 70)

    if report.topic_alignments:
        print("\n  Per-topic similarity:")
        for ta in sorted(report.topic_alignments.values(), key=lambda x: -x.similarity_rate):
            bar_len = int(ta.similarity_rate * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {ta.topic:<14} {bar}  {ta.similarity_rate:.0%}  "
                  f"(baseline: {ta.baseline_dir})")
    print()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    days_ahead: int = 1,
    refresh_corpus: bool = False,
    refresh_model: bool = False,
    quiet: bool = False,
) -> AlignmentReport:
    """
    Full pipeline: load corpus → train/load model → fetch news → align → write report.
    Returns the AlignmentReport.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    if not quiet:
        print("  Loading transcript corpus …")
    corpus = load_corpus(refresh=refresh_corpus)

    if not quiet:
        print("  Loading / training model …")
    model = PHFundamentalModel()
    meta  = model.train(refresh=refresh_model)

    if not quiet:
        print("  Computing news alignment …")
    report = compute_alignment(
        days_ahead=days_ahead,
        corpus=corpus,
        model=model,
    )

    # ── Write markdown ──────────────────────────────────────────────────────
    md_path = REPORT_DIR / f"{date_str}_ph_fundamental_report.md"
    md_text = _build_markdown(date_str, corpus, report, meta)
    md_path.write_text(md_text, encoding="utf-8")

    # ── Write JSON ──────────────────────────────────────────────────────────
    json_path = REPORT_DIR / f"{date_str}_ph_fundamental_report.json"
    payload = {
        "date": date_str,
        "corpus": {
            "source_count":        corpus.source_count,
            "total_words":         corpus.total_words,
            "baseline_direction":  corpus.baseline_direction,
            "baseline_score":      corpus.baseline_score,
            "topic_directions":    corpus.topic_directions,
            "topic_scores":        corpus.topic_scores,
        },
        "alignment": {
            "generated_at":           report.generated_at,
            "scored_items":           report.scored_items,
            "match_count":            report.match_count,
            "partial_count":          report.partial_count,
            "mismatch_count":         report.mismatch_count,
            "overall_similarity_rate": report.overall_similarity_rate,
            "headline":               report.headline,
            "topic_alignments": {
                t: asdict(ta) for t, ta in report.topic_alignments.items()
            },
            "aligned_items": [asdict(i) for i in report.aligned_items],
        },
        "model": {
            "trained_at":    meta.trained_at if meta else "",
            "n_chunks":      meta.n_chunks if meta else 0,
            "accuracy":      meta.accuracy if meta else 0.0,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not quiet:
        _print_console_summary(corpus, report)
        print(f"  📄 Markdown → {md_path.relative_to(_ROOT)}")
        print(f"  📊 JSON     → {json_path.relative_to(_ROOT)}")

    return report


if __name__ == "__main__":
    generate_report()
