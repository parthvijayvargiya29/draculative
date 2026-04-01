"""
fundamental/ph_news_alignment.py
====================================
Measures how well today's live news ALIGNS with the transcript-derived
directional baseline.  The goal is to count SIMILARITIES, not differences.

Alignment methodology
----------------------
For each news item fetched from NewsFetcher:

  1. Run PHFundamentalModel.predict(headline + body) → DirectionalPrediction
  2. Compare predicted direction to the corpus baseline for the same topic cluster
  3. Compute a similarity score:
       MATCH        → +1.0   (same direction as transcript baseline)
       PARTIAL      → +0.5   (news is NEUTRAL but corpus has a strong view,
                               or directions differ by only one step)
       MISMATCH     → 0.0    (opposite direction)

  4. Aggregate per topic and overall:
       similarity_rate = matched / total_scored_items

Output: AlignmentReport dataclass

Usage
-----
    from fundamental.ph_news_alignment import compute_alignment
    report = compute_alignment()
    print(report.overall_similarity_rate)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from fundamental.ph_transcript_parser import load_corpus, TranscriptCorpus, TOPIC_SEEDS
from fundamental.ph_fundamental_model import PHFundamentalModel, get_model, DirectionalPrediction
from fundamental.news_fetcher import NewsFetcher, NewsItem


# ── Data structures ────────────────────────────────────────────────────────────

ALIGN_MATCH   = "MATCH"
ALIGN_PARTIAL = "PARTIAL"
ALIGN_MISS    = "MISMATCH"


@dataclass
class AlignedItem:
    news_title:        str
    news_category:     str
    news_date:         str
    news_source:       str
    predicted_dir:     str          # BULLISH | NEUTRAL | BEARISH from ML model
    predicted_conf:    float
    baseline_dir:      str          # corpus direction for relevant topic
    relevant_topic:    str
    alignment:         str          # MATCH | PARTIAL | MISMATCH
    alignment_score:   float        # 1.0 | 0.5 | 0.0


@dataclass
class TopicAlignment:
    topic:             str
    baseline_dir:      str
    items_scored:      int
    match_count:       int
    partial_count:     int
    mismatch_count:    int
    similarity_rate:   float


@dataclass
class AlignmentReport:
    generated_at:         str
    date_range:           str
    corpus_direction:     str          # overall transcript baseline
    corpus_score:         int
    corpus_sources:       int
    total_news_items:     int
    scored_items:         int
    match_count:          int
    partial_count:        int
    mismatch_count:       int
    overall_similarity_rate: float     # matches / scored  (0→1)
    topic_alignments:     Dict[str, TopicAlignment] = field(default_factory=dict)
    aligned_items:        List[AlignedItem] = field(default_factory=list)
    headline:             str = ""     # one-liner summary


# ── Alignment logic ────────────────────────────────────────────────────────────

def _alignment(predicted: str, baseline: str) -> Tuple[str, float]:
    """
    Compare predicted direction to baseline direction.
    Returns (alignment_label, score).
    """
    if predicted == baseline:
        return ALIGN_MATCH, 1.0
    if predicted == "NEUTRAL" or baseline == "NEUTRAL":
        return ALIGN_PARTIAL, 0.5
    # Both non-neutral but opposite
    return ALIGN_MISS, 0.0


def _map_news_to_topic(item: NewsItem) -> str:
    """Map a NewsItem category to the nearest topic cluster."""
    cat = item.category.upper()
    title = (item.title or "").lower()

    mapping = {
        "CPI":      "INFLATION",
        "FOMC":     "RATES",
        "NFP":      "LABOUR",
        "EARNINGS": "EQUITIES",
    }
    if cat in mapping:
        return mapping[cat]

    # Fall back to keyword scan
    for topic, seeds in TOPIC_SEEDS.items():
        if any(seed in title for seed in seeds):
            return topic

    return "EQUITIES"  # default: most US-market-relevant


def _build_input_text(item: NewsItem) -> str:
    """Combine available fields into a single string for ML prediction."""
    parts = [item.title or ""]
    if hasattr(item, "body") and item.body:
        parts.append(item.body)
    return " ".join(parts).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def compute_alignment(
    days_ahead: int = 1,
    corpus: Optional[TranscriptCorpus] = None,
    model: Optional[PHFundamentalModel] = None,
    news_items: Optional[List[NewsItem]] = None,
) -> AlignmentReport:
    """
    Compute the alignment report between live news and transcript baseline.

    Parameters
    ----------
    days_ahead   : how many calendar days of upcoming news to fetch (default 1 = today)
    corpus       : pre-loaded corpus (loads fresh if None)
    model        : pre-loaded model (loads/trains if None)
    news_items   : override news source (for testing)
    """
    if corpus is None:
        corpus = load_corpus()
    if model is None:
        model = get_model()

    # Fetch news
    if news_items is None:
        try:
            news_items = NewsFetcher().get_upcoming(days_ahead=days_ahead)
        except Exception as _e:
            news_items = []

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Per-topic accumulators
    topic_acc: Dict[str, Dict] = {
        t: {"match": 0, "partial": 0, "miss": 0, "total": 0}
        for t in TOPIC_SEEDS
    }

    aligned_items: List[AlignedItem] = []
    matches = partials = misses = 0

    for item in news_items:
        text  = _build_input_text(item)
        if not text.strip():
            continue

        pred: DirectionalPrediction = model.predict(text)
        topic = _map_news_to_topic(item)
        baseline_dir = corpus.topic_directions.get(topic, corpus.baseline_direction)

        label, score = _alignment(pred.direction, baseline_dir)

        aligned_items.append(AlignedItem(
            news_title=item.title or "",
            news_category=item.category,
            news_date=item.date,
            news_source=item.source,
            predicted_dir=pred.direction,
            predicted_conf=round(pred.confidence, 3),
            baseline_dir=baseline_dir,
            relevant_topic=topic,
            alignment=label,
            alignment_score=score,
        ))

        topic_acc[topic]["total"] += 1
        if label == ALIGN_MATCH:
            topic_acc[topic]["match"] += 1
            matches += 1
        elif label == ALIGN_PARTIAL:
            topic_acc[topic]["partial"] += 1
            partials += 1
        else:
            topic_acc[topic]["miss"] += 1
            misses += 1

    scored = matches + partials + misses
    sim_rate = (matches + 0.5 * partials) / scored if scored > 0 else 0.0

    topic_alignments: Dict[str, TopicAlignment] = {}
    for t, acc in topic_acc.items():
        total = acc["total"]
        if total == 0:
            continue
        t_sim = (acc["match"] + 0.5 * acc["partial"]) / total
        topic_alignments[t] = TopicAlignment(
            topic=t,
            baseline_dir=corpus.topic_directions.get(t, "NEUTRAL"),
            items_scored=total,
            match_count=acc["match"],
            partial_count=acc["partial"],
            mismatch_count=acc["miss"],
            similarity_rate=round(t_sim, 3),
        )

    # One-line headline summary
    if scored == 0:
        headline = "No scorable news items today."
    elif sim_rate >= 0.70:
        headline = (
            f"News strongly aligns with the {corpus.baseline_direction} transcript baseline "
            f"({sim_rate:.0%} similarity across {scored} items)."
        )
    elif sim_rate >= 0.45:
        headline = (
            f"Moderate news alignment with {corpus.baseline_direction} baseline "
            f"({sim_rate:.0%} similarity, {misses} mismatches)."
        )
    else:
        headline = (
            f"Low news alignment — market narrative shifting away from "
            f"{corpus.baseline_direction} baseline ({sim_rate:.0%} similarity)."
        )

    return AlignmentReport(
        generated_at=now_str,
        date_range=f"next {days_ahead} day(s)",
        corpus_direction=corpus.baseline_direction,
        corpus_score=corpus.baseline_score,
        corpus_sources=corpus.source_count,
        total_news_items=len(news_items),
        scored_items=scored,
        match_count=matches,
        partial_count=partials,
        mismatch_count=misses,
        overall_similarity_rate=round(sim_rate, 4),
        topic_alignments=topic_alignments,
        aligned_items=aligned_items,
        headline=headline,
    )


if __name__ == "__main__":
    report = compute_alignment()
    print(f"Corpus baseline: {report.corpus_direction}")
    print(f"News scored    : {report.scored_items}")
    print(f"Similarity rate: {report.overall_similarity_rate:.1%}")
    print(f"Headline       : {report.headline}")
