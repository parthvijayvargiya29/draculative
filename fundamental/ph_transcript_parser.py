"""
fundamental/ph_transcript_parser.py
======================================
Reads all .txt files from transcriptions/PH_macro/transcripts/ and extracts:

  1. Directional themes (BULLISH / BEARISH / NEUTRAL) per macro topic cluster
  2. Key entity mentions (tickers, currencies, commodities, macro indicators)
  3. A scored ``TranscriptTheme`` record per file + a merged corpus baseline

Topic clusters (7)
------------------
RATES       — Fed, interest rates, yields, FOMC, Treasury, bonds
DOLLAR      — DXY, USD, dollar strength/weakness, currency
EQUITIES    — S&P, Nasdaq, Russell, stocks, earnings, equity
COMMODITIES — gold, oil, energy, crude, copper, commodity
GEOPOLITICS — war, sanctions, tariffs, trade, geopolitical
INFLATION   — CPI, PCE, PPI, inflation, deflation, stagflation
LABOUR      — NFP, jobs, unemployment, payroll, labour market

Directionality scoring
-----------------------
Bullish keywords add +1, bearish keywords add -1 per sentence.
Final direction = sign of sum:  > 0 → BULLISH, < 0 → BEARISH, else NEUTRAL.

Usage
-----
    from fundamental.ph_transcript_parser import TranscriptCorpus, load_corpus
    corpus = load_corpus()              # reads all .txt files
    print(corpus.baseline_direction)   # overall US market directional call
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT          = Path(__file__).parent.parent
TRANSCRIPT_DIR = _ROOT / "transcriptions" / "PH_macro" / "transcripts"
CACHE_PATH     = _ROOT / "data" / "ph_macro_corpus.json"

# ── Keyword banks ─────────────────────────────────────────────────────────────

TOPIC_SEEDS: Dict[str, List[str]] = {
    "RATES": [
        "fed", "federal reserve", "fomc", "rate hike", "rate cut",
        "interest rate", "yield", "treasury", "10-year", "2-year",
        "bonds", "quantitative tightening", "qt", "quantitative easing",
        "qe", "pivot", "pause", "terminal rate", "neutral rate",
        "powell", "jerome", "dot plot", "ffr", "fed funds",
    ],
    "DOLLAR": [
        "dollar", "dxy", "usd", "greenback", "dollar strength",
        "dollar weakness", "dollar index", "currency", "forex", "fx",
        "euro", "yen", "pound", "cad", "aussie", "emerging market",
        "reserve currency", "dedollarisation",
    ],
    "EQUITIES": [
        "s&p", "sp500", "nasdaq", "russell", "dow jones", "djia",
        "equity", "equities", "stock market", "stocks", "earnings",
        "eps", "pe ratio", "valuation", "bull market", "bear market",
        "correction", "rally", "sell-off", "tech stocks", "growth stocks",
        "value stocks", "ipo", "buyback", "dividend",
    ],
    "COMMODITIES": [
        "gold", "silver", "oil", "crude", "brent", "wti", "energy",
        "copper", "commodity", "commodities", "natural gas", "coal",
        "aluminium", "platinum", "lithium", "agricultural",
    ],
    "GEOPOLITICS": [
        "war", "conflict", "sanctions", "tariff", "trade war",
        "geopolitical", "russia", "ukraine", "china", "taiwan",
        "middle east", "iran", "opec", "nato", "escalation",
        "de-escalation", "ceasefire", "election", "policy",
    ],
    "INFLATION": [
        "inflation", "cpi", "pce", "ppi", "core inflation",
        "deflation", "stagflation", "disinflation", "price stability",
        "cost of living", "wage growth", "wage inflation",
        "sticky inflation", "transitory",
    ],
    "LABOUR": [
        "nfp", "non-farm payroll", "payroll", "jobs", "unemployment",
        "job market", "labour market", "labor market", "jolts",
        "claims", "initial claims", "wage", "participation rate",
        "full employment", "hiring", "layoffs",
    ],
}

BULLISH_WORDS: List[str] = [
    "bullish", "upside", "higher", "rise", "rising", "rally", "buy",
    "long", "strength", "strong", "outperform", "positive", "growth",
    "expansion", "recovery", "supportive", "tailwind", "uptrend",
    "breakout", "acceleration", "improvement", "optimistic",
    "risk-on", "green light", "opportunity",
]

BEARISH_WORDS: List[str] = [
    "bearish", "downside", "lower", "fall", "falling", "selloff",
    "sell", "short", "weakness", "weak", "underperform", "negative",
    "contraction", "recession", "slowdown", "headwind", "downtrend",
    "breakdown", "deceleration", "deterioration", "pessimistic",
    "risk-off", "red flag", "caution", "concern", "warning",
    "crash", "collapse", "crisis", "downturn",
]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TopicScore:
    topic:       str
    score:       int          # raw signed count
    direction:   str          # BULLISH | BEARISH | NEUTRAL
    mentions:    int          # total keyword hits
    sentences:   List[str] = field(default_factory=list)   # top-3 evidence


@dataclass
class TranscriptTheme:
    filename:     str
    title:        str
    topics:       Dict[str, TopicScore] = field(default_factory=dict)
    overall_score: int = 0
    overall_direction: str = "NEUTRAL"
    word_count:   int = 0


@dataclass
class TranscriptCorpus:
    files:               List[TranscriptTheme]
    topic_scores:        Dict[str, int]        # sum across all files
    topic_directions:    Dict[str, str]        # BULLISH | BEARISH | NEUTRAL per topic
    baseline_direction:  str                   # overall US market call
    baseline_score:      int
    total_words:         int
    source_count:        int


# ── Core parsing logic ─────────────────────────────────────────────────────────

def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]


def _score_sentence(sent: str, topic: str) -> int:
    """Return +1 (bullish), -1 (bearish), or 0 for a sentence in context of topic."""
    s = sent.lower()
    bullish = any(w in s for w in BULLISH_WORDS)
    bearish = any(w in s for w in BEARISH_WORDS)
    if bullish and not bearish:
        return 1
    if bearish and not bullish:
        return -1
    return 0


def _direction(score: int) -> str:
    if score > 0:
        return "BULLISH"
    if score < 0:
        return "BEARISH"
    return "NEUTRAL"


def parse_transcript(path: Path) -> TranscriptTheme:
    """Parse a single .txt transcript file into a TranscriptTheme."""
    text  = path.read_text(encoding="utf-8", errors="replace")
    sents = _sentences(text)
    words = len(text.split())

    topics: Dict[str, TopicScore] = {}

    for topic, seeds in TOPIC_SEEDS.items():
        topic_score  = 0
        topic_sents: List[Tuple[int, str]] = []   # (score, sentence)
        mentions     = 0

        for sent in sents:
            s_lower = sent.lower()
            hit = any(seed in s_lower for seed in seeds)
            if hit:
                mentions += 1
                sc = _score_sentence(sent, topic)
                topic_score += sc
                topic_sents.append((sc, sent))

        # Keep top 3 evidence sentences (most polarised first)
        top3 = [s for _, s in sorted(topic_sents, key=lambda x: abs(x[0]), reverse=True)][:3]

        topics[topic] = TopicScore(
            topic=topic,
            score=topic_score,
            direction=_direction(topic_score),
            mentions=mentions,
            sentences=top3,
        )

    overall = sum(ts.score for ts in topics.values())
    title   = path.stem[6:].strip() if len(path.stem) > 5 else path.stem   # strip "01 - "

    return TranscriptTheme(
        filename=path.name,
        title=title,
        topics=topics,
        overall_score=overall,
        overall_direction=_direction(overall),
        word_count=words,
    )


def load_corpus(
    transcript_dir: Path = TRANSCRIPT_DIR,
    refresh: bool = False,
) -> TranscriptCorpus:
    """
    Parse all .txt transcripts in transcript_dir and return a merged
    TranscriptCorpus.  Results are cached in data/ph_macro_corpus.json.
    """
    # Cache check
    if CACHE_PATH.exists() and not refresh:
        try:
            data = json.loads(CACHE_PATH.read_text())
            # Rebuild from cache
            files = []
            for fd in data.get("files", []):
                topics = {k: TopicScore(**v) for k, v in fd["topics"].items()}
                files.append(TranscriptTheme(
                    filename=fd["filename"],
                    title=fd["title"],
                    topics=topics,
                    overall_score=fd["overall_score"],
                    overall_direction=fd["overall_direction"],
                    word_count=fd["word_count"],
                ))
            return TranscriptCorpus(
                files=files,
                topic_scores=data["topic_scores"],
                topic_directions=data["topic_directions"],
                baseline_direction=data["baseline_direction"],
                baseline_score=data["baseline_score"],
                total_words=data["total_words"],
                source_count=data["source_count"],
            )
        except Exception:
            pass  # fall through to re-parse

    txt_files = sorted(transcript_dir.glob("*.txt"))
    if not txt_files:
        print(f"  [WARN] No .txt files in {transcript_dir}. "
              "Run scripts/transcribe_ph_macro.py first.")
        # Return empty corpus so downstream code doesn't crash
        return TranscriptCorpus(
            files=[],
            topic_scores={t: 0 for t in TOPIC_SEEDS},
            topic_directions={t: "NEUTRAL" for t in TOPIC_SEEDS},
            baseline_direction="NEUTRAL",
            baseline_score=0,
            total_words=0,
            source_count=0,
        )

    themes = [parse_transcript(f) for f in txt_files]

    # Aggregate across files
    agg: Dict[str, int] = defaultdict(int)
    for theme in themes:
        for topic, ts in theme.topics.items():
            agg[topic] += ts.score

    topic_directions = {t: _direction(s) for t, s in agg.items()}
    baseline = sum(agg.values())

    corpus = TranscriptCorpus(
        files=themes,
        topic_scores=dict(agg),
        topic_directions=topic_directions,
        baseline_direction=_direction(baseline),
        baseline_score=baseline,
        total_words=sum(t.word_count for t in themes),
        source_count=len(themes),
    )

    # Cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": [
            {
                "filename": th.filename,
                "title": th.title,
                "topics": {k: asdict(v) for k, v in th.topics.items()},
                "overall_score": th.overall_score,
                "overall_direction": th.overall_direction,
                "word_count": th.word_count,
            }
            for th in themes
        ],
        "topic_scores": corpus.topic_scores,
        "topic_directions": corpus.topic_directions,
        "baseline_direction": corpus.baseline_direction,
        "baseline_score": corpus.baseline_score,
        "total_words": corpus.total_words,
        "source_count": corpus.source_count,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2))
    return corpus


if __name__ == "__main__":
    c = load_corpus(refresh=True)
    print(f"Parsed {c.source_count} transcripts ({c.total_words:,} words)")
    print(f"Baseline US market direction: {c.baseline_direction} (score={c.baseline_score})")
    for topic, direction in c.topic_directions.items():
        score = c.topic_scores.get(topic, 0)
        print(f"  {topic:<14} {direction:<8}  score={score:+d}")
