"""
transcript_parser.py — Section 1 Transcript Processing Pipeline

Reads raw .txt transcripts, strips noise, extracts structured trading concepts,
deduplicates across transcripts, and writes transcripts/processed/concept_library.json.

Usage:
    python fundamental/transcript_parser.py
    python fundamental/transcript_parser.py --dir transcriptions/playlist_mVS4OSyj0Zg/transcripts
"""
from __future__ import annotations

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# ── NOISE PATTERNS ────────────────────────────────────────────────────────────
# Applied at SENTENCE level only. Multi-word technical phrases are never stripped
# by keyword alone — full sentence context is checked first.

NOISE_PHRASES: List[str] = [
    r"\blike\s+and\s+subscri",
    r"\bsubscri\w+\b",
    r"\bfollow\s+the\s+social",
    r"\bdrop\s+your\s+country",
    r"\bcomment\s+below\b",
    r"\bmuch\s+love\b",
    r"\bpeace\s+out\b",
    r"\bsee\s+you\s+(next|later)\b",
    r"\bcatch\s+you\b",
    r"\bthanks\s+for\s+watching\b",
    r"\bmake\s+sure\s+to\s+like\b",
    r"\bfollow\s+(our\s+)?socials\b",
    r"\bcheck\s+out\s+(the\s+)?merch\b",
    r"\bsponsor\b",
    r"\bdiscord\s+link\b",
    r"\bpatreon\b",
]

NOISE_RE = re.compile("|".join(NOISE_PHRASES), re.IGNORECASE)

# Sentence-ending punctuation splitter
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ── CONCEPT KEYWORD MAP ───────────────────────────────────────────────────────
# Maps concept categories to trigger keyword sets.
# Any sentence that contains one of these keywords is a CANDIDATE for extraction.

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "STRUCTURE": [
        "market structure", "break of structure", "bos", "choch",
        "change of character", "swing high", "swing low", "higher high",
        "higher low", "lower high", "lower low", "hh", "hl", "lh", "ll",
        "internal structure", "external structure", "mss", "market structure shift",
    ],
    "LIQUIDITY": [
        "liquidity", "ssl", "bsl", "buy side", "sell side", "equal highs",
        "equal lows", "stop hunt", "stop loss hunt", "sweep", "inducement",
        "liquidity void", "pool", "draw on liquidity", "drawn liquidity",
        "inducement level", "trendline liquidity",
    ],
    "FVG": [
        "fair value gap", "fvg", "imbalance", "inefficiency",
        "inversion fvg", "inverted fvg", "volume imbalance",
        "price inefficiency", "single candle", "gap fill",
        "cell side imbalance",
    ],
    "ORDER_BLOCK": [
        "order block", "ob", "breaker block", "breaker", "mitigation block",
        "propulsion block", "rejection block", "consolidation break",
        "bullish order block", "bearish order block",
    ],
    "PREMIUM_DISCOUNT": [
        "premium", "discount", "equilibrium", "ote", "optimal trade entry",
        "61.8", "78.6", "fibonacci", "fib", "retracement",
        "above equilibrium", "below equilibrium",
    ],
    "TIME": [
        "killzone", "kill zone", "london", "new york", "asian session",
        "session", "time of day", "tuesday", "wednesday", "thursday",
        "day of week", "macro", "accumulation manipulation distribution",
        "amd", "power of three", "opening range", "midnight open",
    ],
    "FUNDAMENTAL": [
        "dxy", "dollar", "interest rate", "fed", "fomc", "cpi", "nfp",
        "macro", "institutional", "correlation", "bond", "gold", "oil",
        "risk off", "risk on", "geopolit",
    ],
    "ENTRY_MECHANIC": [
        "entry", "trigger", "inversion", "displacement", "confirmation",
        "engulf", "limit order", "market order", "entry rule", "enter",
        "execute", "set up", "setup", "a plus", "a+ setup",
    ],
    "EXIT_MECHANIC": [
        "take profit", "tp", "target", "exit", "trailing stop",
        "draw on liquidity", "drawn liquidity", "objective", "profit target",
    ],
    "RISK_MANAGEMENT": [
        "risk", "stop loss", "position size", "percent risk", "drawdown",
        "max loss", "profit factor", "rr", "risk reward", "risk management",
        "one percent", "half percent", "micros", "minis", "buffer",
        "payout", "prop firm", "funded", "eval",
    ],
    "BIAS_FRAMEWORK": [
        "bias", "daily bias", "weekly bias", "directional", "bullish bias",
        "bearish bias", "top down", "higher time frame", "htf", "ltf",
    ],
    "CONFLUENCE": [
        "confluence", "stack", "multiple", "combine", "align", "converge",
        "pd array", "array",
    ],
    "CORRELATION": [
        "smt", "smart money divergence", "divergence", "correlated",
        "correlation", "dxy", "inverse", "intermarket",
    ],
    "ANOMALY": [
        "statistic", "historically", "tend to", "tendency", "pattern",
        "backtest", "empirically", "observation", "80%", "90%", "percent of the time",
    ],
}


def clean_text(raw: str) -> str:
    """Remove whisper artefacts (hyphen-split words, excess whitespace)."""
    text = re.sub(r"-\n\s*", "", raw)       # re-join hyphenated line breaks
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def is_noise(sentence: str) -> bool:
    """
    Returns True only if the sentence matches a noise pattern AND contains
    no technical keyword from any category. This prevents false-positive removal.
    """
    if not NOISE_RE.search(sentence):
        return False
    # Check if ANY technical keyword is also present — if so, keep it.
    lower = sentence.lower()
    for keywords in CATEGORY_KEYWORDS.values():
        if any(kw in lower for kw in keywords):
            return False
    return True


def classify_sentence(sentence: str) -> str:
    """Returns the best-matching ConceptCategory name, or 'NOISE'."""
    lower = sentence.lower()
    scores: Dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[category] = score
    if not scores:
        return "NOISE"
    return max(scores, key=lambda k: scores[k])


def extract_mechanical_rule(sentence: str, category: str) -> str:
    """
    Produce a concise if/then mechanical rule from a raw sentence.
    For now this is a lightweight heuristic; the full NLP pass happens
    in process_transcript() where context windows are used.
    """
    sentence = sentence.strip().rstrip(".")
    # Already has if/then structure
    if re.search(r"\bif\b.+\bthen\b", sentence, re.IGNORECASE):
        return sentence
    # Contains "when X, Y"
    m = re.match(r"when\s+(.+?),\s*(.+)", sentence, re.IGNORECASE)
    if m:
        return f"IF {m.group(1).strip()} THEN {m.group(2).strip()}"
    return f"OBSERVE: {sentence}"


def get_context_window(sentences: List[str], idx: int, window: int = 3) -> str:
    """Return surrounding sentences for ambiguous classification."""
    start = max(0, idx - window)
    end   = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


def process_transcript(filepath: Path) -> List[Dict]:
    """
    Full Section 1 pipeline for a single transcript file.
    Returns a list of extracted concept dicts.
    """
    raw  = filepath.read_text(encoding="utf-8", errors="replace")
    text = clean_text(raw)
    sentences = split_sentences(text)

    concepts: List[Dict] = []
    source = filepath.stem  # filename without extension

    for i, sentence in enumerate(sentences):
        # Step 1: skip clear noise
        if is_noise(sentence):
            continue

        # Step 2: classify
        category = classify_sentence(sentence)
        if category == "NOISE":
            # Step 3: ambiguity check — look at context window
            ctx = get_context_window(sentences, i)
            category = classify_sentence(ctx)
            if category == "NOISE":
                continue  # confirmed noise

        # Step 4: extract concept
        rule = extract_mechanical_rule(sentence, category)

        concepts.append({
            "name": _auto_name(sentence, category),
            "category": category,
            "source_transcript": source,
            "source_quote": sentence,
            "mechanical_rule": rule,
            "timeframe": _infer_timeframe(sentence),
            "entry_condition": _infer_entry(sentence),
            "invalidation": _infer_invalidation(sentence),
            "edge_rationale": _infer_edge(sentence, category),
            "conflicts_with": [],
            "confidence": _score_confidence(sentence, category),
            "instruments": _infer_instruments(sentence),
            "occurrences": 1,
        })

    return concepts


# ── HEURISTIC HELPERS ─────────────────────────────────────────────────────────

def _auto_name(sentence: str, category: str) -> str:
    """Generate a short 3–6 word name from the sentence."""
    # Take first significant words
    words = re.findall(r"\b[A-Za-z]{4,}\b", sentence)[:5]
    base = " ".join(words).title() if words else category
    return f"{category}: {base}"[:80]


def _infer_timeframe(sentence: str) -> str:
    lower = sentence.lower()
    tfs = []
    if any(w in lower for w in ["monthly", "monthly chart"]):
        tfs.append("1MO")
    if any(w in lower for w in ["weekly", "weekly chart"]):
        tfs.append("1W")
    if any(w in lower for w in ["daily", "daily chart", "daily bias"]):
        tfs.append("1D")
    if any(w in lower for w in ["4 hour", "4h", "four hour"]):
        tfs.append("4H")
    if any(w in lower for w in ["1 hour", "1h", "one hour", "hourly"]):
        tfs.append("1H")
    if any(w in lower for w in ["15 min", "15m", "fifteen"]):
        tfs.append("15M")
    if any(w in lower for w in ["5 min", "5m", "five minute"]):
        tfs.append("5M")
    if any(w in lower for w in ["1 min", "1m", "one minute"]):
        tfs.append("1M")
    return ",".join(tfs) if tfs else "ANY"


def _infer_entry(sentence: str) -> str:
    lower = sentence.lower()
    if "when price" in lower:
        return sentence
    if "inversion" in lower:
        return "Enter when FVG inverts and price returns to it"
    if "mitigation" in lower:
        return "Enter at mitigated OB/FVG level"
    if "displacement" in lower:
        return "Enter after displacement candle confirms"
    if "sweep" in lower or "stop hunt" in lower:
        return "Enter on rejection after liquidity sweep"
    return "See source_quote for entry details"


def _infer_invalidation(sentence: str) -> str:
    lower = sentence.lower()
    if "break" in lower and ("high" in lower or "low" in lower):
        return "Setup invalidated if structure level is broken"
    if "stop loss" in lower:
        return "Stop loss placed per source_quote level"
    return "Price closes beyond the defined invalidation zone"


def _infer_edge(sentence: str, category: str) -> str:
    edges = {
        "LIQUIDITY":        "Exploits retail stop losses clustered at obvious levels",
        "FVG":              "Price is drawn to fill price inefficiencies created by institutional orders",
        "ORDER_BLOCK":      "Institutional order origin — large players re-test their own entry levels",
        "STRUCTURE":        "Defines the directional bias; trading with structure increases probability",
        "PREMIUM_DISCOUNT": "Institutions accumulate in discount, distribute in premium",
        "TIME":             "Institutional activity is highest at specific session windows",
        "ENTRY_MECHANIC":   "Confirmation-based entry reduces false positive rate",
        "RISK_MANAGEMENT":  "Consistent risk prevents account ruin regardless of win rate",
        "BIAS_FRAMEWORK":   "Top-down analysis ensures trade direction aligns with higher timeframe intent",
        "CONFLUENCE":       "Multiple confirming signals reduce noise and increase edge probability",
    }
    return edges.get(category, "See source_quote for edge rationale")


def _score_confidence(sentence: str, category: str) -> float:
    """
    Simple heuristic confidence score based on keyword density.
    Real score is updated during deduplication and simulation.
    """
    lower = sentence.lower()
    keywords = CATEGORY_KEYWORDS.get(category, [])
    hits = sum(1 for kw in keywords if kw in lower)
    base = min(0.5 + 0.1 * hits, 0.95)
    return round(base, 2)


def _infer_instruments(sentence: str) -> List[str]:
    lower = sentence.lower()
    instruments = []
    if any(w in lower for w in ["spy", "s&p", "s&p 500", "es", "e-mini"]):
        instruments.append("SPY")
    if any(w in lower for w in ["qqq", "nasdaq", "nq"]):
        instruments.append("QQQ")
    if any(w in lower for w in ["dxy", "dollar index"]):
        instruments.append("DXY")
    if any(w in lower for w in ["gold", "xau"]):
        instruments.append("GLD")
    if any(w in lower for w in ["oil", "crude"]):
        instruments.append("USO")
    if any(w in lower for w in ["bond", "treasury", "tlt"]):
        instruments.append("TLT")
    return instruments if instruments else ["ANY"]


# ── DEDUPLICATION ─────────────────────────────────────────────────────────────

def deduplicate(concepts: List[Dict]) -> List[Dict]:
    """
    Merge concepts with identical mechanical_rule across transcripts.
    Increment occurrences counter and boost confidence.
    """
    seen: Dict[str, Dict] = {}
    for concept in concepts:
        key = concept["mechanical_rule"].strip().lower()[:120]
        if key in seen:
            seen[key]["occurrences"] += 1
            seen[key]["confidence"] = min(seen[key]["confidence"] + 0.05, 0.99)
            # Append source if different
            existing_src = seen[key]["source_transcript"]
            new_src = concept["source_transcript"]
            if new_src not in existing_src:
                seen[key]["source_transcript"] += f" | {new_src}"
        else:
            seen[key] = dict(concept)
    return list(seen.values())


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def main(transcript_dir: str, output_path: str):
    transcript_dir_path = Path(transcript_dir)
    txt_files = sorted(transcript_dir_path.glob("*.txt"))

    if not txt_files:
        print(f"[ERROR] No .txt files found in {transcript_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} transcript files.")
    all_concepts: List[Dict] = []

    for fp in txt_files:
        print(f"  Parsing: {fp.name} ...", end=" ")
        concepts = process_transcript(fp)
        all_concepts.extend(concepts)
        print(f"{len(concepts)} concepts extracted")

    print(f"\nTotal raw concepts: {len(all_concepts)}")
    merged = deduplicate(all_concepts)
    print(f"After deduplication: {len(merged)} unique concepts")

    # Sort by confidence descending
    merged.sort(key=lambda c: c["confidence"], reverse=True)

    output = {
        "metadata": {
            "source_dir":    transcript_dir,
            "transcript_count": len(txt_files),
            "total_concepts":   len(merged),
        },
        "concepts": merged,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nConcept library written to: {out_path}")

    # Print summary by category
    from collections import Counter
    cats = Counter(c["category"] for c in merged)
    print("\nConcept distribution by category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} {count:>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcript concept extractor")
    parser.add_argument(
        "--dir",
        default="transcriptions/playlist_mVS4OSyj0Zg/transcripts",
        help="Directory containing .txt transcript files",
    )
    parser.add_argument(
        "--out",
        default="transcripts/processed/concept_library.json",
        help="Output path for concept_library.json",
    )
    args = parser.parse_args()
    main(args.dir, args.out)
