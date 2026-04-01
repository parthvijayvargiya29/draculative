#!/usr/bin/env python3
"""
ICT2 Concept Extractor & Deduplication Pipeline
=================================================
Loads all 41 ICT2 transcript files from transcriptions/ICT2/transcripts/,
extracts named ICT/SMC concepts using a keyword-seed + regex NLP pipeline,
deduplicates against the existing ICT1 concept registry using TF-IDF cosine
similarity (sklearn) with Jaccard fallback, clusters NEW concepts via KMeans
(elbow-method k selection), and writes:

  data/concept_registry.yaml        — full merged ICT1 + ICT2 registry
  data/ict2_delta_concepts.yaml     — only NEW ICT2 concepts
  data/ict2_extraction_report.txt   — human-readable summary

Usage:
    python core/ict2_concept_extractor.py
"""

from __future__ import annotations

import re
import sys
import math
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT        = Path(__file__).parent.parent
TRANSCRIPTS_DIR  = REPO_ROOT / "transcriptions" / "ICT2" / "transcripts"
DATA_DIR         = REPO_ROOT / "data"
REGISTRY_PATH    = DATA_DIR / "concept_registry.yaml"
DELTA_PATH       = DATA_DIR / "ict2_delta_concepts.yaml"
REPORT_PATH      = DATA_DIR / "ict2_extraction_report.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Concept seeds ─────────────────────────────────────────────────────────────
CONCEPT_SEEDS: Dict[str, List[str]] = {
    "structure": [
        "market structure shift", "mss", "displacement", "impulse move",
        "mitigation", "fair value gap", "fvg", "imbalance", "inefficiency",
        "order block", "ob", "breaker block", "breaker", "propulsion block",
        "rejection block", "balanced price range", "bpr",
        "consequent encroachment", "ce", "return to breaker",
        "reclaimed order block", "inversion fair value gap", "ifvg",
        "volume imbalance", "gap and go", "single print", "range contraction",
        "change of character", "choch", "break of structure", "bos",
        "swing high", "swing low", "higher high", "higher low",
        "lower high", "lower low", "internal structure", "external structure",
    ],
    "liquidity": [
        "buy side liquidity", "sell side liquidity", "bsl", "ssl",
        "equal highs", "equal lows", "stops", "liquidity pool", "inducement",
        "turtle soup", "stop hunt", "run on liquidity", "old high", "old low",
        "swing failure pattern", "sfp", "raid", "liquidity void",
        "engineering liquidity", "sponsored move", "no dealing desk",
        "hunt for stops", "relative equal highs", "relative equal lows",
        "previous high", "previous low", "weekly high", "weekly low",
        "daily high", "daily low", "yearly high", "yearly low",
    ],
    "entry": [
        "optimal trade entry", "ote", "fibonacci", "62%", "79%", "70.5%",
        "entry model", "sniper entry", "confirmation", "candle close",
        "15 minute", "5 minute", "1 minute", "entry timeframe",
        "higher timeframe", "htf", "ltf", "model", "setup", "trigger",
        "silver bullet", "london open kill zone", "new york open kill zone",
        "asia kill zone", "power of three", "accumulation distribution",
        "manipulation", "institutional reference point", "daily bias",
        "weekly bias", "monthly bias", "smt divergence", "smt",
        "intraday bias", "dealing range", "pdh", "pdl",
    ],
    "time": [
        "kill zone", "london killzone", "new york killzone", "asia killzone",
        "time of day", "session", "new week opening gap", "nwog",
        "new month opening gap", "nmog", "opening range", "midnight open",
        "cme open", "true open", "open of day", "quarterly shift",
        "ict quarterly theory", "turtle dove", "daily candle",
        "8:30 macro", "9:30 macro", "10 am macro", "2pm macro",
        "judas swing", "expansion", "retracement", "consolidation",
        "london close", "new york close", "asian range",
        "first hour", "last hour", "power hour",
    ],
    "risk": [
        "stop loss", "above high", "below low", "2.5r", "3r",
        "partial", "scale out", "position sizing", "risk management",
        "kill shot", "entry to exit", "drawdown", "risk per trade",
        "max loss", "reward to risk", "risk reward", "trail stop",
        "breakeven", "partial profit",
    ],
    "macro": [
        "intermarket", "dollar", "dxy", "bonds", "treasuries",
        "equity correlation", "risk on", "risk off", "seasonal",
        "quarterly theory", "co", "delivery", "smart money",
        "institutional order flow", "central bank", "fed", "fomc",
        "nfp", "cpi", "news event", "economic calendar",
    ],
}

# ── Regex patterns for concept explanation sentences ─────────────────────────
EXPLANATION_PATTERNS = [
    r"(?:this is called|that's called|we call this|known as)\s+(?:the\s+)?(.{5,60})",
    r"(?:the key thing about|the important thing about)\s+(.{5,60})\s+is",
    r"(?:what you're looking for is|you're looking for)\s+(.{5,60})",
    r"(?:when price does|when price)\s+(.{5,80}),?\s+(?:that means|that's)",
    r"(?:that is|this is)\s+(?:a|an|the)\s+(.{5,60})\s+(?:setup|concept|level|zone|area|model)",
    r"(?:i call this|i refer to this as)\s+(?:the\s+)?(.{5,60})",
]
_EXPL_RE = [re.compile(p, re.IGNORECASE) for p in EXPLANATION_PATTERNS]

# ── Normalisation ─────────────────────────────────────────────────────────────
def _snake(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:80]

def _normalise_name(raw: str) -> str:
    raw = raw.lower().strip()
    raw = re.sub(r"[^a-z0-9%.: /_-]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw

# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class RawConcept:
    concept_name: str
    raw_definition: str
    source_episode: int
    source_playlist: str = "ICT2_2022_Mentorship"
    ict_category: str = "structure"
    related_concepts: List[str] = field(default_factory=list)
    implementation_status: str = "PENDING"
    blocker_type: str = ""

@dataclass
class ConceptEntry:
    concept_id: str
    canonical_name: str
    status: str                    # NEW | VARIANT | DUPLICATE | IMPLEMENTED | PENDING
    blocker_type: str              # "" | SIMULATION_GAP | WFE_FAILURE | IMPL_GAP
    ict_category: str
    source_episodes: List[int]
    source_playlists: List[str]
    cluster_id: int
    similarity_score: float
    raw_definition: str
    related_concepts: List[str]
    implementation_notes: str


# ── Paragraph splitter ────────────────────────────────────────────────────────
def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; fall back to sentence chunks of ~3 sentences."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) < 5:
        # transcript has no blank lines — split on sentence groups
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunk, result = [], []
        for s in sentences:
            chunk.append(s)
            if len(chunk) >= 4:
                result.append(" ".join(chunk))
                chunk = []
        if chunk:
            result.append(" ".join(chunk))
        paras = result
    return paras


# ── Concept detector ──────────────────────────────────────────────────────────
def _detect_category(para_lower: str) -> str:
    best, best_score = "structure", 0
    for cat, seeds in CONCEPT_SEEDS.items():
        score = sum(1 for s in seeds if s in para_lower)
        if score > best_score:
            best_score, best = score, cat
    return best


def _detect_concepts_in_paragraph(
    para: str,
    episode_num: int,
    all_seeds_flat: List[str],
) -> List[RawConcept]:
    """Extract concepts mentioned in a single paragraph."""
    lower = para.lower()
    found_seeds = [s for s in all_seeds_flat if s in lower]
    if not found_seeds:
        return []

    category = _detect_category(lower)
    definition = para[:500]
    related = [_snake(_normalise_name(s)) for s in found_seeds]

    concepts: List[RawConcept] = []

    # 1. Seed-based: for each seed found, create a concept entry
    for seed in found_seeds:
        cname = _snake(_normalise_name(seed))
        concepts.append(RawConcept(
            concept_name=cname,
            raw_definition=definition,
            source_episode=episode_num,
            ict_category=category,
            related_concepts=[r for r in related if r != cname],
        ))

    # 2. Explanation-pattern bonus: extract explicitly named concepts
    for pattern in _EXPL_RE:
        for m in pattern.finditer(para):
            name_raw = m.group(1).strip()
            cname = _snake(_normalise_name(name_raw))
            if 3 <= len(cname) <= 80:
                concepts.append(RawConcept(
                    concept_name=cname,
                    raw_definition=definition,
                    source_episode=episode_num,
                    ict_category=category,
                    related_concepts=related[:5],
                ))

    return concepts


# ── Load transcripts ──────────────────────────────────────────────────────────
def load_transcripts(transcripts_dir: Path) -> Dict[int, str]:
    """Returns {episode_number: text}"""
    result = {}
    txt_files = sorted(transcripts_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", transcripts_dir)
        return result

    for path in txt_files:
        # Extract episode number from filename prefix "NN - ..."
        m = re.match(r"^(\d+)", path.stem)
        ep_num = int(m.group(1)) if m else len(result) + 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            continue
        result[ep_num] = text
        logger.info("  Loaded episode %02d  (%d chars)  %s", ep_num, len(text), path.name)

    return result


# ── Load existing registry ────────────────────────────────────────────────────
def load_existing_registry(path: Path) -> List[ConceptEntry]:
    if not path.exists():
        return []
    with open(path) as fh:
        raw = yaml.safe_load(fh) or []
    entries = []
    for r in raw:
        try:
            entries.append(ConceptEntry(**{
                k: v for k, v in r.items()
                if k in ConceptEntry.__dataclass_fields__
            }))
        except Exception:
            pass
    return entries


# ── TF-IDF similarity (with Jaccard fallback) ─────────────────────────────────
def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_similarity_fn(existing_defs: List[str]):
    """Returns a function (new_def: str) -> List[float] (scores vs each existing)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not existing_defs:
            return lambda _: []

        vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            stop_words="english", sublinear_tf=True
        )
        matrix = vec.fit_transform(existing_defs)

        def _score(new_def: str) -> List[float]:
            v = vec.transform([new_def])
            return cosine_similarity(v, matrix)[0].tolist()

        logger.info("  Using TF-IDF cosine similarity (sklearn)")
        return _score

    except ImportError:
        logger.warning("  sklearn not available — using Jaccard similarity fallback")

        def _score_jaccard(new_def: str) -> List[float]:
            return [_jaccard(new_def, ex) for ex in existing_defs]

        return _score_jaccard


# ── KMeans clustering (with manual fallback) ──────────────────────────────────
def cluster_concepts(
    concepts: List[RawConcept],
    existing_entries: List[ConceptEntry],
) -> Dict[str, int]:
    """Returns {concept_name: cluster_id}. Uses elbow method to pick k."""
    texts = [c.raw_definition for c in concepts]
    if not texts:
        return {}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import numpy as np

        vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            stop_words="english", max_features=5000
        )
        X = vec.fit_transform(texts)

        # Elbow method: try k = 5..min(60, n//2)
        n = X.shape[0]
        k_max = min(60, max(5, n // 2))
        inertias = []
        k_range = range(5, k_max + 1, max(1, (k_max - 5) // 20))
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            km.fit(X)
            inertias.append((k, km.inertia_))

        # Pick elbow: largest second-derivative of inertia
        if len(inertias) >= 3:
            best_k = 20
            max_delta2 = -1e9
            for i in range(1, len(inertias) - 1):
                k1, i1 = inertias[i - 1]
                k2, i2 = inertias[i]
                k3, i3 = inertias[i + 1]
                delta2 = (i1 - i2) - (i2 - i3)
                if delta2 > max_delta2:
                    max_delta2 = delta2
                    best_k = k2
        else:
            best_k = 20

        km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = km_final.fit_predict(X)
        logger.info("  KMeans: k=%d  (n_concepts=%d)", best_k, n)
        return {c.concept_name: int(labels[i]) for i, c in enumerate(concepts)}

    except ImportError:
        logger.warning("  sklearn not available — using category-based clusters")
        cat_ids = {"structure": 0, "liquidity": 1, "entry": 2,
                   "time": 3, "risk": 4, "macro": 5}
        return {c.concept_name: cat_ids.get(c.ict_category, 0) for c in concepts}


# ── Main extraction pipeline ──────────────────────────────────────────────────
def extract_all_concepts(transcripts: Dict[int, str]) -> List[RawConcept]:
    """Run the full extraction over all episodes."""
    all_seeds_flat = list({s for seeds in CONCEPT_SEEDS.values() for s in seeds})
    all_concepts: List[RawConcept] = []
    total_paragraphs = 0

    for ep_num in sorted(transcripts.keys()):
        text = transcripts[ep_num]
        paras = _split_paragraphs(text)
        total_paragraphs += len(paras)
        ep_concepts = []

        for para in paras:
            ep_concepts.extend(
                _detect_concepts_in_paragraph(para, ep_num, all_seeds_flat)
            )

        all_concepts.extend(ep_concepts)
        logger.info(
            "  Episode %02d: %d paragraphs → %d concept mentions",
            ep_num, len(paras), len(ep_concepts)
        )

    logger.info(
        "Extraction complete: %d total paragraphs, %d concept mentions",
        total_paragraphs, len(all_concepts)
    )
    return all_concepts


def deduplicate_raw(concepts: List[RawConcept]) -> List[RawConcept]:
    """
    Collapse exact concept_name duplicates within the raw ICT2 extraction,
    merging their episode lists and choosing the richest definition.
    """
    merged: Dict[str, RawConcept] = {}
    for c in concepts:
        key = c.concept_name
        if key not in merged:
            merged[key] = c
        else:
            # Prefer longer definition, merge related
            existing = merged[key]
            if len(c.raw_definition) > len(existing.raw_definition):
                existing.raw_definition = c.raw_definition
            existing.related_concepts = list(
                set(existing.related_concepts + c.related_concepts)
            )[:10]
    return list(merged.values())


def classify_against_registry(
    new_concepts: List[RawConcept],
    existing_entries: List[ConceptEntry],
    sim_threshold_dup: float = 0.72,
    sim_threshold_variant: float = 0.50,
) -> Tuple[List[ConceptEntry], Dict[str, str]]:
    """
    Returns (all_entries_merged, {concept_name: status}).
    Status: NEW | VARIANT | DUPLICATE
    """
    existing_defs = [e.raw_definition for e in existing_entries]
    sim_fn = build_similarity_fn(existing_defs)

    new_entries: List[ConceptEntry] = []
    status_map: Dict[str, str] = {}

    for i, rc in enumerate(new_concepts):
        scores = sim_fn(rc.raw_definition)

        if scores:
            max_score = max(scores)
            best_idx  = scores.index(max_score)
        else:
            max_score = 0.0
            best_idx  = -1

        if max_score >= sim_threshold_dup:
            status = "DUPLICATE"
            notes = f"Duplicate of {existing_entries[best_idx].canonical_name} (sim={max_score:.3f})"
        elif max_score >= sim_threshold_variant:
            status = "VARIANT"
            notes = f"Extends {existing_entries[best_idx].canonical_name} (sim={max_score:.3f})"
        else:
            status = "NEW"
            notes = "New concept from ICT2 not found in ICT1 registry"

        status_map[rc.concept_name] = status
        concept_id = f"ICT2_{i:04d}_{rc.concept_name[:30]}"

        new_entries.append(ConceptEntry(
            concept_id=concept_id,
            canonical_name=rc.concept_name,
            status=status,
            blocker_type="",
            ict_category=rc.ict_category,
            source_episodes=[rc.source_episode],
            source_playlists=[rc.source_playlist],
            cluster_id=-1,    # filled by cluster step
            similarity_score=round(max_score, 4),
            raw_definition=rc.raw_definition,
            related_concepts=rc.related_concepts,
            implementation_notes=notes,
        ))

    return new_entries, status_map


def assign_clusters(
    entries: List[ConceptEntry],
    raw_concepts: List[RawConcept],
    cluster_map: Dict[str, int],
) -> None:
    """Mutate entries in-place: assign cluster_id and canonical_name per cluster."""
    for entry in entries:
        entry.cluster_id = cluster_map.get(entry.canonical_name, 0)

    # For each cluster, find the most frequent concept name → becomes canonical
    cluster_freq: Dict[int, Counter] = defaultdict(Counter)
    for entry in entries:
        cluster_freq[entry.cluster_id][entry.canonical_name] += 1

    cluster_canonical: Dict[int, str] = {
        cid: counter.most_common(1)[0][0]
        for cid, counter in cluster_freq.items()
    }

    for entry in entries:
        entry.canonical_name = cluster_canonical.get(entry.cluster_id, entry.canonical_name)


# ── YAML serialisation ────────────────────────────────────────────────────────
def _entry_to_dict(e: ConceptEntry) -> dict:
    return {
        "concept_id":           e.concept_id,
        "canonical_name":       e.canonical_name,
        "status":               e.status,
        "blocker_type":         e.blocker_type,
        "ict_category":         e.ict_category,
        "source_episodes":      e.source_episodes,
        "source_playlists":     e.source_playlists,
        "cluster_id":           e.cluster_id,
        "similarity_score":     e.similarity_score,
        "raw_definition":       e.raw_definition[:500],
        "related_concepts":     e.related_concepts[:10],
        "implementation_notes": e.implementation_notes,
    }


def write_registry(path: Path, existing: List[ConceptEntry], new: List[ConceptEntry]) -> None:
    all_entries = existing + new
    data = [_entry_to_dict(e) for e in all_entries]
    with open(path, "w") as fh:
        yaml.dump(data, fh, allow_unicode=True, sort_keys=False, width=120)
    logger.info("Registry written: %d entries → %s", len(data), path)


def write_delta(path: Path, new_entries: List[ConceptEntry]) -> None:
    delta_only = [e for e in new_entries if e.status == "NEW"]
    # Sort by category then cluster_id
    delta_only.sort(key=lambda e: (e.ict_category, e.cluster_id))
    data = [_entry_to_dict(e) for e in delta_only]
    with open(path, "w") as fh:
        yaml.dump(data, fh, allow_unicode=True, sort_keys=False, width=120)
    logger.info("Delta written: %d new concepts → %s", len(data), path)


def write_report(
    path: Path,
    all_raw: List[RawConcept],
    new_entries: List[ConceptEntry],
    existing_count: int,
    total_paragraphs: int,
) -> None:
    counts = Counter(e.status for e in new_entries)
    new_only = [e for e in new_entries if e.status == "NEW"]

    # Top 20 by concept_name frequency across raw mentions
    name_freq = Counter(r.concept_name for r in all_raw)
    top20 = name_freq.most_common(20)

    # Cluster summary
    cluster_counts: Dict[int, Dict[str, int]] = defaultdict(Counter)
    for e in new_entries:
        cluster_counts[e.cluster_id][e.canonical_name] += 1

    # Immediate implementation candidates: NEW, no blocker, entry/structure/liquidity
    impl_candidates = [
        e for e in new_only
        if not e.blocker_type and e.ict_category in ("entry", "structure", "liquidity")
    ]

    lines = [
        "=" * 80,
        "  ICT2 CONCEPT EXTRACTION REPORT — 2022 ICT Mentorship (41 Episodes)",
        "=" * 80,
        "",
        f"  Total paragraphs scanned  : {total_paragraphs:,}",
        f"  Total concept mentions    : {len(all_raw):,}",
        f"  Unique concepts extracted : {len(new_entries):,}",
        f"  Existing ICT1 registry    : {existing_count} entries",
        "",
        "  DEDUPLICATION BREAKDOWN",
        "  ─────────────────────────────────────────",
        f"  NEW         : {counts.get('NEW', 0):>5}  (not in ICT1, unique to ICT2)",
        f"  VARIANT     : {counts.get('VARIANT', 0):>5}  (extends an ICT1 concept)",
        f"  DUPLICATE   : {counts.get('DUPLICATE', 0):>5}  (same as existing ICT1 concept)",
        "",
        "  TOP 20 NEW CONCEPTS BY FREQUENCY",
        "  ─────────────────────────────────────────",
    ]
    for rank, (name, freq) in enumerate(top20, 1):
        status = "NEW" if name in {e.canonical_name for e in new_only} else "dup/var"
        lines.append(f"  {rank:>2}. {name:<55} freq={freq:>4}  [{status}]")

    lines += [
        "",
        "  CLUSTER SUMMARY (KMeans)",
        "  ─────────────────────────────────────────",
    ]
    for cid in sorted(cluster_counts.keys()):
        canonical = max(cluster_counts[cid], key=cluster_counts[cid].get)
        count = sum(cluster_counts[cid].values())
        lines.append(f"  Cluster {cid:>3}  canonical={canonical:<50} n={count}")

    lines += [
        "",
        "  RECOMMENDED FOR IMMEDIATE IMPLEMENTATION",
        "  ─────────────────────────────────────────",
        f"  (NEW concepts, no blocker, category=entry/structure/liquidity, top {min(30, len(impl_candidates))})",
    ]
    for e in sorted(impl_candidates, key=lambda x: x.cluster_id)[:30]:
        lines.append(
            f"  • [{e.ict_category:<10}] {e.canonical_name:<55} cluster={e.cluster_id}"
        )

    lines += ["", "=" * 80]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written → %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────
def run() -> None:
    logger.info("=" * 60)
    logger.info("ICT2 Concept Extractor")
    logger.info("=" * 60)

    # 1. Load transcripts
    logger.info("Loading ICT2 transcripts from %s …", TRANSCRIPTS_DIR)
    transcripts = load_transcripts(TRANSCRIPTS_DIR)
    if not transcripts:
        logger.error("No transcripts found. Run the transcription pipeline first.")
        sys.exit(1)

    # 2. Load existing registry
    logger.info("Loading existing concept registry from %s …", REGISTRY_PATH)
    existing_entries = load_existing_registry(REGISTRY_PATH)
    logger.info("  Existing entries: %d", len(existing_entries))

    # 3. Extract raw concepts from ICT2
    logger.info("Extracting concepts …")
    all_raw = extract_all_concepts(transcripts)
    total_paragraphs = sum(
        len(_split_paragraphs(t)) for t in transcripts.values()
    )

    # 4. Deduplicate within ICT2 raw mentions
    logger.info("Deduplicating within ICT2 raw mentions …")
    deduped_raw = deduplicate_raw(all_raw)
    logger.info("  Unique ICT2 concepts: %d", len(deduped_raw))

    # 5. Classify against existing registry
    logger.info("Classifying against ICT1 registry …")
    new_entries, status_map = classify_against_registry(deduped_raw, existing_entries)

    new_count      = sum(1 for s in status_map.values() if s == "NEW")
    variant_count  = sum(1 for s in status_map.values() if s == "VARIANT")
    dup_count      = sum(1 for s in status_map.values() if s == "DUPLICATE")
    logger.info("  NEW=%d  VARIANT=%d  DUPLICATE=%d", new_count, variant_count, dup_count)

    # 6. Cluster NEW concepts
    logger.info("Clustering NEW concepts …")
    new_raw = [r for r in deduped_raw if status_map.get(r.concept_name) == "NEW"]
    cluster_map = cluster_concepts(new_raw, existing_entries)
    assign_clusters(new_entries, deduped_raw, cluster_map)

    # 7. Write outputs
    logger.info("Writing outputs …")
    write_registry(REGISTRY_PATH, existing_entries, new_entries)
    write_delta(DELTA_PATH, new_entries)
    write_report(REPORT_PATH, all_raw, new_entries, len(existing_entries), total_paragraphs)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Registry : %s", REGISTRY_PATH)
    logger.info("  Delta    : %s", DELTA_PATH)
    logger.info("  Report   : %s", REPORT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
