#!/usr/bin/env python3
"""
core/concept_backlog_classifier.py
====================================
Classifies every PENDING concept in ``data/concept_registry.yaml``
into one of four implementation categories and produces a prioritised
backlog with concrete next-action labels.

Classification Categories
--------------------------
IMPL_GAP
    Concept is concretely described in ICT transcripts (has an entry/
    structure/liquidity rule) but no Python module exists for it yet.
    → Highest value for Sprint 3.

SIMULATION_GAP
    Concept requires execution-level data not available in daily OHLCV
    bars (tape, footprint, order flow, delta, DOM, tick data).
    → Cannot be back-tested with AlpacaDataFetcher; needs real-time feed.

WFE_FAILURE
    Concept appeared in ``data/ict2_validation_report.yaml`` with a
    REJECT status, meaning it was implemented but failed walk-forward
    evaluation.  Root cause is poor generalisation, not missing code.

ABSTRACT
    Concept is psychological, philosophical or methodological
    (mindset, patience, discipline, narrative) with no quantifiable
    signal rule.
    → Out of scope for this engine.

Priority
--------
Priority is HIGH when the concept is classified IMPL_GAP *and* the
originating cluster has ≥ 3 source episodes in the registry.

Outputs
-------
data/implementation_backlog.yaml  — full structured backlog (YAML)
Console                            — top-10 HIGH priority IMPL_GAP items

Usage
-----
    python -m core.concept_backlog_classifier
    python -m core.concept_backlog_classifier --registry data/concept_registry.yaml
    python -m core.concept_backlog_classifier --top 20
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ── Repo root on path ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_DATA_DIR = _ROOT / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Category constants ────────────────────────────────────────────────────────
IMPL_GAP       = "IMPL_GAP"
SIMULATION_GAP = "SIMULATION_GAP"
WFE_FAILURE    = "WFE_FAILURE"
ABSTRACT       = "ABSTRACT"

CATEGORIES = (IMPL_GAP, SIMULATION_GAP, WFE_FAILURE, ABSTRACT)

# ── Keyword banks for each category ──────────────────────────────────────────
_SIM_KEYWORDS: List[str] = [
    "tape", "footprint", "order flow", "delta", "dom",
    "tick", "level 2", "level ii", "depth of market", "book",
    "bid ask", "bid/ask", "volume profile", "cvd", "cumulative delta",
    "market profile", "vwap bands", "absorption",
    "time and sales", "print", "iceberg", "sweep",
]

_ABSTRACT_KEYWORDS: List[str] = [
    "mindset", "psychology", "patience", "discipline",
    "confidence", "fear", "greed", "emotion", "mental",
    "journaling", "journal", "meditation", "focus",
    "belief", "trust the process", "narrative", "storytelling",
    "philosophy", "wisdom", "consistency", "confidence",
    "routine", "lifestyle", "habits", "personal development",
    "motivation", "self", "trader psychology",
]

_IMPL_KEYWORDS: List[str] = [
    # concrete structural rules
    "entry", "exit", "setup", "model", "trigger",
    "order block", "fair value gap", "fvg", "imbalance",
    "displacement", "mitigation", "breaker", "bpr",
    "turtle soup", "silver bullet", "kill zone", "power of three",
    "new week opening gap", "nwog", "liquidity pool", "bsl", "ssl",
    "swing high", "swing low", "mss", "choch", "bos",
    "fibonacci", "ote", "retracement", "structure",
    "liquidity", "stop hunt", "inducement", "raid",
    "pdh", "pdl", "daily high", "daily low",
    "weekly bias", "daily bias", "smt", "divergence",
    "equal highs", "equal lows", "rejection", "propulsion",
    "balanced price range", "consequent encroachment",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacklogItem:
    concept:         str
    category:        str          # IMPL_GAP | SIMULATION_GAP | WFE_FAILURE | ABSTRACT
    priority:        str          # HIGH | MEDIUM | LOW
    source_episodes: int
    cluster:         Optional[str]
    reason:          str          # one-liner explaining why this category
    next_action:     str
    raw_entry:       Dict = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lower-case, strip punctuation for keyword matching."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _kw_hit(text: str, keywords: List[str]) -> Optional[str]:
    """Return the first keyword from `keywords` that appears in `text`, else None."""
    t = _norm(text)
    for kw in keywords:
        if kw in t:
            return kw
    return None


def _load_validation_rejected() -> set:
    """
    Return the set of signal names that appear as REJECT in the latest
    ict2_validation_report.yaml (if it exists).
    """
    rejected: set = set()
    report_path = _DATA_DIR / "ict2_validation_report.yaml"
    if not report_path.exists():
        return rejected
    try:
        with open(report_path) as fh:
            data = yaml.safe_load(fh) or {}
        for sig, agg in data.get("signals", {}).items():
            if agg.get("recommended_status", "") == "REJECT":
                rejected.add(sig.lower())
                rejected.add(sig.replace("_", " ").lower())
    except Exception:
        pass
    return rejected


def _load_existing_modules() -> set:
    """
    Collect the set of signal module names that already exist under
    trading_system/ict_signals/.  Used to avoid labelling something IMPL_GAP
    when a module already covers it.
    """
    modules: set = set()
    sig_dir = _ROOT / "trading_system" / "ict_signals"
    if sig_dir.exists():
        for py in sig_dir.glob("*.py"):
            stem = py.stem.replace("_detector", "").replace("_filter", "").replace("_setup", "")
            modules.add(stem.lower())
            modules.add(py.stem.lower())
    return modules


# ─────────────────────────────────────────────────────────────────────────────
# Classification logic
# ─────────────────────────────────────────────────────────────────────────────

def classify_concept(
    entry: Dict,
    rejected_signals: set,
    existing_modules: set,
) -> BacklogItem:
    """
    Classify a single registry entry into one of the 4 backlog categories.

    Decision tree (in priority order):
    1.  Text hits a SIMULATION_GAP keyword           → SIMULATION_GAP
    2.  Concept name matches a validation-rejected signal → WFE_FAILURE
    3.  Text hits an ABSTRACT keyword with no impl keyword → ABSTRACT
    4.  Otherwise (concrete rule, no module yet)     → IMPL_GAP
    """
    name:    str  = entry.get("name", "") or entry.get("concept", "") or ""
    cluster: str  = entry.get("cluster", "") or entry.get("category", "") or ""
    src_eps: int  = int(entry.get("source_episodes", entry.get("episode_count", 0)) or 0)

    full_text = f"{name} {cluster} {entry.get('description', '')}".strip()

    # 1. Simulation gap?
    sim_kw = _kw_hit(full_text, _SIM_KEYWORDS)
    if sim_kw:
        return BacklogItem(
            concept=name,
            category=SIMULATION_GAP,
            priority="LOW",
            source_episodes=src_eps,
            cluster=cluster or None,
            reason=f"Requires execution-level data ('{sim_kw}' detected)",
            next_action=(
                "Add to simulation_gap_backlog.yaml. "
                "Implement when real-time Level-2 / tick feed is available."
            ),
            raw_entry=entry,
        )

    # 2. WFE failure?
    norm_name = _norm(name)
    for rej in rejected_signals:
        if rej and rej in norm_name:
            return BacklogItem(
                concept=name,
                category=WFE_FAILURE,
                priority="MEDIUM",
                source_episodes=src_eps,
                cluster=cluster or None,
                reason=f"Signal '{rej}' rejected in walk-forward validation (WFE < 0.60)",
                next_action=(
                    "Review FAILURE_RECOVERY_MAP in validate_ict2_signals.py. "
                    "Simplify entry condition, add kill-zone filter, or switch to "
                    "ATR-relative threshold."
                ),
                raw_entry=entry,
            )

    # 3. Abstract?
    abs_kw  = _kw_hit(full_text, _ABSTRACT_KEYWORDS)
    impl_kw = _kw_hit(full_text, _IMPL_KEYWORDS)
    if abs_kw and not impl_kw:
        return BacklogItem(
            concept=name,
            category=ABSTRACT,
            priority="LOW",
            source_episodes=src_eps,
            cluster=cluster or None,
            reason=f"Psychology/mindset concept ('{abs_kw}' keyword, no signal rule)",
            next_action=(
                "Document in docs/ict2_principles.md for reference. "
                "Out of scope for quantitative engine."
            ),
            raw_entry=entry,
        )

    # 4. IMPL_GAP (default)
    # Check if an existing module already covers it
    already_covered = any(
        (_norm(mod) in norm_name) or (norm_name in _norm(mod))
        for mod in existing_modules
    )
    if already_covered:
        # Module exists — it may still be PENDING in registry until validated
        priority = "MEDIUM"
        reason   = "Concept partially covered by existing module (pending registry promotion)"
        action   = "Run validate_ict2_signals.py and promote to APPROVED if gates pass."
    elif src_eps >= 3:
        priority = "HIGH"
        reason   = (
            f"Concrete ICT rule referenced in {src_eps} episodes — "
            f"no implementation found"
        )
        action   = (
            f"Create trading_system/ict_signals/{norm_name.replace(' ', '_')}_detector.py. "
            "Use DisplacementDetector as template."
        )
    elif src_eps >= 1:
        priority = "MEDIUM"
        reason   = f"Concrete ICT rule with {src_eps} source episode(s)"
        action   = (
            "Add stub detector. Collect more transcript evidence before "
            "full implementation."
        )
    else:
        priority = "LOW"
        reason   = "Concrete concept but no episode count — unverified"
        action   = "Tag for review in concept_registry.yaml."

    return BacklogItem(
        concept=name,
        category=IMPL_GAP,
        priority=priority,
        source_episodes=src_eps,
        cluster=cluster or None,
        reason=reason,
        next_action=action,
        raw_entry=entry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pending_concepts(registry_path: Path) -> List[Dict]:
    """Load all PENDING entries from concept_registry.yaml."""
    if not registry_path.exists():
        print(f"  [WARN] Registry not found at {registry_path}. "
              "Run run_sprint2_smoke_test.py first to generate it.")
        return []

    with open(registry_path) as fh:
        data = yaml.safe_load(fh) or {}

    # Support two schemas:
    # Schema A: {"concepts": [{"name": ..., "status": "PENDING", ...}]}
    # Schema B: {"new_concepts": [...], "existing_concepts": [...]}
    # Schema C: flat list at root
    entries: List[Dict] = []

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("concepts", "new_concepts", "pending", "entries"):
            if key in data:
                entries.extend(data[key] if isinstance(data[key], list) else [])
        # Also scan existing_concepts for PENDING
        for key in ("existing_concepts",):
            if key in data:
                entries.extend(data[key] if isinstance(data[key], list) else [])

    # Filter to PENDING only (or include all if no status field)
    pending = [
        e for e in entries
        if isinstance(e, dict) and
        e.get("status", e.get("approval_status", "PENDING")).upper() == "PENDING"
    ]

    return pending


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def save_backlog(items: List[BacklogItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by category
    grouped: Dict[str, List] = {cat: [] for cat in CATEGORIES}
    for item in items:
        grouped[item.category].append(asdict(item))

    payload = {
        "generated_by": "core.concept_backlog_classifier",
        "total_pending": len(items),
        "category_counts": {cat: len(grouped[cat]) for cat in CATEGORIES},
        "backlog": grouped,
    }

    with open(out_path, "w") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)
    print(f"\n✅ Backlog saved → {out_path}")


def print_summary(items: List[BacklogItem], top_n: int = 10) -> None:
    """Print console summary with top-N HIGH priority IMPL_GAP items."""
    from collections import Counter

    cat_counts = Counter(i.category for i in items)
    pri_counts = Counter((i.category, i.priority) for i in items)

    print()
    print("=" * 68)
    print("  DRACULATIVE — CONCEPT BACKLOG CLASSIFIER")
    print("=" * 68)
    print(f"  Total PENDING concepts classified: {len(items)}")
    print()
    print(f"  {'Category':<20}  {'Count':>6}  {'High':>5}  {'Med':>5}  {'Low':>5}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for cat in CATEGORIES:
        total = cat_counts[cat]
        h = pri_counts[(cat, "HIGH")]
        m = pri_counts[(cat, "MEDIUM")]
        l = pri_counts[(cat, "LOW")]
        print(f"  {cat:<20}  {total:>6}  {h:>5}  {m:>5}  {l:>5}")
    print("=" * 68)

    # Top N HIGH priority IMPL_GAP
    high_items = sorted(
        [i for i in items if i.category == IMPL_GAP and i.priority == "HIGH"],
        key=lambda x: x.source_episodes,
        reverse=True,
    )[:top_n]

    if not high_items:
        print("\n  No HIGH priority IMPL_GAP items found.")
    else:
        print(f"\n  TOP {min(top_n, len(high_items))} HIGH PRIORITY IMPL_GAP CONCEPTS")
        print(f"  {'#':<3}  {'Concept':<40}  {'Eps':>4}  {'Cluster'}")
        print(f"  {'-'*3}  {'-'*40}  {'-'*4}  {'-'*20}")
        for rank, item in enumerate(high_items, 1):
            cluster = (item.cluster or "")[:20]
            concept = item.concept[:40]
            print(f"  {rank:<3}  {concept:<40}  {item.source_episodes:>4}  {cluster}")
            print(f"       → {item.next_action[:65]}")

    print("=" * 68)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Public API (used by run_full_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_classifier(
    registry_path: Optional[Path] = None,
    out_path:      Optional[Path] = None,
    top_n:         int = 10,
) -> List[BacklogItem]:
    """
    Main entry point.  Returns the full list of BacklogItems and writes
    ``data/implementation_backlog.yaml``.
    """
    if registry_path is None:
        registry_path = _DATA_DIR / "concept_registry.yaml"
    if out_path is None:
        out_path = _DATA_DIR / "implementation_backlog.yaml"

    print(f"  Loading pending concepts from: {registry_path}")
    pending = load_pending_concepts(registry_path)
    print(f"  Found {len(pending)} PENDING entries.")

    if not pending:
        print("  Nothing to classify.  Exiting.")
        return []

    rejected_signals = _load_validation_rejected()
    existing_modules = _load_existing_modules()

    items: List[BacklogItem] = []
    for entry in pending:
        item = classify_concept(entry, rejected_signals, existing_modules)
        items.append(item)

    save_backlog(items, out_path)
    print_summary(items, top_n=top_n)

    return items


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify PENDING concepts in concept_registry.yaml into "
                    "IMPL_GAP / SIMULATION_GAP / WFE_FAILURE / ABSTRACT"
    )
    parser.add_argument(
        "--registry",
        default=str(_DATA_DIR / "concept_registry.yaml"),
        help="Path to concept_registry.yaml (default: data/concept_registry.yaml)",
    )
    parser.add_argument(
        "--out",
        default=str(_DATA_DIR / "implementation_backlog.yaml"),
        help="Path to write implementation_backlog.yaml",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top HIGH priority IMPL_GAP items to show (default: 10)",
    )
    args = parser.parse_args()

    run_classifier(
        registry_path=Path(args.registry),
        out_path=Path(args.out),
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
