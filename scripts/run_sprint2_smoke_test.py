#!/usr/bin/env python3
"""
scripts/run_sprint2_smoke_test.py
==================================
Sprint 2 pre-flight smoke test. Verifies:

  1A  41 ICT2 transcript files are readable
  1B  ICT2ConceptExtractor runs end-to-end
  1C  All 8 ICT2 signal modules import and run on synthetic data
  1D  All 3 config YAML files load

Exit code 0 = all clear. Exit code 1 = any failure.

Run: python scripts/run_sprint2_smoke_test.py
"""

from __future__ import annotations

import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

# ── Repo root ─────────────────────────────────────────────────────────────────
_REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

_FAIL = False


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    global _FAIL
    _FAIL = True
    print(f"  ✗ {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1A — transcript file check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("STEP 1A — ICT2 Transcript Files")
print("=" * 68)

transcripts_dir = _REPO / "transcriptions" / "ICT2" / "transcripts"
txt_files = sorted(transcripts_dir.glob("*.txt"))

try:
    assert len(txt_files) == 41, f"Expected 41 transcripts, found {len(txt_files)}"
    total_bytes = sum(f.stat().st_size for f in txt_files)
    total_mb = total_bytes / (1024 * 1024)
    _ok(f"41 ICT2 transcripts found. Total size: {total_mb:.1f} MB")
except AssertionError as e:
    _fail(str(e))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1B — Concept extractor
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("STEP 1B — ICT2 Concept Extractor")
print("=" * 68)

try:
    # Import the module's standalone run() function
    from core.ict2_concept_extractor import (
        load_transcripts,
        load_existing_registry,
        extract_all_concepts,
        deduplicate_raw,
        classify_against_registry,
        cluster_concepts,
        assign_clusters,
        write_registry,
        write_delta,
        write_report,
        _split_paragraphs,
        TRANSCRIPTS_DIR,
        REGISTRY_PATH,
        DELTA_PATH,
        REPORT_PATH,
        DATA_DIR,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading transcripts …")
    transcripts = load_transcripts(TRANSCRIPTS_DIR)
    print("  Loading existing registry …")
    existing_entries = load_existing_registry(REGISTRY_PATH)

    print("  Extracting concepts …")
    all_raw = extract_all_concepts(transcripts)
    total_paragraphs = sum(len(_split_paragraphs(t)) for t in transcripts.values())

    print("  Deduplicating …")
    deduped_raw = deduplicate_raw(all_raw)

    print("  Classifying against ICT1 registry …")
    new_entries, status_map = classify_against_registry(deduped_raw, existing_entries)

    new_count     = sum(1 for s in status_map.values() if s == "NEW")
    variant_count = sum(1 for s in status_map.values() if s == "VARIANT")
    dup_count     = sum(1 for s in status_map.values() if s == "DUPLICATE")

    print("  Clustering …")
    new_raw = [r for r in deduped_raw if status_map.get(r.concept_name) == "NEW"]
    cluster_map = cluster_concepts(new_raw, existing_entries)
    assign_clusters(new_entries, deduped_raw, cluster_map)

    print("  Writing outputs …")
    write_registry(REGISTRY_PATH, existing_entries, new_entries)
    write_delta(DELTA_PATH, new_entries)
    write_report(REPORT_PATH, all_raw, new_entries, len(existing_entries), total_paragraphs)

    unique_count = len(deduped_raw)
    # Count clusters
    cluster_ids = {e.cluster_id for e in new_entries if e.status == "NEW"}
    cluster_count = len(cluster_ids)

    _ok("Concept extraction complete")
    print(f"    Paragraphs scanned:   {total_paragraphs:,}")
    print(f"    Raw concepts found:   {len(all_raw):,}")
    print(f"    After dedup:          {unique_count:,}")
    print(f"    New (not in ICT1):    {new_count:,}")
    print(f"    Variants of ICT1:     {variant_count:,}")
    print(f"    Duplicates (skipped): {dup_count:,}")
    print(f"    Clusters identified:  {cluster_count:,}")

    # Top 15 new concepts by cluster frequency
    from collections import Counter
    new_only = [e for e in new_entries if e.status == "NEW"]
    # Source episode count proxy: how many entries share the same canonical_name
    canonical_ep_count: Counter = Counter()
    for e in new_only:
        canonical_ep_count[e.canonical_name] += len(e.source_episodes)

    # Build a quick lookup: canonical_name → entry
    canonical_map = {}
    for e in new_only:
        if e.canonical_name not in canonical_map:
            canonical_map[e.canonical_name] = e

    print(f"\n    Top 15 new ICT2 concepts:")
    for name, ep_count in canonical_ep_count.most_common(15):
        e = canonical_map.get(name)
        if e:
            print(
                f"      [{e.cluster_id}] {e.canonical_name:<50} "
                f"({e.ict_category}) — {ep_count} ep-mentions"
            )

except Exception:
    _fail("Concept extractor raised an exception (see traceback below)")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1C — Module smoke test on synthetic data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("STEP 1C — ICT2 Signal Module Smoke Tests (synthetic data)")
print("=" * 68)


def _synthetic_bars(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    close  = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high   = close + rng.uniform(0.1, 1.0, n)
    low    = close - rng.uniform(0.1, 1.0, n)
    open_  = close + rng.normal(0, 0.2, n)
    vol    = rng.integers(500_000, 2_000_000, n).astype(float)
    dates  = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates
    )


_df = _synthetic_bars(n=120)  # 120 bars for better detection coverage

modules = [
    ("KillZoneFilter",       "trading_system.ict_signals.killzone_filter",    "KillZoneDetector"),
    ("DisplacementDetector", "trading_system.ict_signals.displacement_detector", "DisplacementDetector"),
    ("NWOGDetector",         "trading_system.ict_signals.nwog_detector",       "NWOGDetector"),
    ("PropulsionBlockDetector", "trading_system.ict_signals.propulsion_block_detector", "PropulsionBlockDetector"),
    ("BPRDetector",          "trading_system.ict_signals.balanced_price_range", "BPRDetector"),
    ("TurtleSoupDetector",   "trading_system.ict_signals.turtle_soup_detector", "TurtleSoupDetector"),
    ("PowerOfThreeDetector", "trading_system.ict_signals.power_of_three",       "PowerOfThreeDetector"),
    ("SilverBulletDetector", "trading_system.ict_signals.silver_bullet_setup",  "SilverBulletDetector"),
]

for display_name, module_path, class_name in modules:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        if class_name == "KillZoneDetector":
            detector = cls(htf_bias="bullish")
            # daily bar call
            result = detector.process(None, float(_df["close"].iloc[-1]))
        elif class_name == "PowerOfThreeDetector":
            detector = cls(expected_direction="bullish")
            result = detector.update(_df)
        else:
            detector = cls()
            result = detector.update(_df)

        result_type = type(result).__name__
        # Show key fields
        if hasattr(result, "detected"):
            detail = f"detected={result.detected}"
        elif hasattr(result, "active_zone"):
            detail = f"active_zone={result.active_zone!r}"
        elif hasattr(result, "bias_from_gaps"):
            detail = f"bias_from_gaps={result.bias_from_gaps!r}"
        elif hasattr(result, "setup_valid"):
            detail = f"setup_valid={result.setup_valid}"
        elif hasattr(result, "phase"):
            detail = f"phase={result.phase!r}"
        else:
            detail = repr(result)[:80]

        _ok(f"{display_name:<26} → {result_type}({detail})")

    except Exception:
        _fail(f"{display_name} raised an exception:")
        traceback.print_exc()
        sys.exit(1)   # per spec: exit on any module failure


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1D — Config file verification
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("STEP 1D — Config File Verification")
print("=" * 68)

cfg_files = [
    "configs/ict_signals.yaml",
    "configs/convergence_weights.yaml",
    "configs/scheduler.yaml",
]

for cfg_rel in cfg_files:
    cfg_path = _REPO / cfg_rel
    try:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(cfg_path)
            n_keys = len(cfg)
        except ImportError:
            import yaml
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh) or {}
            n_keys = len(cfg)
        assert cfg is not None, "Config is None"
        _ok(f"{cfg_rel} loads OK ({n_keys} top-level keys)")
    except FileNotFoundError:
        _fail(f"{cfg_rel} NOT FOUND")
    except Exception:
        _fail(f"{cfg_rel} failed to load")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
if _FAIL:
    print("❌ SMOKE TEST FAILED — fix issues above before proceeding to Task 2")
    sys.exit(1)
else:
    print("✅ ALL SMOKE TESTS PASSED — proceed to Task 2")
    print("=" * 68)
