"""
concept_registry.py — Central registry of all approved and pending concepts.

Every concept module is registered here with:
- Its class import path
- Its default parameter set
- Its current approval status
- Its active/inactive flag (only active concepts are loaded by signal_router)

To add a new concept:
    1. Implement the module in technical/concepts/
    2. Add an entry to CONCEPT_REGISTRY below
    3. Run the simulation pipeline (stages 1–7 in Section 3.5)
    4. Update approval_status once simulation passes

Usage:
    from technical.concept_registry import load_active_concepts
    concepts = load_active_concepts()
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# ── REGISTRY ─────────────────────────────────────────────────────────────────
# Each entry: concept_id, module_path, class_name, params, approval_status, active

CONCEPT_REGISTRY: List[Dict[str, Any]] = [

    # ── GROUP A: MARKET STRUCTURE ──────────────────────────────────────────
    {
        "concept_id":      "ICT_MarketStructureShift",
        "module":          "technical.concepts.ICT_MarketStructureShift",
        "class_name":      "ICT_MarketStructureShift",
        "category":        "STRUCTURE",
        "params":          {"swing_lookback": 5, "min_displacement": 0.002},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },
    {
        "concept_id":      "ICT_BreakOfStructure",
        "module":          "technical.concepts.ICT_BreakOfStructure",
        "class_name":      "ICT_BreakOfStructure",
        "category":        "STRUCTURE",
        "params":          {"swing_lookback": 5, "min_break_pct": 0.001, "adx_filter": 20},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },
    {
        "concept_id":      "ICT_ChangeOfCharacter",
        "module":          "technical.concepts.ICT_ChangeOfCharacter",
        "class_name":      "ICT_ChangeOfCharacter",
        "category":        "STRUCTURE",
        "params":          {"swing_lookback": 5},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },

    # ── GROUP B: LIQUIDITY ─────────────────────────────────────────────────
    {
        "concept_id":      "ICT_LiquidityPool",
        "module":          "technical.concepts.ICT_LiquidityPool",
        "class_name":      "ICT_LiquidityPool",
        "category":        "LIQUIDITY",
        "params":          {"lookback_bars": 50, "equal_threshold": 0.001, "min_pool_touches": 2},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },

    # ── GROUP C: PRICE DELIVERY ────────────────────────────────────────────
    {
        "concept_id":      "ICT_FairValueGap_V2",
        "module":          "technical.concepts.ICT_FairValueGap_V2",
        "class_name":      "ICT_FairValueGap_V2",
        "category":        "FVG",
        "params":          {"min_gap_atr": 0.25, "partial_fill_pct": 0.50, "displacement_min": 1.5},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },
    {
        "concept_id":      "ICT_OrderBlock_V2",
        "module":          "technical.concepts.ICT_OrderBlock_V2",
        "class_name":      "ICT_OrderBlock_V2",
        "category":        "ORDER_BLOCK",
        "params":          {"impulse_bars": 3, "min_impulse_atr_mult": 1.5},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },

    # ── GROUP D: TIMEFRAME / TIME ──────────────────────────────────────────
    {
        "concept_id":      "ICT_PremiumDiscount",
        "module":          "technical.concepts.ICT_PremiumDiscount",
        "class_name":      "ICT_PremiumDiscount",
        "category":        "PREMIUM_DISCOUNT",
        "params":          {"swing_lookback": 10},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },
    {
        "concept_id":      "ICT_KillZone",
        "module":          "technical.concepts.ICT_KillZone",
        "class_name":      "ICT_KillZone",
        "category":        "TIME",
        "params":          {"standalone": False},
        "approval_status": "APPROVED",  # Time gate — always valid
        "active":          True,
        "wfe":             1.0,
    },
    {
        "concept_id":      "ICT_PowerOfThree",
        "module":          "technical.concepts.ICT_PowerOfThree",
        "class_name":      "ICT_PowerOfThree",
        "category":        "TIME",
        "params":          {"asian_bars_required": 6},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },

    # ── GROUP E: ENTRY MECHANICS ───────────────────────────────────────────
    {
        "concept_id":      "ICT_DailyBias",
        "module":          "technical.concepts.ICT_DailyBias",
        "class_name":      "ICT_DailyBias",
        "category":        "BIAS_FRAMEWORK",
        "params":          {"swing_lookback": 10},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
    },

    # ── GROUP F: CORRELATION ───────────────────────────────────────────────
    {
        "concept_id":      "ICT_SMTDivergence",
        "module":          "technical.concepts.ICT_SMTDivergence",
        "class_name":      "ICT_SMTDivergence",
        "category":        "CORRELATION",
        "params":          {"swing_lookback": 5, "smt_lookback_bars": 20},
        "approval_status": "PENDING",
        "active":          True,
        "wfe":             None,
        "notes":           "Requires two instrument feeds — use update(snap_a, snap_b) not detect()",
    },

    # ── GROUP G: QUANTITATIVE ──────────────────────────────────────────────
    {
        "concept_id":      "QUANT_ATR_Regime",
        "module":          "technical.concepts.QUANT_ATR_Regime",
        "class_name":      "QUANT_ATR_Regime",
        "category":        "ANOMALY",
        "params":          {"atr_lookback": 14, "percentile_window": 100},
        "approval_status": "APPROVED",  # Utility classifier — no directional signals
        "active":          True,
        "wfe":             1.0,
    },

    # ── RISK MANAGEMENT (utility, always active) ───────────────────────────
    {
        "concept_id":      "ICT_RiskManagement",
        "module":          "technical.concepts.ICT_RiskManagement",
        "class_name":      "ICT_RiskManagement",
        "category":        "RISK_MANAGEMENT",
        "params":          {"account_size": 100_000, "phase": "EVAL"},
        "approval_status": "APPROVED",
        "active":          True,
        "wfe":             1.0,
    },
]


def load_active_concepts(params_override: Optional[Dict[str, Dict]] = None) -> List[Any]:
    """
    Instantiates and returns all active concept modules.

    params_override: optional dict of {concept_id: {param_key: value}} to
                     override default params for specific concepts.
    """
    instances = []
    for entry in CONCEPT_REGISTRY:
        if not entry.get("active", True):
            continue
        try:
            module = importlib.import_module(entry["module"])
            cls: Type = getattr(module, entry["class_name"])
            params = dict(entry.get("params", {}))
            if params_override and entry["concept_id"] in params_override:
                params.update(params_override[entry["concept_id"]])
            instance = cls(params=params)
            instances.append(instance)
            logger.debug("Loaded concept: %s", entry["concept_id"])
        except Exception as e:
            logger.error("Failed to load concept %s: %s", entry["concept_id"], e)
    return instances


def get_concept_entry(concept_id: str) -> Optional[Dict[str, Any]]:
    for entry in CONCEPT_REGISTRY:
        if entry["concept_id"] == concept_id:
            return entry
    return None


def update_approval_status(concept_id: str, status: str, wfe: float, reason: str = ""):
    """Update the in-memory approval status after simulation."""
    for entry in CONCEPT_REGISTRY:
        if entry["concept_id"] == concept_id:
            entry["approval_status"] = status
            entry["wfe"] = wfe
            if reason:
                entry["rejection_reason"] = reason
            logger.info("Updated %s → %s (WFE=%.3f)", concept_id, status, wfe)
            return
    logger.warning("Concept not found in registry: %s", concept_id)


def get_approved_concepts() -> List[Dict[str, Any]]:
    return [e for e in CONCEPT_REGISTRY if e["approval_status"] == "APPROVED"]


def get_pending_concepts() -> List[Dict[str, Any]]:
    return [e for e in CONCEPT_REGISTRY if e["approval_status"] == "PENDING"]


def print_registry_summary():
    """Print a formatted table of all registered concepts."""
    from collections import Counter
    status_counts = Counter(e["approval_status"] for e in CONCEPT_REGISTRY)
    print(f"\n{'='*70}")
    print(f"  DRACULATIVE CONCEPT REGISTRY — {len(CONCEPT_REGISTRY)} total concepts")
    print(f"  APPROVED: {status_counts['APPROVED']}  |  PENDING: {status_counts['PENDING']}  |  REJECTED: {status_counts['REJECTED']}")
    print(f"{'='*70}")
    print(f"  {'ID':<35} {'CATEGORY':<20} {'STATUS':<10} {'WFE'}")
    print(f"  {'-'*65}")
    for e in CONCEPT_REGISTRY:
        wfe = f"{e['wfe']:.2f}" if e['wfe'] is not None else "N/A"
        print(f"  {e['concept_id']:<35} {e['category']:<20} {e['approval_status']:<10} {wfe}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print_registry_summary()
