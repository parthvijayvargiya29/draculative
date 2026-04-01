"""
signal_router.py — Central Signal Dispatch Layer.

For every bar, routes the BarSnapshot to all active concepts and
collects emitted signals. Applies KillZone gate: if KillZone.detect()
returns None, non-kill-zone-aware concepts are still allowed but
concepts explicitly tagged with REQUIRES_KILL_ZONE are suppressed.

SMTDivergence special case: requires two simultaneous BarSnapshots
(e.g. ES + NQ). The router accepts an optional second symbol snapshot
and calls concept.update(snap_a, snap_b) instead of detect(snap).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from technical.bar_snapshot import BarSnapshot, Signal

logger = logging.getLogger(__name__)

# Concept classes that require two-instrument feed
DUAL_FEED_CLASSES = {"ICTSMTDivergence", "SMTDivergence"}


class SignalRouter:
    """
    Parameters
    ----------
    concepts : list
        Active, instantiated concept objects.
    kill_zone_concept : optional
        Instance of ICT_KillZone. If provided, its detect() gates certain concepts.
    require_kill_zone : bool
        If True, ALL signals are suppressed outside kill zones.
    """

    def __init__(
        self,
        concepts: list,
        kill_zone_concept=None,
        require_kill_zone: bool = False,
    ):
        self.concepts         = concepts
        self.kill_zone        = kill_zone_concept
        self.require_kill_zone = require_kill_zone

        # Separate dual-feed concepts from single-feed
        self._single: list = []
        self._dual:   list = []
        for c in concepts:
            cname = type(c).__name__
            if cname in DUAL_FEED_CLASSES or getattr(c, "REQUIRES_DUAL_FEED", False):
                self._dual.append(c)
            else:
                self._single.append(c)

    def route(
        self,
        snapshot: BarSnapshot,
        snapshot_b: Optional[BarSnapshot] = None,
    ) -> List[Signal]:
        """
        Run all concepts against the snapshot(s) and return collected signals.

        Parameters
        ----------
        snapshot   : Primary symbol BarSnapshot (e.g. NQ, ES, SPY)
        snapshot_b : Optional secondary BarSnapshot for SMT divergence (e.g. ES when primary is NQ)
        """
        signals: List[Signal] = []

        # ── Kill-zone check ──────────────────────────────────────────────
        in_kill_zone = False
        if self.kill_zone is not None:
            kz_signal = self._safe_detect(self.kill_zone, snapshot)
            in_kill_zone = kz_signal is not None

        if self.require_kill_zone and not in_kill_zone:
            return []  # Hard gate: outside kill zones, silence everything

        # ── Single-feed concepts ─────────────────────────────────────────
        for concept in self._single:
            sig = self._safe_detect(concept, snapshot)
            if sig is not None:
                signals.append(sig)

        # ── Dual-feed concepts (SMT Divergence, etc.) ────────────────────
        if snapshot_b is not None:
            for concept in self._dual:
                try:
                    sig = concept.update(snapshot, snapshot_b)
                except Exception as exc:
                    logger.debug("Dual-feed concept %s raised: %s", type(concept).__name__, exc)
                    sig = None
                if sig is not None:
                    signals.append(sig)

        if signals:
            logger.debug(
                "SignalRouter @ %s: %d concept(s) fired %d signal(s)",
                snapshot.timestamp, len(self.concepts), len(signals),
            )

        return signals

    def route_multi(
        self,
        snapshots: Dict[str, BarSnapshot],
    ) -> Dict[str, List[Signal]]:
        """
        Route multiple symbols at once.
        Pairs the first two symbols for SMT divergence if dual-feed concepts exist.

        Parameters
        ----------
        snapshots : dict {symbol: BarSnapshot}

        Returns
        -------
        dict {symbol: [Signal, ...]}
        """
        result: Dict[str, List[Signal]] = {}
        symbols = list(snapshots.keys())

        # Detect SMT pair (first two symbols if dual-feed concepts exist)
        snap_b = snapshots.get(symbols[1]) if len(symbols) >= 2 and self._dual else None

        for sym, snap in snapshots.items():
            result[sym] = self.route(snap, snap_b if sym == symbols[0] else None)

        return result

    @staticmethod
    def _safe_detect(concept, snapshot: BarSnapshot) -> Optional[Signal]:
        try:
            return concept.detect(snapshot)
        except Exception as exc:
            logger.debug("Concept %s.detect() raised: %s", type(concept).__name__, exc)
            return None
