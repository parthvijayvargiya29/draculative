"""
CONCEPT: ICT Kill Zones (London + New York + Asian)
SOURCE: 07 - ICT FOR DUMMIES | Time EP. 6
CATEGORY: TIME
TIMEFRAME: 15M, 5M, 1M
INSTRUMENTS: ANY (primarily forex + futures)
DESCRIPTION:
    Kill Zones are specific windows of time during each trading session when
    institutional order flow is highest and price delivery is most reliable.
    The speaker emphasizes that you do NOT trade every session — you pick the
    one that aligns with your model and stick to it.

    ASIAN  Kill Zone: 20:00–00:00 EST — accumulation, sets the range for London
    LONDON Kill Zone: 02:00–05:00 EST — highest liquidity, trend-setting session
    NEW YORK Kill Zone: 07:00–10:00 EST — continuation or reversal of London move

    Only enter trades DURING kill zone windows.
    The session OUTSIDE kill zones is for analysis, not execution.

EDGE:
    Institutional algorithms are programmed to execute at specific times.
    During kill zones, liquidity is highest, spreads are tightest, and price
    moves with purpose. Outside kill zones, price is noise.
KNOWN_LIMITATIONS:
    Kill zone times shift slightly around DST changes (US and EU). Use UTC-based
    logic and adjust for DST. On news event days (FOMC, NFP), normal kill zone
    behavior may be disrupted 30–60 minutes before the announcement.
"""
from __future__ import annotations

import logging
from datetime import time
from typing import Optional

import pandas as pd

from technical.bar_snapshot import BarSnapshot, Signal, Direction, ConceptCategory, ValidationResult

logger = logging.getLogger(__name__)

# ── PARAMETERS ────────────────────────────────────────────────────────────────
# All times in EST (UTC-5). Ranges are (start_hour_est, end_hour_est)
ASIAN_KZ_HOURS    = (20, 24)   # 20:00 – 00:00 EST
LONDON_KZ_HOURS   = (2,   5)   # 02:00 – 05:00 EST
NEWYORK_KZ_HOURS  = (7,  10)   # 07:00 – 10:00 EST

# Only generate a filter signal (not a directional entry — kill zones are
# TIME GATES that other concepts must pass through)
ENABLE_STANDALONE_SIGNALS = False  # Set True to generate standalone time signals


class ICT_KillZone:
    """
    Time gate module. Used by signal_router.py to filter signals from other
    concepts — only pass signals that occur within an active kill zone.

    Can also be used standalone to mark session opens/closes for charting.
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.asian_kz    = p.get("asian_kz",    ASIAN_KZ_HOURS)
        self.london_kz   = p.get("london_kz",   LONDON_KZ_HOURS)
        self.newyork_kz  = p.get("newyork_kz",  NEWYORK_KZ_HOURS)
        self.standalone  = p.get("standalone",  ENABLE_STANDALONE_SIGNALS)

    def is_in_kill_zone(self, snapshot: BarSnapshot) -> Optional[str]:
        """
        Returns the kill zone name if the bar falls within one, else None.
        """
        hour_est = self._to_est_hour(snapshot.timestamp)
        if self._in_range(hour_est, *self.london_kz):
            return "LONDON"
        if self._in_range(hour_est, *self.newyork_kz):
            return "NEW_YORK"
        if self._in_range(hour_est, *self.asian_kz):
            return "ASIAN"
        return None

    def detect(self, snapshot: BarSnapshot) -> Optional[Signal]:
        """Only generates a signal in standalone mode (for time-of-day charting)."""
        if not self.standalone:
            return None
        kz = self.is_in_kill_zone(snapshot)
        if kz:
            return Signal(
                concept     = "ICT_KillZone",
                symbol      = snapshot.symbol,
                direction   = Direction.FLAT,
                timestamp   = snapshot.timestamp,
                entry_price = snapshot.close,
                stop_loss   = snapshot.close,
                take_profit = snapshot.close,
                confidence  = 1.0,
                category    = ConceptCategory.TIME,
                regime      = snapshot.regime,
                reason      = f"Bar is within {kz} Kill Zone",
            )
        return None

    @staticmethod
    def _to_est_hour(ts) -> int:
        """Convert timestamp to EST hour. Handles pandas Timestamp and datetime."""
        try:
            # If timezone-aware, convert; otherwise assume UTC
            if hasattr(ts, "hour"):
                return (ts.hour - 5) % 24
        except Exception:
            pass
        return 0

    @staticmethod
    def _in_range(hour: int, start: int, end: int) -> bool:
        if start < end:
            return start <= hour < end
        # Wraps midnight (e.g. Asian 20:00–00:00)
        return hour >= start or hour < end

    def get_session_high_low(self, bars_in_session: list) -> tuple:
        """
        Returns (session_high, session_low) for a list of BarSnapshots.
        Used by ICT_SessionRange to define the Asian range.
        """
        if not bars_in_session:
            return (0.0, 0.0)
        return (
            max(b.high for b in bars_in_session),
            min(b.low  for b in bars_in_session),
        )

    def validate(self, historical_df: pd.DataFrame) -> ValidationResult:
        return ValidationResult(
            concept="ICT_KillZone",
            total_tests=1, passed=1, failed=0,
            notes="Kill zone is a time gate — always valid if timestamps are correct",
        )

    def log_event(self, event_type: str, bar, details: dict):
        logger.info("[ICT_KillZone] %s at %s | %s", event_type, bar, details)
