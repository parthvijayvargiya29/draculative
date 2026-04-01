"""
weekly_report.py — Friday 6pm ET Automated Weekly Report Generator.

Format matches Section 2.5 exactly:
  1. Market Regime Summary
  2. Concept Performance Scorecard
  3. New Signals Logged This Week
  4. Walk-Forward Gate Status (pending → approved / rejected)
  5. Nucleus Distribution Chart (text)
  6. Risk Management Status (phase, draws, remaining capacity)
  7. Recommended Concepts for Next Week
  8. Macro Outlook for Next Week

Output: reports/weekly/YYYY-MM-DD_weekly_report.txt
        reports/weekly/YYYY-MM-DD_weekly_report.json (machine-readable)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fundamental.macro_tracker import MacroTracker
from fundamental.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/weekly")


@dataclass
class WeeklyReport:
    week_ending:         str
    generated_at:        str
    market_regime:       str
    macro_vix:           float = 0.0
    macro_spy_vs_200:    float = 0.0

    # Concept scorecard: {concept_name: {pf, win_rate, trades, status}}
    concept_scorecard:   Dict[str, dict] = field(default_factory=dict)

    # Signals this week: list of signal dicts
    signals_this_week:   List[dict] = field(default_factory=list)

    # Walk-forward results this week
    wf_results:          List[dict] = field(default_factory=list)

    # Nucleus distribution: {nucleus_type: count}
    nucleus_distribution: Dict[str, int] = field(default_factory=dict)

    # Risk management
    risk_phase:           str = "EVAL"
    weekly_pnl:           float = 0.0
    open_drawdown_pct:    float = 0.0
    trades_remaining:     int = 0

    # Recommended for next week
    recommended_concepts: List[str] = field(default_factory=list)
    concepts_on_watch:    List[str] = field(default_factory=list)

    # Macro outlook
    upcoming_events:      List[str] = field(default_factory=list)
    macro_outlook:        str = ""


class WeeklyReportGenerator:
    """
    Compiles all available data into a WeeklyReport and writes
    formatted text + JSON to disk.
    """

    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.macro = MacroTracker()
        self.news  = NewsFetcher()

    def generate(
        self,
        simulation_results: List = None,
        wf_results:         List = None,
        signal_log:         List[dict] = None,
        nucleus_distribution: Dict[str, int] = None,
        risk_phase:         str = "EVAL",
        weekly_pnl:         float = 0.0,
        open_drawdown_pct:  float = 0.0,
        trades_remaining:   int = 0,
    ) -> WeeklyReport:
        today       = datetime.utcnow().strftime("%Y-%m-%d")
        macro_state = self.macro.get_state()
        upcoming    = self.news.get_high_impact(days_ahead=7)

        report = WeeklyReport(
            week_ending        = today,
            generated_at       = datetime.utcnow().isoformat(),
            market_regime      = macro_state.regime.value,
            macro_vix          = macro_state.vix,
            macro_spy_vs_200   = round(macro_state.spy_vs_200 * 100, 2),
            nucleus_distribution = nucleus_distribution or {},
            risk_phase         = risk_phase,
            weekly_pnl         = weekly_pnl,
            open_drawdown_pct  = open_drawdown_pct,
            trades_remaining   = trades_remaining,
            upcoming_events    = [f"{e.date} {e.time_et} — {e.title}" for e in upcoming],
        )

        # Concept scorecard from simulation results
        if simulation_results:
            for sr in simulation_results:
                if hasattr(sr, "concept"):
                    report.concept_scorecard[sr.concept] = {
                        "profit_factor": round(sr.profit_factor, 2),
                        "win_rate":      round(sr.win_rate * 100, 1),
                        "total_trades":  sr.total_trades,
                        "sharpe":        round(sr.sharpe_annualized, 2),
                        "max_dd":        round(sr.max_drawdown_pct * 100, 2),
                        "status":        sr.approval_status.value if hasattr(sr.approval_status, "value") else str(sr.approval_status),
                    }

        # Walk-forward gate results
        if wf_results:
            for wfr in wf_results:
                report.wf_results.append({
                    "concept":      getattr(wfr, "concept", "?"),
                    "pf_train":     round(getattr(wfr, "pf_train", 0), 2),
                    "pf_test":      round(getattr(wfr, "pf_test", 0), 2),
                    "wfe":          round(getattr(wfr, "wfe", 0), 2),
                    "passes_gate":  getattr(wfr, "passes_gate", False),
                })

        # Signal log this week
        report.signals_this_week = signal_log or []

        # Recommendations
        report.recommended_concepts = [
            c for c, s in report.concept_scorecard.items()
            if s.get("profit_factor", 0) >= 1.5 and s.get("win_rate", 0) >= 45
        ]
        report.concepts_on_watch = [
            c for c, s in report.concept_scorecard.items()
            if 1.0 < s.get("profit_factor", 0) < 1.5
        ]

        # Macro outlook text
        report.macro_outlook = self._macro_outlook_text(macro_state, upcoming)

        # Write outputs
        txt_path  = REPORTS_DIR / f"{today}_weekly_report.txt"
        json_path = REPORTS_DIR / f"{today}_weekly_report.json"
        txt_path.write_text(self._format_text(report))
        json_path.write_text(json.dumps(asdict(report), indent=2))
        logger.info("Weekly report written → %s", txt_path)

        return report

    # ── Formatting ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_text(r: WeeklyReport) -> str:
        lines = [
            "=" * 70,
            f"  DRACULATIVE ALPHA ENGINE — WEEKLY REPORT",
            f"  Week ending: {r.week_ending}   Generated: {r.generated_at}",
            "=" * 70,
            "",
            "1. MARKET REGIME",
            f"   Regime : {r.market_regime}",
            f"   VIX    : {r.macro_vix:.1f}",
            f"   SPY vs SMA200 : {r.macro_spy_vs_200:+.1f}%",
            "",
            "2. CONCEPT PERFORMANCE SCORECARD",
        ]
        if r.concept_scorecard:
            lines.append(f"   {'Concept':<35} {'PF':>6} {'WR%':>6} {'Trades':>7} {'Sharpe':>8} {'MaxDD%':>8} {'Status':<12}")
            lines.append("   " + "-" * 86)
            for name, s in r.concept_scorecard.items():
                lines.append(
                    f"   {name:<35} {s.get('profit_factor',0):>6.2f} "
                    f"{s.get('win_rate',0):>6.1f} "
                    f"{s.get('total_trades',0):>7d} "
                    f"{s.get('sharpe',0):>8.2f} "
                    f"{s.get('max_dd',0):>8.2f} "
                    f"{s.get('status','?'):<12}"
                )
        else:
            lines.append("   (no simulation data this week)")

        lines += [
            "",
            "3. WALK-FORWARD GATE RESULTS",
        ]
        if r.wf_results:
            for w in r.wf_results:
                gate = "✅ PASS" if w.get("passes_gate") else "❌ FAIL"
                lines.append(
                    f"   {w['concept']:<35} train_PF={w['pf_train']:.2f}  "
                    f"test_PF={w['pf_test']:.2f}  WFE={w['wfe']:.2f}  {gate}"
                )
        else:
            lines.append("   (no walk-forward runs this week)")

        lines += [
            "",
            "4. SIGNALS LOGGED THIS WEEK",
            f"   Total signals: {len(r.signals_this_week)}",
            "",
            "5. NUCLEUS DISTRIBUTION",
        ]
        for nt, cnt in sorted(r.nucleus_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt, 30)
            lines.append(f"   {nt:<30} {bar} ({cnt})")

        lines += [
            "",
            "6. RISK MANAGEMENT STATUS",
            f"   Phase            : {r.risk_phase}",
            f"   Weekly P&L       : ${r.weekly_pnl:,.2f}",
            f"   Open Drawdown    : {r.open_drawdown_pct:.1f}%",
            f"   Trades Remaining : {r.trades_remaining}",
            "",
            "7. RECOMMENDED CONCEPTS FOR NEXT WEEK",
        ]
        for c in r.recommended_concepts:
            lines.append(f"   ✅  {c}")
        if not r.recommended_concepts:
            lines.append("   (none meet threshold PF ≥ 1.5, WR ≥ 45%)")

        lines += [
            "",
            "8. MACRO OUTLOOK",
            f"   {r.macro_outlook}",
            "",
        ]
        if r.upcoming_events:
            lines.append("   UPCOMING HIGH-IMPACT EVENTS:")
            for ev in r.upcoming_events:
                lines.append(f"     📅  {ev}")

        lines += ["", "=" * 70, "  END OF REPORT", "=" * 70]
        return "\n".join(lines)

    @staticmethod
    def _macro_outlook_text(macro_state, upcoming) -> str:
        regime = macro_state.regime.value
        vix    = macro_state.vix
        if regime == "CRISIS":
            return "CRISIS mode — no discretionary risk. Cash is a position."
        if regime == "RISK_OFF":
            return f"Risk-off environment (VIX={vix:.1f}). Reduce size. Prefer shorts or cash."
        if regime == "RISK_ON":
            return f"Risk-on environment (VIX={vix:.1f}). Trend strategies favoured."
        return f"Neutral macro (VIX={vix:.1f}). Wait for clearer regime signal."
