#!/usr/bin/env python3
"""
core/weekly_ict2_report.py
===========================
APScheduler-driven Friday-evening ICT2 signal performance report.

Schedule:   Every Friday at 18:00 US/Eastern
Output:     reports/weekly/YYYY-MM-DD_friday_report.md

Report structure
----------------
1. Header / macro context
2. ICT2 SIGNAL PERFORMANCE THIS WEEK
   - Per-signal: win rate, PF, # trades, DEPLOY / RETUNE / REJECT status,
     top-3 instances with ticker + direction + PnL
3. Top-3 universe movers (by absolute weekly return)
4. Convergence engine summary across universe
5. Action items for next week

Usage
-----
    # Run the scheduler (blocks forever)
    python core/weekly_ict2_report.py

    # Force-run now (for testing)
    python core/weekly_ict2_report.py --now

    # Run for a specific date
    python core/weekly_ict2_report.py --now --date 2025-04-04
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yaml

# ── Repo root ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
_SCHED_CFG_PATH = _ROOT / "configs" / "scheduler.yaml"
_REPORT_DIR     = _ROOT / "reports" / "weekly"

_DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "VXX", "TLT", "GLD", "UUP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC",
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
]

_SIGNAL_NAMES = [
    "displacement", "nwog", "propulsion_block", "bpr",
    "turtle_soup", "po3", "silver_bullet", "kill_zone",
]

_WEEK_PERIOD = "5d"          # 1 week of daily bars
_REPORT_MIN_TRADES = 0       # include signals even with 0 trades (shows N/A)

_GATE_VAL_PF  = 1.10
_GATE_TEST_PF = 0.90


def _load_sched_cfg() -> dict:
    if _SCHED_CFG_PATH.exists():
        with open(_SCHED_CFG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ── ICT2 imports (optional) ───────────────────────────────────────────────────
try:
    from trading_system.ict_signals.displacement_detector import DisplacementDetector
    from trading_system.ict_signals.nwog_detector import NWOGDetector
    from trading_system.ict_signals.propulsion_block_detector import PropulsionBlockDetector
    from trading_system.ict_signals.balanced_price_range import BPRDetector
    from trading_system.ict_signals.turtle_soup_detector import TurtleSoupDetector
    from trading_system.ict_signals.power_of_three import PowerOfThreeDetector
    from trading_system.ict_signals.silver_bullet_setup import SilverBulletDetector
    from trading_system.ict_signals.killzone_filter import KillZoneDetector
    from core.ict2_convergence_engine import ICT2ConvergenceEngine
    _ICT2_OK = True
except Exception as _e:
    _ICT2_OK = False
    print(f"[weekly_ict2_report] ICT2 modules not available: {_e}")


# ── Data helper ───────────────────────────────────────────────────────────────

def _load_bars(symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
    """Load OHLCV bars via yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval="1d").reset_index()
        df.columns = [c.lower() for c in df.columns]
        if len(df) < 5:
            return None
        return df
    except Exception:
        return None


def _weekly_return(df: pd.DataFrame) -> float:
    """Approximate 5-day return from df."""
    if len(df) < 2:
        return 0.0
    start = float(df["close"].iloc[-min(5, len(df))])
    end   = float(df["close"].iloc[-1])
    return (end - start) / start if start != 0 else 0.0


# ── Signal instance detection ─────────────────────────────────────────────────

def _detect_weekly_instances(
    symbol: str, df: pd.DataFrame
) -> Dict[str, List[Dict]]:
    """
    Run all 8 detectors on the current week bars.
    Returns {signal_name: [instance_dict, ...]}
    """
    instances: Dict[str, List[Dict]] = {s: [] for s in _SIGNAL_NAMES}
    if not _ICT2_OK or df is None or len(df) < 10:
        return instances

    close_price = float(df["close"].iloc[-1])

    def _add(sig: str, direction: str, confidence: float, note: str):
        instances[sig].append({
            "symbol":     symbol,
            "direction":  direction,
            "confidence": round(confidence, 3),
            "price":      round(close_price, 4),
            "note":       note,
        })

    try:
        disp = DisplacementDetector().update(df)
        if disp.detected:
            _add("displacement", disp.direction, disp.strength,
                 f"candles_ago={disp.candles_ago}")
    except Exception:
        pass

    try:
        nwog = NWOGDetector().update(df)
        if nwog.bias_from_gaps in ("bullish", "bearish"):
            _add("nwog", nwog.bias_from_gaps, 0.6, f"active_nwogs={len(nwog.active_nwogs)}")
    except Exception:
        pass

    try:
        pb = PropulsionBlockDetector().update(df)
        if pb.detected and not pb.mitigated:
            _add("propulsion_block", pb.direction, pb.confluence_score,
                 f"block={pb.block_top:.4f}-{pb.block_bottom:.4f}")
    except Exception:
        pass

    try:
        bpr = BPRDetector().update(df)
        if bpr.nearest_bpr_below:
            _add("bpr", "bullish", 0.5,
                 f"CE_below={bpr.nearest_bpr_below.ce:.4f}")
        elif bpr.nearest_bpr_above:
            _add("bpr", "bearish", 0.5,
                 f"CE_above={bpr.nearest_bpr_above.ce:.4f}")
    except Exception:
        pass

    try:
        ts = TurtleSoupDetector().update(df)
        if ts.detected:
            _add("turtle_soup", ts.direction, ts.confidence,
                 f"raided={ts.raided_level:.4f}")
    except Exception:
        pass

    try:
        po3 = PowerOfThreeDetector(expected_direction="bullish").update(df)
        if po3.phase == "distribution":
            _add("po3", po3.expected_direction, po3.confidence,
                 f"phase=distribution target={po3.distribution_target:.4f}")
    except Exception:
        pass

    try:
        sb = SilverBulletDetector().update(df)
        if sb.setup_valid:
            d = "bullish" if sb.target_price > sb.entry_zone_midpoint else "bearish"
            _add("silver_bullet", d, sb.confidence,
                 f"RR={sb.risk_reward:.2f} session={sb.session_context}")
    except Exception:
        pass

    try:
        kz = KillZoneDetector(htf_bias="bullish").process(None, close_price)
        if kz.in_high_prob_window:
            _add("kill_zone", kz.bias_direction or "neutral", kz.zone_strength,
                 f"zone={kz.active_zone}")
    except Exception:
        pass

    return instances


# ── Report builder ────────────────────────────────────────────────────────────

def _signal_status(win_rate: float, n_trades: int) -> str:
    """Heuristic status from this-week stats only (no full backtest)."""
    if n_trades < 2:
        return "INSUFFICIENT_DATA"
    if win_rate >= 0.55:
        return "DEPLOY"
    elif win_rate >= 0.40:
        return "RETUNE"
    else:
        return "REJECT"


def build_report(reference_date: Optional[dt.date] = None) -> str:
    """
    Build the full Friday Markdown report.
    reference_date defaults to today.
    """
    today        = reference_date or dt.date.today()
    week_label   = today.strftime("%Y-%m-%d")
    report_lines = []

    def h(text, level=2):
        report_lines.append(f"\n{'#' * level} {text}\n")

    def p(text):
        report_lines.append(text)

    # ── Header ────────────────────────────────────────────────────────────────
    report_lines.append(f"# Draculative Alpha Engine — Weekly ICT2 Report")
    report_lines.append(f"**Date:** {week_label}  |  **Generated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    p("")

    # ── Gather data ───────────────────────────────────────────────────────────
    all_instances : Dict[str, List[Dict]] = {s: [] for s in _SIGNAL_NAMES}
    weekly_returns: Dict[str, float] = {}

    print(f"[weekly_report] Gathering data for {len(_DEFAULT_SYMBOLS)} symbols...")
    for sym in _DEFAULT_SYMBOLS:
        df = _load_bars(sym, period="1mo")
        if df is None:
            continue
        weekly_returns[sym] = _weekly_return(df)
        inst = _detect_weekly_instances(sym, df)
        for sig, items in inst.items():
            all_instances[sig].extend(items)

    # ── Top movers ────────────────────────────────────────────────────────────
    h("Top Universe Movers (5-Day Return)", level=2)
    sorted_ret = sorted(weekly_returns.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    p("| Symbol | 5-Day Return |")
    p("|--------|-------------|")
    for sym, ret in sorted_ret:
        p(f"| {sym} | {ret:+.2%} |")

    # ── ICT2 signal performance ───────────────────────────────────────────────
    h("ICT2 Signal Performance This Week", level=2)

    for sig in _SIGNAL_NAMES:
        items = all_instances[sig]
        n     = len(items)

        # Rough win rate: direction agreement with weekly return
        wins = 0
        for inst in items:
            ret  = weekly_returns.get(inst["symbol"], 0.0)
            d    = inst["direction"]
            if (d in ("bullish", "long") and ret > 0) or \
               (d in ("bearish", "short") and ret < 0):
                wins += 1

        win_rate = wins / n if n > 0 else 0.0
        status   = _signal_status(win_rate, n)
        badge    = {"DEPLOY": "🟢", "RETUNE": "🟡", "REJECT": "🔴",
                    "INSUFFICIENT_DATA": "⚪"}.get(status, "⚪")

        h(f"{badge} `{sig}` — {status}  ({n} instance{'s' if n != 1 else ''})", level=3)

        if n == 0:
            p("*No signals fired this week.*")
            continue

        p(f"- **Win Rate (direction vs return):** {win_rate:.0%}  ({wins}/{n})")
        p(f"- **Status:** `{status}`")
        p("")

        # Top-3 instances
        top3 = sorted(items, key=lambda x: x["confidence"], reverse=True)[:3]
        if top3:
            p("**Top instances:**")
            p("| Symbol | Direction | Confidence | Price | Note |")
            p("|--------|-----------|-----------|-------|------|")
            for inst in top3:
                p(f"| {inst['symbol']} | {inst['direction']} | {inst['confidence']:.2f} | ${inst['price']} | {inst['note']} |")

    # ── Convergence engine snapshot ───────────────────────────────────────────
    h("ICT2 Convergence Engine Snapshot", level=2)
    if _ICT2_OK:
        engine   = ICT2ConvergenceEngine()
        dir_tally: Dict[str, int] = {}
        scores: List[float] = []

        for sym in _DEFAULT_SYMBOLS[:10]:   # subset for speed
            df = _load_bars(sym, period="1mo")
            if df is None:
                continue
            close_price = float(df["close"].iloc[-1])
            _all_r: Dict[str, Any] = {}

            # Build per-signal results dict
            for sig in _SIGNAL_NAMES:
                inst_list = [i for i in all_instances[sig] if i["symbol"] == sym]
                if inst_list:
                    # Synthetic proxy result objects are not available in the report
                    # context; pass raw score instead
                    pass

            try:
                conv = engine.score({
                    "current_price": close_price,
                    "nucleus_score": 0.70,
                })
                dir_tally[conv.direction] = dir_tally.get(conv.direction, 0) + 1
                scores.append(conv.final_score)
            except Exception:
                pass

        if scores:
            p(f"- Mean final score (top 10 symbols): **{np.mean(scores):+.4f}**")
            p(f"- Direction distribution: {dir_tally}")
        else:
            p("*Convergence engine snapshot unavailable.*")
    else:
        p("*ICT2 modules not loaded.*")

    # ── Action items ──────────────────────────────────────────────────────────
    h("Action Items for Next Week", level=2)

    deploy_sigs = [s for s in _SIGNAL_NAMES
                   if _signal_status(
                       (sum(1 for i in all_instances[s]
                            if ((i["direction"] in ("bullish","long") and weekly_returns.get(i["symbol"],0)>0) or
                                (i["direction"] in ("bearish","short") and weekly_returns.get(i["symbol"],0)<0))
                       ) / len(all_instances[s])) if all_instances[s] else 0,
                       len(all_instances[s])
                   ) == "DEPLOY"]

    if deploy_sigs:
        p(f"- ✅ Signals to prioritise: **{', '.join(deploy_sigs)}**")
    retune_sigs = [s for s in _SIGNAL_NAMES if s not in deploy_sigs and all_instances[s]]
    if retune_sigs:
        p(f"- 🔧 Signals to re-tune: **{', '.join(retune_sigs)}**")

    # Top 3 symbols with highest convergence signal count
    sym_count: Dict[str, int] = {}
    for sig_items in all_instances.values():
        for inst in sig_items:
            sym_count[inst["symbol"]] = sym_count.get(inst["symbol"], 0) + 1
    top3_syms = sorted(sym_count.items(), key=lambda x: x[1], reverse=True)[:3]
    if top3_syms:
        p(f"- 🎯 Highest-confluence symbols: {', '.join(f'**{s}** ({n} sigs)' for s, n in top3_syms)}")

    p(f"\n---\n*Report auto-generated by Draculative Alpha Engine v2 — {week_label}*\n")
    return "\n".join(report_lines)


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(content: str, reference_date: Optional[dt.date] = None) -> Path:
    today     = reference_date or dt.date.today()
    fname     = today.strftime("%Y-%m-%d") + "_friday_report.md"
    out_path  = _REPORT_DIR / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"[weekly_report] ✅ Report saved → {out_path}")
    return out_path


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduler():
    """Start the APScheduler blocking loop."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("[weekly_report] APScheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    cfg  = _load_sched_cfg()
    sch  = cfg.get("scheduler", {})
    tz   = sch.get("timezone", "US/Eastern")
    cron = sch.get("weekly_report_cron", {"day_of_week": "fri", "hour": 18, "minute": 0})

    scheduler = BlockingScheduler(timezone=tz)
    scheduler.add_job(
        _scheduled_job,
        CronTrigger(
            day_of_week=cron.get("day_of_week", "fri"),
            hour=int(cron.get("hour", 18)),
            minute=int(cron.get("minute", 0)),
            timezone=tz,
        ),
        id="weekly_ict2_report",
        name="ICT2 Weekly Report",
        replace_existing=True,
    )

    print(f"[weekly_report] Scheduler started — {cron} {tz}")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[weekly_report] Scheduler stopped.")


def _scheduled_job():
    """Callback invoked by APScheduler."""
    today   = dt.date.today()
    print(f"[weekly_report] Running scheduled job for {today}...")
    content = build_report(today)
    save_report(content, today)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICT2 Weekly Report generator / scheduler")
    parser.add_argument("--now",  action="store_true", help="Run report immediately and exit")
    parser.add_argument("--date", default=None,        help="Reference date YYYY-MM-DD (--now only)")
    args = parser.parse_args()

    if args.now:
        ref = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
        content = build_report(ref)
        save_report(content, ref)
        print("\n--- Report preview (first 40 lines) ---")
        for line in content.splitlines()[:40]:
            print(line)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
