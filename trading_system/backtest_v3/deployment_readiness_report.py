#!/usr/bin/env python3
"""
deployment_readiness_report.py  --  V4.0 Deployment Readiness Report
======================================================================
Generates the Section 10 deployment readiness report before Day 1 of live
trading. Run once, sign off every checklist item, then start the clock.

USAGE
-----
  python deployment_readiness_report.py
  python deployment_readiness_report.py --output deployment_readiness_report.txt

OUTPUT SECTIONS
---------------
  1. Deployment path decision
  2. Pine Script verification checklist (items to check manually in TradingView)
  3. Per-instrument trade count table (backtest vs Pine Script -- fill in Pine counts)
  4. Parameter lock confirmation
  5. Hard risk limits summary
  6. Condensed daily routine checklist (one-page reference)
  7. Pre-entry checklist (condensed)
  8. Go/no-go timeline
  9. Emergency protocol (manual exits without Pine Script)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# Backtest per-instrument counts (source: bt34_comboC_trades_20260319_1806.csv)
BACKTEST_TOTAL = {
    "GLD": 12, "WMT": 10, "USMV": 18, "NVDA": 3,
    "AMZN": 12, "GOOGL": 12, "COST": 11, "XOM": 10,
    "HD": 21, "MA": 22,
}

INSTRUMENTS = ["GLD", "WMT", "USMV", "NVDA", "AMZN", "GOOGL", "COST", "XOM", "HD", "MA"]

REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
BACKTEST_PERIOD = "Apr 2023 – Mar 2026"
TEST_PERIOD = "Aug 2025 – Mar 2026"


def generate_report() -> str:
    lines = []

    def h1(title):
        lines.append("\n" + "=" * 80)
        lines.append(f"  {title}")
        lines.append("=" * 80)

    def h2(title):
        lines.append(f"\n  --- {title} ---")

    def item(check: bool | None, text: str):
        if check is True:
            icon = "[X]"
        elif check is False:
            icon = "[ ]"
        else:
            icon = "[ ]"
        lines.append(f"  {icon} {text}")

    # Header
    lines.append("=" * 80)
    lines.append("  COMBO C V4.0 — DEPLOYMENT READINESS REPORT")
    lines.append(f"  Generated: {REPORT_DATE}")
    lines.append("  Status: PENDING (complete all [ ] items before Day 1)")
    lines.append("=" * 80)

    # --------------------------------------------------------------------------
    # SECTION 1 — Deployment Path
    # --------------------------------------------------------------------------
    h1("SECTION 1: DEPLOYMENT PATH DECISION")
    lines.append("""
  CHOSEN PATH: C — Direct live deployment at 20% position size.

  Rationale:
  - 131-trade backtest on 36 months of clean data, WFE=0.724, five development
    versions with confirmed parameter stability constitutes sufficient prior
    evidence for edge existence.
  - Path A (full paper trading) requires 6-9 months to reach N=30 at the
    observed signal rate of 3.6 trades/month — that is not a 60-day process.
    60 days of paper trading yields ~7 trades, far below the N=30 statistical
    minimum, producing no meaningful additional validation.
  - Path B (60-day paper + judgment entry) obscures the exact moment real risk
    begins and does not substantially improve the statistical picture.
  - Path C exposes real capital to a 20% position size from Day 1 while
    maintaining the N=30 statistical validation gate before any ramp-up.
    The first 30 live trades ARE the formal validation period.

  PATH C PREREQUISITE (GATE — must pass before Day 1):
    Pine Script per-instrument trade count within +/-2 of backtest for all
    10 instruments. See Section 3. Do not start the clock until every
    instrument passes the count verification.

  POSITION SIZE SCHEDULE:
    Day 1 through N=30 live trades:  20% of target size
      target = floor(equity * 0.005 / ATR(10))
      20%    = floor(equity * 0.001 / ATR(10))

    After N=30 FULL GO verdict:
      Month 1 post-go:  50% of target (floor(equity * 0.0025 / ATR(10)))
      Month 2 post-go:  75% if M1 PF>=1.0 and DD<15%
      Month 3 post-go: 100% if M2 PF>=1.0 and DD<15%
    Two consecutive months PF<0.8 or single month DD>20%: hold current size.
""")

    # --------------------------------------------------------------------------
    # SECTION 2 — Pine Script Verification Checklist
    # --------------------------------------------------------------------------
    h1("SECTION 2: PINE SCRIPT VERIFICATION CHECKLIST")
    lines.append("  Complete every item before loading instruments for Day 1.")
    lines.append("")

    h2("Signal Condition Verification")
    for label, expected in [
        ("BB: ta.bb(close, 20, 2.0)", "period 20, multiplier 2.0"),
        ("RSI: ta.rsi(close, 2)", "period 2, NOT 14"),
        ("RSI threshold: rsi_val < 15", "15, NOT 10"),
        ("Volume gate: NOT PRESENT", "no cond_volume in entry logic"),
        ("ATR: ta.atr(10)", "period 10"),
        ("ACCEL_SL: bb_lower - (1.0 * atr10)", "static floor, non-ratcheting, non-var"),
        ("ACCEL_SL recomputed every bar", "no 'var accel_sl_armed' state variable"),
        ("Stop: close <= accel_sl triggers exit", "on close bar, not following bar"),
        ("Time stop: exit on bar_count == 10", "bar 1 = entry bar; exit on bar 10"),
        ("Target: close >= bb_basis (20-period SMA)", "evaluated on bar close"),
        ("Entry: fills at NEXT bar open", "strategy.entry fires on signal close"),
        ("Position size: floor(equity*0.005/atr)", "0.5% equity at risk per trade"),
        ("Position cap: position*close <= equity*0.10", "10% equity max per instrument"),
        ("Initial capital: 2500 per chart", "in strategy() call"),
        ("Commission: 1.0 per order", "per order, not per share"),
        ("Adjusted data: ENABLED in chart settings", "all 10 charts"),
        ("Timeframe: DAILY bars", "all 10 charts"),
    ]:
        item(None, f"{label:<50}  expect: {expected}")

    h2("ACCEL_SL Plot Verification")
    for chk in [
        "ACCEL_SL line plots ONLY when a position is open (not on flat bars)",
        "ACCEL_SL value matches bb_lower - 1.0 * ATR(10) visually (spot check 3 bars)",
        "ACCEL_SL line resets / disappears when position closes",
        "ACCEL_SL exit fires on the CLOSE bar where close < accel_sl (not following bar)",
    ]:
        item(None, chk)

    h2("Info Table Verification")
    for chk in [
        "Table shows 6 rows (no volume row — removed in V3.5 fix)",
        "Table updates in real-time on each bar close",
        "All displayed values match computed indicator values (spot-check 3 recent bars)",
    ]:
        item(None, chk)

    h2("Priority Order Verification (exit logic)")
    lines.append("""  Priority order in Pine Script must match backtest exactly:
    1. Hard SL: close <= (entry - 1*ATR)    -> 'SL'     [fires on CLOSE, fills OPEN]
    2. ACCEL_SL: close <= (bb_lower-1*ATR)  -> 'ACCEL_SL'
    3. BB_MID: close >= bb_basis            -> 'BB_MID'
    4. Time stop: bar_count >= 10           -> 'TIME'
  Note: In practice only ACCEL_SL/BB_MID/TIME appear in live results
  because SL and ACCEL_SL levels converge quickly after entry.""")

    # --------------------------------------------------------------------------
    # SECTION 3 — Per-Instrument Trade Count Verification
    # --------------------------------------------------------------------------
    h1("SECTION 3: PER-INSTRUMENT TRADE COUNT VERIFICATION")
    lines.append(f"  Backtest period: {BACKTEST_PERIOD}")
    lines.append("  In TradingView: load combo_c_pine.pine, run Strategy Tester over")
    lines.append("  same date range (Apr 2023 - Mar 2026), record trade count per chart.")
    lines.append("")
    lines.append(f"  {'Symbol':<8} {'Backtest':<12} {'Pine Script':<15} {'Delta':<10} {'Pass?'}")
    lines.append(f"  {'------':<8} {'--------':<12} {'-----------':<15} {'-----':<10} {'-----'}")
    for sym in INSTRUMENTS:
        bt = BACKTEST_TOTAL[sym]
        lines.append(f"  {sym:<8} {bt:<12} {'[ fill in ]':<15} {'[ calc ]':<10} {'[ ] <=+-2'}")
    lines.append("")
    lines.append("  GATE: ALL instruments must show |Pine - Backtest| <= 2 before Day 1.")
    lines.append("")
    lines.append("  Common discrepancy causes:")
    lines.append("    - TradingView using UNADJUSTED prices -> enable 'Adjusted data'")
    lines.append("    - Bar indexing: backtest bars include signal date; Pine Script")
    lines.append("      bar_count may differ by 1 -> acceptable within +-2 tolerance")
    lines.append("    - RSI initialization: first RSI(2) bar differs between engines")
    lines.append("      by construction -- no fix needed if within +-2 per instrument")

    # --------------------------------------------------------------------------
    # SECTION 4 — Parameter Lock
    # --------------------------------------------------------------------------
    h1("SECTION 4: PARAMETER LOCK CONFIRMATION")
    lines.append("  Locked at Phase 1 decision, 19-Mar-2026. IMMUTABLE during trading.")
    lines.append("  Do NOT modify any parameter until annual strategy review triggers a")
    lines.append("  fresh backtest AND the fresh backtest confirms the change improves OOS PF.")
    lines.append("")
    params = [
        ("BB_PERIOD",       "20"),
        ("BB_STD_MULT",     "2.0"),
        ("RSI_PERIOD",      "2"),
        ("RSI_THRESHOLD",   "15.0  (< 15, not <= 15)"),
        ("VOLUME_GATE",     "NONE  (confirmed anti-correlated, removed)"),
        ("ATR_PERIOD",      "10"),
        ("STOP_LOSS",       "entry_price - 1.0 * ATR(10)  [hard stop, set at entry]"),
        ("ACCEL_SL",        "bb_lower - 1.0 * ATR(10)     [static floor, each bar]"),
        ("TIME_STOP_BARS",  "10   (exit on bar 10 from entry)"),
        ("TARGET",          "bb_basis (20-period SMA)"),
        ("POS_SIZE",        "floor(equity * 0.005 / ATR(10))"),
        ("POS_CAP",         "10% of equity per instrument"),
        ("MAX_CONCURRENT",  "4 positions across all instruments"),
        ("COMMISSION",      "$1.00 per order (round-trip = $2.00)"),
        ("INITIAL_CAP",     "$2,500 per chart in TradingView"),
        ("INSTRUMENTS",     "GLD  WMT  USMV  NVDA  AMZN  GOOGL  COST  XOM  HD  MA"),
        ("DEPLOYMENT_PATH", "C -- 20% size until N=30 FULL GO, then ramp schedule"),
    ]
    max_k = max(len(k) for k, _ in params)
    for k, v in params:
        lines.append(f"  {k:<{max_k+2}}: {v}")

    lines.append("")
    lines.append("  Phase 1 variant test summary (locked decision rationale):")
    lines.append("    Baseline (RSI=15, no vol, static SL): Test_PF=1.112, WFE=0.724, N_Test=34")
    lines.append("    V1_RSI10  (RSI=10):  Test_PF=0.902  -- worse, -0.210")
    lines.append("    V2_RSI5   (RSI=5):   Test_PF=1.199  -- +0.087, below +0.10 threshold")
    lines.append("    V3_VolGate (vol):    Test_PF=0.000, N_Test=2  -- catastrophic")
    lines.append("    V4_RatchetSL:        Test_PF=1.088  -- worse, -0.024")
    lines.append("    Decision: Baseline is optimal. No variant met the +0.10 improvement bar.")

    # --------------------------------------------------------------------------
    # SECTION 5 — Hard Risk Limits
    # --------------------------------------------------------------------------
    h1("SECTION 5: HARD RISK LIMITS (NEVER VIOLATED)")
    limits = [
        ("Per-trade max loss",
         "0.5% of equity. Formula: floor(equity*0.005/ATR)*ATR = equity*0.005.\n"
         "    Flag if actual (entry - accel_sl) * shares / equity > 0.8%."),
        ("Per-instrument max exposure",
         "10% of equity. Check: shares * entry_price <= equity * 0.10."),
        ("Total portfolio at-risk",
         "sum((entry-stop)*shares) across all open positions < 5% of equity.\n"
         "    If above 4%, do not open new positions."),
        ("Max concurrent positions",
         "4. If >4 signals fire simultaneously, take 4 lowest RSI(2) instruments."),
        ("Monthly loss limit",
         "8% of equity in any calendar month -> halt new entries for rest of month.\n"
         "    Let existing positions close. Resume first day of next month."),
        ("Annual loss limit",
         "15% of equity in any rolling 12 months -> FULL STRATEGY PAUSE.\n"
         "    Run complete annual strategy review before resuming."),
        ("Drawdown escalation",
         "DD <  8%: normal trading\n"
         "    DD  8-15%: reduce to 50% size if trending regime\n"
         "    DD 15-25%: halt entries, run 18-month fresh backtest\n"
         "    DD > 25%: FULL PAUSE, triggers go/no-go failure"),
        ("Correlated cluster sizing",
         "NVDA/AMZN/GOOGL and COST/WMT/HD: if 2+ instruments in same cluster\n"
         "    signal simultaneously, enter each at 50% position size (not full)."),
    ]
    for name, desc in limits:
        lines.append(f"\n  [{name}]")
        for dl in desc.split("\n"):
            lines.append(f"    {dl}")

    # --------------------------------------------------------------------------
    # SECTION 6 — Daily Routine (condensed, one-page reference)
    # --------------------------------------------------------------------------
    h1("SECTION 6: DAILY ROUTINE CHECKLIST (POST-MARKET, ~15 MINUTES)")
    lines.append("""
  STEP 1 — Open position audit (5 min)
  For each open position:
    a. Record bar count since entry
    b. Compute today's ACCEL_SL: current bb_lower - 1*ATR(10)
    c. Record distance to BB_MID: bb_basis - current_close
    d. FLAG if bar_count >= 8 (TIME STOP in 1-2 bars)
    e. FLAG if close within 0.5*ATR of hard stop

  STEP 2 — Signal proximity scan (5 min)
  For each flat instrument:
    a. Distance from close to bb_lower (in ATR units):
       > 2*ATR: not watching
       1-2*ATR: watch zone
       < 1*ATR: HIGH ALERT - signal possible tomorrow
    b. RSI(2) value: < 20 = watch closely; < 15 = condition met

  STEP 3 — Exit preparation (5 min)
  For each open position:
    a. Profit target level = current bb_basis
    b. ACCEL_SL level = current bb_lower - 1*ATR(10)
    c. If bar_count = 10 today: TIME STOP fires on today's close -> prepare sell order

  STEP 4 — Entry execution (if signal fired)
    1. Confirm signal: close < bb_lower AND RSI(2) < 15 on closed bar
    2. Run pre-entry checklist (Section 7 below)
    3. Calculate shares: floor(equity * 0.0010 / atr) [20% of target during ramp]
    4. Verify: shares * next_open_est <= equity * 0.10 (position cap)
    5. Prepare market order for next morning open
    6. DO NOT place order tonight -- confirm open price first at market open

  STEP 5 — Post-close logging
    Log exits: python paper_trading_monitor.py log [options]
    Update open positions: python paper_trading_monitor.py open list
    Run check: python paper_trading_monitor.py check
    Log any skipped signals: python paper_trading_monitor.py decide --type SKIP ...
""")

    # --------------------------------------------------------------------------
    # SECTION 7 — Pre-Entry Checklist
    # --------------------------------------------------------------------------
    h1("SECTION 7: PRE-ENTRY CHECKLIST (RUN BEFORE EVERY ENTRY)")
    lines.append("  ALL items must be checked. Missing any one is not acceptable.")
    lines.append("")
    pre_entry = [
        ("Earnings",
         "Earnings announcement within 5 trading days? -> SKIP entry.\n"
         "     Earnings create binary outcome risk not present in the backtest.\n"
         "     Check: earnings calendar for the symbol before every entry."),
        ("Ex-dividend",
         "Ex-dividend date within 3 trading days? -> Evaluate and likely SKIP.\n"
         "     XOM, HD, WMT, COST: large dividend drops look like BB breaks.\n"
         "     Check: dividend calendar for the symbol before every entry."),
        ("Corporate action",
         "Merger, spin-off, stock split pending? -> SKIP.\n"
         "     Check: news search for symbol name."),
        ("Company-specific news",
         "Did a specific company event (recall, lawsuit, CEO departure) cause\n"
         "     this BB break? -> SKIP. These are out-of-distribution for backtest.\n"
         "     Only enter on broad market mean-reversion conditions."),
        ("Concurrent positions",
         "How many positions are currently open? If 4 -> do not open more.\n"
         "     If 3 open and this would make 4, run total at-risk check first."),
        ("Total at-risk",
         "sum((entry-stop)*shares)/equity < 5% after adding this position.\n"
         "     Compute before placing order."),
    ]
    for name, desc in pre_entry:
        lines.append(f"  [ ] {name}")
        for dl in desc.split("\n"):
            lines.append(f"      {dl}")
        lines.append("")

    # --------------------------------------------------------------------------
    # SECTION 8 — Go/No-Go Timeline
    # --------------------------------------------------------------------------
    h1("SECTION 8: GO/NO-GO TIMELINE")
    lines.append(f"""
  Deployment path: C -- live at 20% size from Day 1.
  Go/no-go gate applies to first 30 LIVE trades.

  Signal rate benchmark: ~3.6 trades/month (131 trades over 36 months, 10 instruments)
  Minimum sample: N = 30 live trades
  Expected time to N=30: 30 / 3.6 = 8.3 months from Day 1

  TIMELINE MILESTONES:
    Day 1         : 20% position size begins
    N = 10 trades : First informal drift check (not a go/no-go -- premature)
    N = 20 trades : Mid-point review; flag anything outside 30% of backtest
    N = 30 trades : FORMAL GO/NO-GO EVALUATION
                     All 5 required criteria evaluated simultaneously:
                       (1) PF >= 1.10
                       (2) WR >= 38%
                       (3) WR <= 68%
                       (4) Max drawdown < 25%
                       (5) N >= 30 [this criterion automatically passes at N=30]
                     Outcome:
                       FULL GO     -> ramp to 50%, then 75%, then 100% per schedule
                       CONDITIONAL -> continue 15 more trades at 20%, reassess
                       NO-GO       -> halt entries, diagnose, do not ramp

  OPTIMISTIC SCENARIO (market volatile, ~5 trades/month):
    N=30 in ~6 months from Day 1

  CONSERVATIVE SCENARIO (market calm, ~2 trades/month):
    N=30 in ~15 months from Day 1

  REGIME ISSUE TRIGGER:
    If N < 30 after 9 months, classify as REGIME ISSUE.
    Document market conditions. Do not abandon strategy without regime analysis.
    Signal frequency below 2/month suggests trending/calm regime unfavorable to
    BB mean-reversion. Wait for corrective/choppy regime rather than abandoning.

  MONTHLY LOSS LIMIT INTERACTION:
    Monthly loss limit (8% of equity) can pause new entries within a month.
    This does not reset the N count -- trades already logged still count toward N=30.
    Annual loss limit (15%) triggers full strategy pause and resets the review clock.

  IMPORTANT: go/no-go cannot be evaluated early. At N=15, even perfect metrics
  are statistically meaningless. Do not increase position size before N=30.
""")

    # --------------------------------------------------------------------------
    # SECTION 9 — Emergency Protocol
    # --------------------------------------------------------------------------
    h1("SECTION 9: EMERGENCY PROTOCOL")
    lines.append("""
  SCENARIO A: Broker platform goes down mid-trade
  ------------------------------------------------
  Manual exit calculation (no Pine Script needed):
    ACCEL_SL = (today's BB lower band) - 1.0 * (today's ATR(10))
    BB lower = close - 2.0 * StdDev(close, 20)   [20-bar std dev * 2]
    ATR(10)  = EWM of |H-L, H-prev_close, L-prev_close| over 10 bars (Wilder)

  Practical method without calculation:
    1. Look up the 20-period Bollinger Band lower in any financial platform
       (Yahoo Finance, Google Finance, Bloomberg -- all show BB(20,2))
    2. Look up ATR(10) (available on Yahoo Finance or TradingView free tier)
    3. ACCEL_SL = BB_lower - ATR(10)
    4. If current price < ACCEL_SL -> place market sell order immediately
    5. If bar count = 10 or later -> place market sell order immediately
    6. If price >= BB_basis (middle band = 20-day SMA) -> place market sell

  If broker is completely down (cannot place orders):
    1. Call broker's trade desk phone line immediately
    2. Give: symbol, direction (sell), shares, order type (market)
    3. Note confirmation number and rep name
    4. Log the exit in paper_trading_monitor.py as exit_reason=MANUAL_OVERRIDE

  SCENARIO B: TradingView inaccessible, need to check signal
  -----------------------------------------------------------
  Manual signal check (no TradingView needed):
    Entry condition: close < BB_lower(20,2) AND RSI(2) < 15
    BB_lower = SMA(20) - 2 * StdDev(close, 20)  [last 20 bars]
    RSI(2) = 100 - 100 / (1 + avg_gain_2 / avg_loss_2)
             where avg_gain/loss are 2-bar Wilder averages

  Practical alternative:
    Use Yahoo Finance or Google Finance charts with BB(20,2) overlay.
    RSI(2) is less common -- use stockanalysis.com or barchart.com for RSI(2).

  SCENARIO C: Position size calculation error discovered after fill
  ----------------------------------------------------------------
  If you realize you entered more shares than intended:
    DO NOT immediately sell the excess to 'correct' it.
    Let the position run to its natural exit (time stop / ACCEL_SL / BB_MID).
    Log the actual shares in the trade log.
    Log the discrepancy in paper_trading_monitor.py decide:
      python paper_trading_monitor.py decide --type SIZE_ADJUST \\
             --detail "Entered N shares, intended M shares, reason: ..."
    After the trade closes, add an explicit position size calculation step
    to your pre-market routine to prevent recurrence.

  SCENARIO D: Earnings discovered AFTER entry (should have been caught pre-entry)
  --------------------------------------------------------------------------------
  If earnings were not in the calendar at entry time but announcement is made:
    1. Do NOT hold through earnings -- exit at market immediately
    2. Record as MANUAL_OVERRIDE with note "Earnings discovered post-entry"
    3. This trade is excluded from go/no-go PF/WR calculations (contaminated)
    4. Add a secondary earnings check (e.g., earningswhispers.com) to
       pre-entry checklist to catch late-added announcements.

  CONTACTS TO KEEP ON HAND
  -------------------------
  Alpaca broker trade desk: See Alpaca support at alpaca.markets/support
  Alternative charting (no TradingView): stockanalysis.com, barchart.com
  BB(20,2) alternative: finance.yahoo.com -> symbol -> Chart -> Indicators -> BB
  ATR(10) alternative:  finance.yahoo.com -> symbol -> Chart -> Indicators -> ATR
""")

    # --------------------------------------------------------------------------
    # SECTION 10 — Sign-Off
    # --------------------------------------------------------------------------
    h1("SECTION 10: PRE-DAY-1 SIGN-OFF")
    lines.append("""
  Complete this sign-off after all checklist items above are verified.

  [ ] Section 2: All Pine Script parameters verified
  [ ] Section 3: All 10 instruments pass trade count check (within +-2)
  [ ] Section 4: Parameter lock confirmed -- no deviations from spec
  [ ] Section 5: Hard risk limits understood and operational procedures in place
  [ ] Section 6: Daily routine printed or accessible
  [ ] Section 7: Pre-entry checklist printed or accessible
  [ ] Section 8: Go/no-go timeline understood
  [ ] Section 9: Emergency protocol reviewed and contacts noted

  All items above checked? -> START THE CLOCK. Log first live trade with:
    python paper_trading_monitor.py log \\
        --symbol [SYM] --period LIVE_RAMP \\
        --signal-date [DATE] ... --earnings-checked --dividend-checked --news-checked

  Signed: _________________________ Date: _____________

  REMINDER: Path C -- first 30 live trades at 20% position size.
            Go/no-go ONLY at N>=30. Do not ramp before that gate passes.
""")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Generate V4.0 deployment readiness report")
    p.add_argument("--output", default=None,
                   help="Save to file (default: print to stdout)")
    args = p.parse_args()

    report = generate_report()

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Report written to {out_path} ({out_path.stat().st_size} bytes)")
    else:
        print(report)

    # Also save a timestamped copy in the backtest directory
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    auto = HERE / f"deployment_readiness_{ts}.txt"
    with open(auto, "w") as f:
        f.write(report)
    if not args.output:
        print(f"\n[Saved to {auto}]")


if __name__ == "__main__":
    main()
