#!/usr/bin/env python3
"""
Task 1 Validation Runner
=========================

Runs Alpaca validation for TC-01, TC-02, and TC-03 in sequence.

Usage:
    python scripts/validate_task1_tcs.py
    
Output:
    Summary report with all gate results and ACTIVE/PENDING status per TC.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from technicals.tc_01_supertrend import TC01_Supertrend
from technicals.tc_02_bb_rsi2 import TC02_BollingerRSI2
from technicals.tc_03_breakout import TC03_Breakout


def main():
    print("\n" + "="*80)
    print("TASK 1 VALIDATION — TC-01, TC-02, TC-03")
    print("="*80)
    print("\nRefactored from backtest_v3 Combo A/B/C")
    print("Running 2-year Alpaca simulations (2024-01-01 → 2026-01-01)")
    print("\n" + "-"*80 + "\n")
    
    validation_config = {
        'start_date': '2024-01-01',
        'end_date': '2026-01-01',
        'initial_capital': 100000,
    }
    
    results = []
    
    # ── TC-01: SuperTrend Pullback ───────────────────────────────────────────
    print("\n[1/3] TC-01: SuperTrend Pullback")
    print("     Validating on: SPY, QQQ, NVDA, AAPL, MSFT\n")
    
    tc01 = TC01_Supertrend()
    result01 = tc01.validate({
        **validation_config,
        'symbols': ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT']
    })
    results.append(('TC-01', result01))
    
    # ── TC-02: Bollinger RSI2 ────────────────────────────────────────────────
    print("\n[2/3] TC-02: Bollinger RSI2 Mean Reversion")
    print("     Validating on: GLD, WMT, USMV, COST, XOM\n")
    
    tc02 = TC02_BollingerRSI2()
    result02 = tc02.validate({
        **validation_config,
        'symbols': ['GLD', 'WMT', 'USMV', 'COST', 'XOM']
    })
    results.append(('TC-02', result02))
    
    # ── TC-03: 20-Bar Breakout ───────────────────────────────────────────────
    print("\n[3/3] TC-03: 20-Bar Breakout + ADX")
    print("     Validating on: SPY, QQQ, NVDA, AAPL, MSFT\n")
    
    tc03 = TC03_Breakout()
    result03 = tc03.validate({
        **validation_config,
        'symbols': ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT']
    })
    results.append(('TC-03', result03))
    
    # ── Summary Report ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TASK 1 VALIDATION SUMMARY")
    print("="*80 + "\n")
    
    approved_count = 0
    
    for tc_id, result in results:
        print(f"{tc_id}:")
        print(f"  Metrics:")
        print(f"    • Total Trades:      {result.metrics.get('total_trades', 0)}")
        print(f"    • Win Rate:          {result.metrics.get('win_rate', 0):.1%}")
        print(f"    • Profit Factor:     {result.metrics.get('profit_factor', 0):.2f}")
        print(f"    • Max Drawdown:      {result.metrics.get('max_drawdown_pct', 0):.1%}")
        
        print(f"\n  Gates:")
        for gate in result.gates_passed:
            print(f"    ✅ {gate.name}")
        for gate in result.gates_failed:
            print(f"    ❌ {gate.name} (threshold: {gate.threshold}, actual: {gate.actual:.3f})")
        
        status = "✅ APPROVED → ACTIVE" if result.approved else "❌ REJECTED → PENDING"
        print(f"\n  Status: {status}\n")
        print("-"*80 + "\n")
        
        if result.approved:
            approved_count += 1
    
    # Final summary
    print("="*80)
    print(f"FINAL: {approved_count}/3 TCs APPROVED")
    print("="*80)
    
    if approved_count == 3:
        print("\n🎉 TASK 1 COMPLETE — All 3 TCs validated and ACTIVE")
        return 0
    else:
        print(f"\n⚠️  TASK 1 PARTIAL — {3 - approved_count} TCs need tuning")
        return 1


if __name__ == "__main__":
    sys.exit(main())
