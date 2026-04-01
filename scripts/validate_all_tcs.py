#!/usr/bin/env python3
"""
Complete System Validation Runner
==================================

Validates ALL 15 TCs end-to-end.

FLOW:
1. Run TC-01 through TC-15 validations
2. Check all 6 gates per TC
3. Generate master report with ACTIVE/PENDING status
4. Update TC registry YAML

USAGE:
    python scripts/validate_all_tcs.py --quick  # 5 symbols per TC
    python scripts/validate_all_tcs.py --full   # Full symbol universe
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from technicals.tc_01_supertrend import TC01_Supertrend
from technicals.tc_02_bb_rsi2 import TC02_BollingerRSI2
from technicals.tc_03_breakout import TC03_Breakout
from technicals.tc_04_fvg import TC04_FairValueGap
from technicals.tc_05_order_block import TC05_OrderBlock
from technicals.tc_06_vwap import TC06_VWAPReversion
from technicals.tc_07_adx_gate import TC07_ADXGate
from technicals.tc_08_golden_cross import TC08_GoldenCross
from technicals.tc_09_volume_climax import TC09_VolumeClimax
from technicals.tc_10_liquidity_sweep import TC10_LiquiditySweep
from technicals.tc_11_choch import TC11_ChoCH
from technicals.tc_12_ppo_pvo import TC12_PPOPVO
from technicals.tc_13_stoch_rsi import TC13_StochRSI
from technicals.tc_14_fibonacci_ote import TC14_FibonacciOTE
from technicals.tc_15_pivot_breakout import TC15_PivotBreakout


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate all 15 TCs")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='quick=5 symbols, full=all symbols')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPLETE SYSTEM VALIDATION — ALL 15 TCs")
    print("="*80)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*80 + "\n")
    
    validation_config = {
        'start_date': '2024-01-01',
        'end_date': '2026-01-01',
        'initial_capital': 100000,
    }
    
    # TC instances
    tcs = [
        TC01_Supertrend(),
        TC02_BollingerRSI2(),
        TC03_Breakout(),
        TC04_FairValueGap(),
        TC05_OrderBlock(),
        TC06_VWAPReversion(),
        TC07_ADXGate(),
        TC08_GoldenCross(),
        TC09_VolumeClimax(),
        TC10_LiquiditySweep(),
        TC11_ChoCH(),
        TC12_PPOPVO(),
        TC13_StochRSI(),
        TC14_FibonacciOTE(),
        TC15_PivotBreakout()
    ]
    
    results = []
    approved_count = 0
    
    for i, tc in enumerate(tcs, 1):
        print(f"\n[{i}/15] {tc.TC_ID}: {tc.TC_NAME}")
        print("-" * 60)
        
        # Select symbols based on mode
        if args.mode == 'quick':
            symbols = tc.symbols[:5] if hasattr(tc, 'symbols') else ['SPY', 'QQQ', 'AAPL']
        else:
            symbols = tc.symbols if hasattr(tc, 'symbols') else ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT']
        
        print(f"Symbols: {', '.join(symbols)}\n")
        
        try:
            result = tc.validate({**validation_config, 'symbols': symbols})
            results.append((tc.TC_ID, tc.TC_NAME, result))
            
            if result.approved:
                approved_count += 1
                print(f"✅ APPROVED")
            else:
                print(f"❌ REJECTED")
        
        except Exception as e:
            print(f"⚠️  VALIDATION ERROR: {e}")
            results.append((tc.TC_ID, tc.TC_NAME, None))
    
    # Summary Report
    print("\n\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")
    
    for tc_id, tc_name, result in results:
        if result is None:
            status = "⚠️  ERROR"
        elif result.approved:
            status = "✅ ACTIVE"
        else:
            status = "❌ PENDING"
        
        print(f"{tc_id:8s} {tc_name:40s} {status}")
    
    print("\n" + "="*80)
    print(f"FINAL: {approved_count}/15 TCs APPROVED")
    print("="*80 + "\n")
    
    # Update TC registry
    registry_path = project_root / "config" / "tc_registry.yaml"
    update_registry(registry_path, results)
    
    return 0 if approved_count == 15 else 1


def update_registry(registry_path: Path, results):
    """Update TC registry with validation results"""
    registry = {}
    
    for tc_id, tc_name, result in results:
        if result is not None:
            registry[tc_id.lower().replace('-', '_')] = {
                'name': tc_name,
                'status': 'ACTIVE' if result.approved else 'PENDING',
                'validation_date': result.validation_date.strftime('%Y-%m-%d'),
                'metrics': {
                    'profit_factor': result.metrics.get('profit_factor', 0),
                    'win_rate': result.metrics.get('win_rate', 0),
                    'max_drawdown': result.metrics.get('max_drawdown_pct', 0)
                }
            }
    
    with open(registry_path, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated TC registry: {registry_path}")


if __name__ == "__main__":
    sys.exit(main())
