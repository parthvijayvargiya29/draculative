"""Quick Demo - Run all 5 layers of the trading system."""

import asyncio
from trading_system.data_ingestion import demo_data_ingestion
from trading_system.indicators import demo_indicators
from trading_system.trade_logic import demo_trade_logic
from trading_system.order_execution import demo_order_execution
from trading_system.monitoring import demo_monitoring


async def run_all_demos():
    """Run through all 5 layers sequentially."""
    
    print("\n" + "="*70)
    print("TRADING SYSTEM: 5-LAYER ARCHITECTURE DEMO")
    print("="*70)
    
    print("\n" + "─"*70)
    print("Layer 1: DATA INGESTION")
    print("─"*70)
    await demo_data_ingestion()
    
    print("\n" + "─"*70)
    print("Layer 2: INDICATOR CALCULATION")
    print("─"*70)
    demo_indicators()
    
    print("\n" + "─"*70)
    print("Layer 3: TRADE LOGIC")
    print("─"*70)
    demo_trade_logic()
    
    print("\n" + "─"*70)
    print("Layer 4: ORDER EXECUTION")
    print("─"*70)
    demo_order_execution()
    
    print("\n" + "─"*70)
    print("Layer 5: MONITORING & PERSISTENCE")
    print("─"*70)
    demo_monitoring()
    
    print("\n" + "="*70)
    print("✅ ALL DEMOS COMPLETED")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Install dependencies: pip install -r trading_system/requirements.txt")
    print("2. Backtest strategy: python trading_system/backtest.py")
    print("3. Paper trade: python trading_system/main.py --config config.yml --mode paper")
    print("4. Go live: python trading_system/main.py --config config.yml --mode live")
    print("\nFull documentation in: trading_system/README.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    asyncio.run(run_all_demos())
