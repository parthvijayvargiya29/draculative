#!/usr/bin/env python3
"""
Simple starter script for TradingView trading system
Bypasses import path issues
"""

import sys
import os
from pathlib import Path

# Add trading_system to path
project_root = Path(__file__).parent
trading_system_path = project_root / "trading_system"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(trading_system_path))

# Now run the main trading system
if __name__ == '__main__':
    os.chdir(str(project_root))
    
    # Import and run
    from trading_system.main_tradingview import main
    import asyncio
    
    asyncio.run(main())
