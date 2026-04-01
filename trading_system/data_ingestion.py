"""Layer 1: Data Ingestion - Real-time market data with validation and buffering."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import json

import numpy as np
import pandas as pd


@dataclass
class OHLCV:
    """Open-High-Low-Close-Volume candle."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class DataValidator:
    """Validates OHLC data for logical consistency and gaps."""
    
    @staticmethod
    def is_valid(candle: OHLCV) -> tuple[bool, str]:
        """Check if candle is valid. Returns (is_valid, reason)."""
        # Basic bounds
        if candle.close <= 0 or candle.open <= 0:
            return False, "Negative price"
        if candle.high < candle.low:
            return False, "High < Low"
        if candle.high < max(candle.open, candle.close):
            return False, "High below OHLC"
        if candle.low > min(candle.open, candle.close):
            return False, "Low above OHLC"
        if candle.volume < 0:
            return False, "Negative volume"
        # Reasonable price movement check (no 50% move in one candle for large caps)
        pct_change = abs(candle.close - candle.open) / candle.open
        if pct_change > 0.15:  # More than 15% movement
            return False, f"Suspicious {pct_change*100:.1f}% move in single candle"
        return True, "Valid"
    
    @staticmethod
    def detect_gaps(prev_close: float, curr_open: float) -> bool:
        """Detect gap between candles (> 1% move)."""
        if prev_close <= 0:
            return False
        gap_pct = abs(curr_open - prev_close) / prev_close
        return gap_pct > 0.01


class DataBuffer:
    """Circular in-memory buffer for OHLCV candles. O(1) append/access."""
    
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.last_timestamp: Optional[float] = None
    
    def append(self, candle: OHLCV):
        """Add candle to buffer (evicts oldest if full)."""
        self.buffer.append(candle)
        self.last_timestamp = candle.timestamp
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to pandas DataFrame."""
        if not self.buffer:
            return pd.DataFrame()
        data = [c.to_dict() for c in self.buffer]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def latest(self, n: int = 1) -> List[OHLCV]:
        """Get last n candles."""
        return list(self.buffer)[-n:]
    
    def as_numpy(self, column: str = 'close') -> np.ndarray:
        """Get column as numpy array for fast indicator calculation."""
        df = self.to_dataframe()
        return df[column].values if column in df.columns else np.array([])
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Optional[OHLCV]:
        return list(self.buffer)[idx] if -len(self.buffer) <= idx < len(self.buffer) else None


class MarketDataFetcher:
    """Fetch real-time market data with WebSocket + REST fallback."""
    
    def __init__(self, symbol: str, interval_seconds: int = 900):
        """
        Args:
            symbol: Stock ticker (e.g., 'NVDA')
            interval_seconds: Candle interval (900 = 15 min)
        """
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.buffer = DataBuffer(capacity=500)
        self.validator = DataValidator()
        self.last_fetch_time = 0
    
    async def fetch_realtime_stub(self) -> Optional[OHLCV]:
        """Simulate real-time WebSocket data (stub for demo).
        
        In production, replace with actual WebSocket connection to:
        - Alpaca API (wss://data.alpaca.markets)
        - IB TWS
        - Polygon.io
        """
        # Simulated data for demo (replace with real WebSocket)
        await asyncio.sleep(0.1)
        
        now = time.time()
        # Simulate realistic price movement
        base_price = 132.50
        noise = np.random.randn() * 0.5
        close = base_price + noise
        
        candle = OHLCV(
            timestamp=now,
            open=close - 0.25,
            high=close + 0.50,
            low=close - 0.50,
            close=close,
            volume=int(np.random.uniform(1e6, 3e6))
        )
        
        is_valid, reason = self.validator.is_valid(candle)
        if not is_valid:
            print(f"  ⚠️ Invalid candle: {reason}")
            return None
        
        return candle
    
    def fetch_rest_fallback(self) -> Optional[OHLCV]:
        """Fetch via REST API (fallback if WebSocket unavailable).
        
        Production: Use yfinance, pandas_datareader, or broker API.
        """
        # Stub: return most recent candle from buffer
        if len(self.buffer) > 0:
            return self.buffer.latest(1)[0]
        return None
    
    async def stream_candles(self, max_iterations: int = None):
        """Stream candles continuously (until max_iterations or error)."""
        iteration = 0
        print(f"\n📡 Starting real-time data stream for {self.symbol}...")
        
        while max_iterations is None or iteration < max_iterations:
            # Try WebSocket first
            candle = await self.fetch_realtime_stub()
            
            # Fall back to REST if needed
            if candle is None:
                print(f"  [Iteration {iteration}] WebSocket failed, trying REST...")
                candle = self.fetch_rest_fallback()
            
            if candle:
                self.buffer.append(candle)
                gap_detected = False
                if self.buffer.last_timestamp and len(self.buffer) > 1:
                    prev = self.buffer.latest(2)[0]
                    gap_detected = self.validator.detect_gaps(prev.close, candle.open)
                
                gap_str = " [GAP DETECTED]" if gap_detected else ""
                print(f"  ✓ Candle {iteration}: {self.symbol} ${candle.close:.2f} vol={candle.volume:,.0f}{gap_str}")
            
            iteration += 1
            await asyncio.sleep(0.5)  # Simulate real-time interval
    
    def get_buffer(self) -> DataBuffer:
        """Get reference to internal buffer."""
        return self.buffer


class LiveMarketDataManager:
    """Manages multiple symbol streams and aggregates data."""
    
    def __init__(self, symbols: List[str], interval_seconds: int = 900):
        self.symbols = symbols
        self.fetchers = {sym: MarketDataFetcher(sym, interval_seconds) for sym in symbols}
    
    async def start_all_streams(self, max_iterations: int = None):
        """Start all symbol streams concurrently."""
        tasks = [self.fetchers[sym].stream_candles(max_iterations) for sym in self.symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_buffer(self, symbol: str) -> DataBuffer:
        """Get data buffer for a specific symbol."""
        return self.fetchers.get(symbol, DataBuffer()).get_buffer()


# Demo/test function
async def demo_data_ingestion():
    """Demo: Simulate 10 candles of real-time data."""
    fetcher = MarketDataFetcher('NVDA', interval_seconds=900)
    await fetcher.stream_candles(max_iterations=10)
    
    print(f"\n✅ Collected {len(fetcher.buffer)} candles")
    print(fetcher.buffer.to_dataframe().tail())
    print(f"Latest close: ${fetcher.buffer.latest(1)[0].close:.2f}")


if __name__ == '__main__':
    asyncio.run(demo_data_ingestion())
