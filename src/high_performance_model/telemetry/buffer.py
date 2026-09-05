"""High-performance circular memory buffer for market telemetry."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from high_performance_model.types import Bar, MarketTick, OrderBook


class MarketDataBuffer:
    """In-memory low-latency telemetry buffer with rolling window metric computations."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._ticks: Dict[str, deque[MarketTick]] = {}
        self._order_books: Dict[str, OrderBook] = {}
        self._bars: Dict[str, deque[Bar]] = {}

    def push_tick(self, tick: MarketTick) -> None:
        """Push a new market tick into symbol buffer."""
        if tick.symbol not in self._ticks:
            self._ticks[tick.symbol] = deque(maxlen=self.capacity)
        self._ticks[tick.symbol].append(tick)

    def push_order_book(self, order_book: OrderBook) -> None:
        """Update current order book snapshot for symbol."""
        self._order_books[order_book.symbol] = order_book

    def push_bar(self, bar: Bar) -> None:
        """Push an aggregated OHLCV candle bar."""
        if bar.symbol not in self._bars:
            self._bars[bar.symbol] = deque(maxlen=self.capacity)
        self._bars[bar.symbol].append(bar)

    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Return most recent tick for symbol."""
        ticks = self._ticks.get(symbol)
        return ticks[-1] if ticks else None

    def get_ticks(self, symbol: str, limit: int = 100) -> List[MarketTick]:
        """Return the N most recent ticks for symbol."""
        ticks = self._ticks.get(symbol)
        if not ticks:
            return []
        n = min(limit, len(ticks))
        return list(ticks)[-n:]

    def get_latest_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Return current order book snapshot for symbol."""
        return self._order_books.get(symbol)

    def get_bars(self, symbol: str, limit: int = 100) -> List[Bar]:
        """Return the N most recent bars for symbol."""
        bars = self._bars.get(symbol)
        if not bars:
            return []
        n = min(limit, len(bars))
        return list(bars)[-n:]

    def get_price_series(self, symbol: str, limit: int = 200) -> np.ndarray:
        """Return NumPy 1D array of recent tick prices."""
        ticks = self.get_ticks(symbol, limit=limit)
        if not ticks:
            return np.empty(0, dtype=np.float64)
        return np.array([t.price for t in ticks], dtype=np.float64)

    def get_volume_series(self, symbol: str, limit: int = 200) -> np.ndarray:
        """Return NumPy 1D array of recent tick trade volumes."""
        ticks = self.get_ticks(symbol, limit=limit)
        if not ticks:
            return np.empty(0, dtype=np.float64)
        return np.array([t.size for t in ticks], dtype=np.float64)

    def compute_vwap(self, symbol: str, limit: int = 200) -> Optional[float]:
        """Compute Volume-Weighted Average Price (VWAP) over recent ticks."""
        ticks = self.get_ticks(symbol, limit=limit)
        if not ticks:
            return None
        prices = np.array([t.price for t in ticks], dtype=np.float64)
        sizes = np.array([t.size for t in ticks], dtype=np.float64)
        total_volume = np.sum(sizes)
        if total_volume <= 0:
            return float(prices[-1])
        return float(round(np.sum(prices * sizes) / total_volume, 4))

    def compute_realized_volatility(self, symbol: str, limit: int = 100) -> Optional[float]:
        """Compute annualized realized volatility from log returns."""
        prices = self.get_price_series(symbol, limit=limit)
        if len(prices) < 5:
            return None
        log_returns = np.diff(np.log(prices))
        vol = float(np.std(log_returns))
        # Annualize assuming 252 trading days and standard day granularity
        annualized = vol * np.sqrt(252.0 * 24.0 * 60.0)
        return float(round(annualized, 4))

    def clear(self) -> None:
        """Reset all internal telemetry queues."""
        self._ticks.clear()
        self._order_books.clear()
        self._bars.clear()
