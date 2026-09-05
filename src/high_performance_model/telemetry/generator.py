"""Synthetic market telemetry and order book depth generator."""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

import numpy as np

from high_performance_model.types import (
    Bar,
    MarketTick,
    OrderBook,
    OrderBookLevel,
    Side,
)


class SyntheticMarketFeed:
    """Simulated low-latency market telemetry stream using jump-diffusion dynamics."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        base_prices: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.symbols = symbols or ["AAPL", "MSFT", "NVDA", "GOOGL", "BTC/USD"]
        self.prices: Dict[str, float] = base_prices or {
            "AAPL": 220.50,
            "MSFT": 445.20,
            "NVDA": 128.80,
            "GOOGL": 178.40,
            "BTC/USD": 62500.0,
        }
        self.seq = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_tick(self, symbol: Optional[str] = None) -> MarketTick:
        """Produce a single synthetic tick with realistic stochastic drift."""
        sym = symbol or random.choice(self.symbols)
        curr_price = self.prices.get(sym, 100.0)

        # Micro-drift + volatility jump
        volatility = 0.0008 if not sym.startswith("BTC") else 0.0025
        delta_pct = np.random.normal(0.0, volatility)
        new_price = max(0.01, curr_price * (1.0 + delta_pct))
        self.prices[sym] = round(new_price, 4)

        side = Side.BUY if delta_pct >= 0 else Side.SELL
        size = float(round(abs(np.random.exponential(scale=15.0)) + 1.0, 2))
        self.seq += 1

        return MarketTick(
            symbol=sym,
            price=self.prices[sym],
            size=size,
            side=side,
            sequence=self.seq,
        )

    def generate_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """Generate realistic L2 bid-ask order book around current mid-price."""
        mid = self.prices.get(symbol, 100.0)
        spread_half = max(0.01, mid * 0.0002)

        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []

        for i in range(depth):
            step = (i + 1) * spread_half
            bid_price = round(max(0.01, mid - step), 4)
            ask_price = round(mid + step, 4)

            bid_qty = round(float(np.random.gamma(shape=2.0, scale=50.0) * (1.0 + i * 0.2)), 2)
            ask_qty = round(float(np.random.gamma(shape=2.0, scale=50.0) * (1.0 + i * 0.2)), 2)

            bids.append(OrderBookLevel(price=bid_price, quantity=bid_qty, orders_count=random.randint(1, 8)))
            asks.append(OrderBookLevel(price=ask_price, quantity=ask_qty, orders_count=random.randint(1, 8)))

        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    def generate_history(self, symbol: str, n_points: int = 100) -> List[MarketTick]:
        """Generate batch historical ticks for warming buffers and testing."""
        return [self.generate_tick(symbol=symbol) for _ in range(n_points)]

    def generate_bars(self, symbol: str, n_bars: int = 50) -> List[Bar]:
        """Generate synthetic OHLCV candle bars."""
        bars: List[Bar] = []
        current = self.prices.get(symbol, 100.0)

        for _ in range(n_bars):
            open_p = current
            high_p = open_p * (1.0 + abs(float(np.random.normal(0, 0.005))))
            low_p = open_p * (1.0 - abs(float(np.random.normal(0, 0.005))))
            close_p = float(np.random.uniform(low_p, high_p))
            volume = float(round(np.random.exponential(1000.0) + 100.0, 2))
            vwap = round((open_p + high_p + low_p + close_p) / 4.0, 4)

            bars.append(
                Bar(
                    symbol=symbol,
                    open=round(open_p, 4),
                    high=round(high_p, 4),
                    low=round(low_p, 4),
                    close=round(close_p, 4),
                    volume=volume,
                    vwap=vwap,
                    trades_count=random.randint(20, 200),
                )
            )
            current = close_p

        return bars

    def stream(self, count: Optional[int] = None) -> Iterator[MarketTick]:
        """Yield stream of synthetic ticks."""
        generated = 0
        while count is None or generated < count:
            yield self.generate_tick()
            generated += 1
