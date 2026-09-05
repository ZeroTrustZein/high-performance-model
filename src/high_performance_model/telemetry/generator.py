"""Synthetic market telemetry and order book depth generator."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from high_performance_model.types import (
    AssetClass,
    Bar,
    BarTimeframe,
    MarketTick,
    OrderBook,
    OrderBookLevel,
    Side,
    timeframe_to_seconds,
)


class SyntheticMarketFeed:
    """Simulated low-latency market telemetry stream using jump-diffusion dynamics."""

    DEFAULT_BASE_PRICES: Dict[str, float] = {
        "AAPL": 220.50,
        "MSFT": 445.20,
        "NVDA": 128.80,
        "GOOGL": 178.40,
        "BTC/USD": 62500.0,
        "ETH/USD": 3400.0,
        "EUR/USD": 1.0850,
        "XAU/USD": 2500.0,
    }

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        base_prices: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.symbols = symbols or ["AAPL", "MSFT", "NVDA", "GOOGL", "BTC/USD"]
        self.prices: Dict[str, float] = (
            base_prices.copy() if base_prices is not None else self.DEFAULT_BASE_PRICES.copy()
        )
        self.seq = 0
        self._last_timestamp = datetime.now(timezone.utc)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_asset_class(self, symbol: str) -> AssetClass:
        """Infer asset class from ticker symbol."""
        sym = symbol.upper()
        if sym in {"XAU/USD", "WTI/USD", "BRENT"}:
            return AssetClass.COMMODITY
        if "/" in sym or sym.endswith("USDT") or "BTC" in sym or "ETH" in sym:
            if "USD" in sym and any(c in sym for c in ["EUR", "GBP", "JPY", "CHF", "AUD"]):
                return AssetClass.FX
            return AssetClass.CRYPTO
        return AssetClass.EQUITY

    def _get_volatility_and_jump_params(self, symbol: str) -> tuple[float, float, float]:
        """Return (diffusion_vol, jump_prob, jump_vol) for symbol."""
        asset_class = self.get_asset_class(symbol)
        if asset_class == AssetClass.CRYPTO:
            return 0.0025, 0.05, 0.015
        elif asset_class == AssetClass.FX:
            return 0.0003, 0.01, 0.002
        elif asset_class == AssetClass.COMMODITY:
            return 0.0012, 0.03, 0.008
        else:  # Equity
            return 0.0008, 0.02, 0.006

    def generate_tick(
        self,
        symbol: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> MarketTick:
        """Produce a single synthetic tick with realistic jump-diffusion dynamics."""
        sym = (symbol or random.choice(self.symbols)).upper()
        curr_price = self.prices.get(sym, 100.0)

        volatility, jump_prob, jump_vol = self._get_volatility_and_jump_params(sym)

        # 1. Brownian motion diffusion
        delta_pct = float(np.random.normal(0.0, volatility))

        # 2. Compound Poisson jump
        if random.random() < jump_prob:
            jump = float(np.random.normal(0.0, jump_vol))
            delta_pct += jump

        new_price = max(0.0001, curr_price * (1.0 + delta_pct))
        precision = 4 if curr_price >= 1.0 else 6
        new_price = round(new_price, precision)
        self.prices[sym] = new_price

        side = Side.BUY if delta_pct >= 0 else Side.SELL
        size = float(round(abs(np.random.exponential(scale=15.0)) + 1.0, 2))
        self.seq += 1

        if timestamp is not None:
            ts = timestamp
        else:
            self._last_timestamp += timedelta(milliseconds=random.randint(10, 250))
            ts = self._last_timestamp

        return MarketTick(
            symbol=sym,
            price=self.prices[sym],
            size=size,
            side=side,
            sequence=self.seq,
            timestamp=ts,
        )

    def generate_order_book(
        self,
        symbol: str,
        depth: int = 10,
        imbalance_bias: float = 0.0,
    ) -> OrderBook:
        """Generate realistic L2 bid-ask order book around current mid-price."""
        sym = symbol.upper()
        mid = self.prices.get(sym, 100.0)
        asset_class = self.get_asset_class(sym)

        if asset_class == AssetClass.FX:
            spread_half = max(0.00005, mid * 0.00005)
        elif asset_class == AssetClass.CRYPTO:
            spread_half = max(0.5, mid * 0.0004)
        else:
            spread_half = max(0.01, mid * 0.0002)

        precision = 4 if mid >= 1.0 else 6
        bids: List[OrderBookLevel] = []
        asks: List[OrderBookLevel] = []

        for i in range(depth):
            step = (i + 1) * spread_half
            bid_price = round(max(0.0001, mid - step), precision)
            ask_price = round(mid + step, precision)

            # Volume distribution with imbalance bias
            bid_scale = 50.0 * (1.0 + imbalance_bias)
            ask_scale = 50.0 * (1.0 - imbalance_bias)
            bid_qty = round(
                float(np.random.gamma(shape=2.0, scale=max(5.0, bid_scale)) * (1.0 + i * 0.2)), 2
            )
            ask_qty = round(
                float(np.random.gamma(shape=2.0, scale=max(5.0, ask_scale)) * (1.0 + i * 0.2)), 2
            )

            bids.append(
                OrderBookLevel(
                    price=bid_price, quantity=bid_qty, orders_count=random.randint(1, 10)
                )
            )
            asks.append(
                OrderBookLevel(
                    price=ask_price, quantity=ask_qty, orders_count=random.randint(1, 10)
                )
            )

        return OrderBook(
            symbol=sym,
            bids=bids,
            asks=asks,
            sequence=self.seq,
            timestamp=self._last_timestamp,
        )

    def generate_history(
        self,
        symbol: str,
        n_points: int = 100,
        interval_ms: int = 500,
    ) -> List[MarketTick]:
        """Generate batch historical ticks for warming buffers and testing."""
        sym = symbol.upper()
        start_ts = datetime.now(timezone.utc) - timedelta(milliseconds=n_points * interval_ms)
        ticks = []
        for i in range(n_points):
            t_ts = start_ts + timedelta(milliseconds=i * interval_ms)
            ticks.append(self.generate_tick(symbol=sym, timestamp=t_ts))
        return ticks

    def generate_bars(
        self,
        symbol: str,
        n_bars: int = 50,
        timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1,
    ) -> List[Bar]:
        """Generate synthetic OHLCV candle bars with realistic intraday drift."""
        sym = symbol.upper()
        tf = BarTimeframe(timeframe) if isinstance(timeframe, str) else timeframe
        sec_step = timeframe_to_seconds(tf)

        bars: List[Bar] = []
        current = self.prices.get(sym, 100.0)
        precision = 4 if current >= 1.0 else 6
        start_ts = datetime.now(timezone.utc) - timedelta(seconds=n_bars * sec_step)

        for i in range(n_bars):
            bar_ts = start_ts + timedelta(seconds=i * sec_step)
            open_p = current
            # High / Low variations
            h_pct = abs(float(np.random.normal(0, 0.005)))
            l_pct = abs(float(np.random.normal(0, 0.005)))
            high_p = open_p * (1.0 + h_pct)
            low_p = open_p * (1.0 - l_pct)
            close_p = float(np.random.uniform(low_p, high_p))

            open_p = round(open_p, precision)
            high_p = round(max(high_p, open_p, close_p), precision)
            low_p = round(min(low_p, open_p, close_p), precision)
            close_p = round(close_p, precision)

            volume = float(round(np.random.exponential(1000.0) + 100.0, 2))
            vwap = round((open_p + high_p + low_p + close_p) / 4.0, precision)

            bars.append(
                Bar(
                    symbol=sym,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=volume,
                    timeframe=tf,
                    timestamp=bar_ts,
                    vwap=vwap,
                    trades_count=random.randint(20, 200),
                )
            )
            current = close_p

        self.prices[sym] = current
        return bars

    def stream(self, count: Optional[int] = None) -> Iterator[MarketTick]:
        """Yield stream of synthetic ticks."""
        generated = 0
        while count is None or generated < count:
            yield self.generate_tick()
            generated += 1
