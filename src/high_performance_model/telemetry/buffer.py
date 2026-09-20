"""High-performance circular memory buffer and analytics engine for market telemetry."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from high_performance_model.indicators.series import (
    expected_shortfall,
    max_drawdown,
    realized_volatility,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from high_performance_model.types import (
    Bar,
    BarTimeframe,
    MarketDepthSnapshot,
    MarketState,
    MarketTelemetrySummary,
    MarketTick,
    OrderBook,
    Quote,
    RiskMetrics,
    Side,
    timeframe_to_seconds,
)


def _build_bar_from_bucket(
    symbol: str,
    ticks: List[MarketTick],
    bucket_ts: int,
    timeframe: BarTimeframe,
) -> Bar:
    """Helper to construct Bar model from bucket of ticks."""
    prices = [t.price for t in ticks]
    sizes = [t.size for t in ticks]

    open_p = round(prices[0], 4)
    high_p = round(max(prices), 4)
    low_p = round(min(prices), 4)
    close_p = round(prices[-1], 4)
    volume = round(sum(sizes), 4)

    # High / low safety bounds for rounding edge-cases
    high_p = max(high_p, open_p, close_p)
    low_p = min(low_p, open_p, close_p)

    total_notional = sum(t.price * t.size for t in ticks)
    vwap = round(total_notional / volume, 4) if volume > 0 else close_p

    ts = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)
    return Bar(
        symbol=symbol,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=volume,
        timeframe=timeframe,
        timestamp=ts,
        vwap=vwap,
        trades_count=len(ticks),
    )


class BarAggregator:
    """Streaming real-time candle bar builder for incoming market ticks."""

    def __init__(
        self,
        symbol: str,
        timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = BarTimeframe(timeframe) if isinstance(timeframe, str) else timeframe
        self.interval_seconds = timeframe_to_seconds(self.timeframe)
        self._current_bucket: Optional[int] = None
        self._bucket_ticks: List[MarketTick] = []

    def update(self, tick: MarketTick) -> Optional[Bar]:
        """Incorporate tick. Returns finalized Bar when an interval completes."""
        tick_ts = tick.timestamp.timestamp()
        bucket = int(tick_ts // self.interval_seconds) * self.interval_seconds

        completed_bar: Optional[Bar] = None
        if self._current_bucket is None:
            self._current_bucket = bucket

        if bucket != self._current_bucket:
            if self._bucket_ticks:
                completed_bar = _build_bar_from_bucket(
                    self.symbol, self._bucket_ticks, self._current_bucket, self.timeframe
                )
            self._current_bucket = bucket
            self._bucket_ticks = [tick]
        else:
            self._bucket_ticks.append(tick)

        return completed_bar

    def flush(self) -> Optional[Bar]:
        """Flush and return any unfinalized partial bar."""
        if not self._bucket_ticks or self._current_bucket is None:
            return None
        bar = _build_bar_from_bucket(
            self.symbol, self._bucket_ticks, self._current_bucket, self.timeframe
        )
        self._bucket_ticks.clear()
        self._current_bucket = None
        return bar


def aggregate_bars_from_ticks(
    ticks: List[MarketTick],
    timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1,
) -> List[Bar]:
    """Aggregate a sequence of market ticks into discrete OHLCV candle bars."""
    if not ticks:
        return []
    aggregator = BarAggregator(ticks[0].symbol, timeframe)
    bars: List[Bar] = []
    for tick in ticks:
        bar = aggregator.update(tick)
        if bar is not None:
            bars.append(bar)
    last_bar = aggregator.flush()
    if last_bar is not None:
        bars.append(last_bar)
    return bars


class MarketDataBuffer:
    """In-memory low-latency telemetry buffer with rolling window metric computations."""

    def __init__(self, capacity: int = 50_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._ticks: Dict[str, deque[MarketTick]] = {}
        self._order_books: Dict[str, OrderBook] = {}
        self._bars: Dict[str, deque[Bar]] = {}
        self._aggregators: Dict[Tuple[str, BarTimeframe], BarAggregator] = {}

    @property
    def symbols(self) -> List[str]:
        """List all active symbols currently stored in buffer."""
        return sorted(
            list(set(self._ticks.keys()) | set(self._bars.keys()) | set(self._order_books.keys()))
        )

    def push_tick(self, tick: MarketTick) -> Optional[Bar]:
        """Push a new market tick into symbol buffer and feed default 1m aggregator."""
        sym = tick.symbol
        if sym not in self._ticks:
            self._ticks[sym] = deque(maxlen=self.capacity)
        self._ticks[sym].append(tick)

        # Feed real-time 1m bar aggregator
        agg_key = (sym, BarTimeframe.MIN_1)
        if agg_key not in self._aggregators:
            self._aggregators[agg_key] = BarAggregator(sym, BarTimeframe.MIN_1)
        completed_bar = self._aggregators[agg_key].update(tick)
        if completed_bar:
            self.push_bar(completed_bar)
        return completed_bar

    def push_order_book(self, order_book: OrderBook) -> None:
        """Update current order book snapshot for symbol."""
        self._order_books[order_book.symbol] = order_book

    def push_bar(self, bar: Bar) -> None:
        """Push an aggregated OHLCV candle bar."""
        sym = bar.symbol
        if sym not in self._bars:
            self._bars[sym] = deque(maxlen=self.capacity)
        self._bars[sym].append(bar)

    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Return most recent tick for symbol."""
        ticks = self._ticks.get(symbol.upper())
        return ticks[-1] if ticks else None

    def get_ticks(self, symbol: str, limit: int = 100) -> List[MarketTick]:
        """Return the N most recent ticks for symbol."""
        ticks = self._ticks.get(symbol.upper())
        if not ticks:
            return []
        n = min(limit, len(ticks))
        return list(ticks)[-n:]

    def get_latest_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Return current order book snapshot for symbol."""
        return self._order_books.get(symbol.upper())

    def get_bars(self, symbol: str, limit: int = 100) -> List[Bar]:
        """Return the N most recent bars for symbol."""
        bars = self._bars.get(symbol.upper())
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
        vol = realized_volatility(prices, period=limit, annualize=True)
        return float(round(vol, 4))

    def compute_top_quote(self, symbol: str) -> Optional[Quote]:
        """Produce top-of-book Quote model combining latest order book and latest tick."""
        sym = symbol.upper()
        ob = self.get_latest_order_book(sym)
        tick = self.get_latest_tick(sym)

        if not ob or ob.best_bid is None or ob.best_ask is None:
            return None

        bid_size = ob.bids[0].quantity if ob.bids else 1.0
        ask_size = ob.asks[0].quantity if ob.asks else 1.0

        return Quote(
            symbol=sym,
            bid=ob.best_bid,
            ask=ob.best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=tick.price if tick else ob.mid_price,
            last_size=tick.size if tick else None,
            timestamp=tick.timestamp if tick else ob.timestamp,
        )

    def compute_market_depth_snapshot(
        self, symbol: str, depth: int = 10
    ) -> Optional[MarketDepthSnapshot]:
        """Generate MarketDepthSnapshot contract from current order book."""
        sym = symbol.upper()
        ob = self.get_latest_order_book(sym)
        if not ob:
            return None

        n = max(1, min(depth, 50))
        return MarketDepthSnapshot(
            symbol=sym,
            bids=ob.bids[:n],
            asks=ob.asks[:n],
            spread=ob.spread,
            mid_price=ob.mid_price,
            imbalance=ob.order_book_imbalance,
            timestamp=ob.timestamp,
            sequence=ob.sequence,
        )

    def compute_effective_spread(self, symbol: str) -> Optional[float]:
        """Compute effective spread: 2 * |Trade Price - Order Book Mid Price|."""
        sym = symbol.upper()
        tick = self.get_latest_tick(sym)
        ob = self.get_latest_order_book(sym)
        if not tick or not ob or ob.mid_price is None:
            return None
        return float(round(2.0 * abs(tick.price - ob.mid_price), 4))

    def compute_order_flow_imbalance(self, symbol: str, limit: int = 100) -> Dict[str, float]:
        """Calculate signed order flow volume imbalance across recent trade ticks."""
        ticks = self.get_ticks(symbol, limit=limit)
        if not ticks:
            return {"buy_volume": 0.0, "sell_volume": 0.0, "net_flow": 0.0, "imbalance_ratio": 0.0}

        buy_vol = sum(t.size for t in ticks if t.side == Side.BUY)
        sell_vol = sum(t.size for t in ticks if t.side == Side.SELL)
        total_vol = buy_vol + sell_vol
        net_flow = buy_vol - sell_vol
        ratio = (net_flow / total_vol) if total_vol > 0 else 0.0

        return {
            "buy_volume": round(buy_vol, 4),
            "sell_volume": round(sell_vol, 4),
            "net_flow": round(net_flow, 4),
            "imbalance_ratio": round(ratio, 4),
        }

    def compute_volume_profile(
        self, symbol: str, bins: int = 10, limit: int = 500
    ) -> Dict[str, Any]:
        """Generate volume-at-price profile distribution with Point of Control (POC)."""
        ticks = self.get_ticks(symbol, limit=limit)
        if not ticks or bins <= 0:
            return {"bins": [], "poc_price": None, "total_volume": 0.0}

        prices = np.array([t.price for t in ticks], dtype=np.float64)
        sizes = np.array([t.size for t in ticks], dtype=np.float64)

        min_p = float(np.min(prices))
        max_p = float(np.max(prices))

        if min_p == max_p:
            return {
                "bins": [{"price": min_p, "volume": round(float(np.sum(sizes)), 4)}],
                "poc_price": min_p,
                "total_volume": round(float(np.sum(sizes)), 4),
            }

        bin_edges = np.linspace(min_p, max_p, bins + 1)
        bin_volumes = np.zeros(bins, dtype=np.float64)

        # Distribute volume into price bins
        indices = np.digitize(prices, bin_edges) - 1
        indices = np.clip(indices, 0, bins - 1)
        for idx, size in zip(indices, sizes):
            bin_volumes[idx] += size

        poc_idx = int(np.argmax(bin_volumes))
        poc_price = round(float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0), 4)

        profile_bins = []
        for i in range(bins):
            mid_bin = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            profile_bins.append(
                {
                    "price_low": round(float(bin_edges[i]), 4),
                    "price_high": round(float(bin_edges[i + 1]), 4),
                    "price_mid": round(float(mid_bin), 4),
                    "volume": round(float(bin_volumes[i]), 4),
                }
            )

        return {
            "bins": profile_bins,
            "poc_price": poc_price,
            "total_volume": round(float(np.sum(sizes)), 4),
        }

    def estimate_market_impact(self, symbol: str, size: float, side: Side) -> Dict[str, Any]:
        """Simulate order book execution to estimate slippage and market impact."""
        sym = symbol.upper()
        ob = self.get_latest_order_book(sym)
        if not ob or ob.mid_price is None or size <= 0:
            return {
                "symbol": sym,
                "requested_size": size,
                "executed_size": 0.0,
                "avg_execution_price": None,
                "slippage_bps": None,
                "unfilled_size": size,
            }

        mid = ob.mid_price
        levels = ob.asks if side == Side.BUY else ob.bids
        if not levels:
            return {
                "symbol": sym,
                "requested_size": size,
                "executed_size": 0.0,
                "avg_execution_price": None,
                "slippage_bps": None,
                "unfilled_size": size,
            }

        remaining = size
        total_cost = 0.0
        executed_qty = 0.0

        for lvl in levels:
            fill_qty = min(remaining, lvl.quantity)
            total_cost += fill_qty * lvl.price
            executed_qty += fill_qty
            remaining -= fill_qty
            if remaining <= 1e-9:
                break

        if executed_qty <= 0:
            return {
                "symbol": sym,
                "requested_size": size,
                "executed_size": 0.0,
                "avg_execution_price": None,
                "slippage_bps": None,
                "unfilled_size": size,
            }

        avg_price = total_cost / executed_qty
        # Slippage in bps compared to mid-price
        slippage_bps = abs(avg_price - mid) / mid * 10_000.0

        return {
            "symbol": sym,
            "requested_size": size,
            "executed_size": round(executed_qty, 4),
            "avg_execution_price": round(avg_price, 4),
            "slippage_bps": round(slippage_bps, 4),
            "unfilled_size": round(remaining, 4),
        }

    def compute_risk_metrics(self, symbol: str, limit: int = 150) -> Optional[RiskMetrics]:
        """Compute full quantitative risk contract for symbol."""
        sym = symbol.upper()
        prices = self.get_price_series(sym, limit=limit)
        if len(prices) < 10:
            return None

        returns = np.diff(np.log(prices))
        vol = realized_volatility(prices, period=limit, annualize=True)
        sr = sharpe_ratio(returns, risk_free_rate=0.0, annualize=True)
        sort_r = sortino_ratio(returns, risk_free_rate=0.0, annualize=True)
        mdd, _ = max_drawdown(prices)
        var_95 = value_at_risk(returns, confidence_level=0.95, method="historical")
        cvar_95 = expected_shortfall(returns, confidence_level=0.95)

        return RiskMetrics(
            symbol=sym,
            realized_volatility=round(vol, 4),
            sharpe_ratio=round(sr, 2),
            sortino_ratio=round(sort_r, 2),
            max_drawdown=round(mdd, 4),
            value_at_risk_95=round(var_95, 4),
            expected_shortfall_95=round(cvar_95, 4),
            sample_size=len(prices),
        )

    def compute_telemetry_summary(
        self, symbol: str, limit: int = 200
    ) -> Optional[MarketTelemetrySummary]:
        """Assemble top-level market telemetry summary."""
        sym = symbol.upper()
        ticks = self.get_ticks(sym, limit=limit)
        if not ticks:
            return None

        latest_tick = ticks[-1]
        ob = self.get_latest_order_book(sym)
        prices = [t.price for t in ticks]
        sizes = [t.size for t in ticks]

        high_p = max(prices)
        low_p = min(prices)
        tot_vol = sum(sizes)
        vwap = self.compute_vwap(sym, limit=limit)

        first_p = prices[0]
        last_p = latest_tick.price
        change_pct = round(((last_p - first_p) / first_p) * 100.0, 4) if first_p > 0 else 0.0

        return MarketTelemetrySummary(
            symbol=sym,
            last_price=last_p,
            last_size=latest_tick.size,
            last_side=latest_tick.side,
            volume_24h=round(tot_vol, 2),
            vwap=vwap,
            high_24h=high_p,
            low_24h=low_p,
            spread=ob.spread if ob else None,
            change_24h_pct=change_pct,
            market_state=MarketState.OPEN,
            timestamp=latest_tick.timestamp,
        )

    def aggregate_and_store_bars(
        self, symbol: str, timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1
    ) -> List[Bar]:
        """Aggregate all buffered ticks for symbol and store produced bars."""
        sym = symbol.upper()
        ticks = self.get_ticks(sym, limit=self.capacity)
        bars = aggregate_bars_from_ticks(ticks, timeframe=timeframe)
        for b in bars:
            self.push_bar(b)
        return bars

    def clear(self) -> None:
        """Reset all internal telemetry queues and aggregators."""
        self._ticks.clear()
        self._order_books.clear()
        self._bars.clear()
        self._aggregators.clear()
