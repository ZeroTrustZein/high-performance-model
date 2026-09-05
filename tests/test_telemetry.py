"""Comprehensive unit tests for telemetry engine, buffer analytics, aggregation, and benchmarking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from high_performance_model.telemetry.benchmark import run_telemetry_benchmark
from high_performance_model.telemetry.buffer import (
    BarAggregator,
    MarketDataBuffer,
    aggregate_bars_from_ticks,
)
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import (
    AssetClass,
    BarTimeframe,
    BenchmarkRunResult,
    MarketDepthSnapshot,
    MarketState,
    MarketTelemetrySummary,
    MarketTick,
    Quote,
    RiskMetrics,
    Side,
)


class TestBarAggregation:
    """Test streaming and batch tick-to-candlestick aggregation algorithms."""

    def test_batch_aggregate_bars_from_ticks(self) -> None:
        base_time = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
        # Generate ticks across 3 minutes
        ticks = [
            MarketTick(
                symbol="AAPL",
                price=150.0,
                size=10.0,
                side=Side.BUY,
                timestamp=base_time + timedelta(seconds=10),
            ),
            MarketTick(
                symbol="AAPL",
                price=152.0,
                size=20.0,
                side=Side.BUY,
                timestamp=base_time + timedelta(seconds=25),
            ),
            MarketTick(
                symbol="AAPL",
                price=149.0,
                size=15.0,
                side=Side.SELL,
                timestamp=base_time + timedelta(seconds=40),
            ),
            MarketTick(
                symbol="AAPL",
                price=151.0,
                size=5.0,
                side=Side.BUY,
                timestamp=base_time + timedelta(seconds=55),
            ),
            # Minute 2
            MarketTick(
                symbol="AAPL",
                price=151.5,
                size=30.0,
                side=Side.BUY,
                timestamp=base_time + timedelta(seconds=70),
            ),
            MarketTick(
                symbol="AAPL",
                price=153.0,
                size=10.0,
                side=Side.BUY,
                timestamp=base_time + timedelta(seconds=90),
            ),
            MarketTick(
                symbol="AAPL",
                price=150.5,
                size=25.0,
                side=Side.SELL,
                timestamp=base_time + timedelta(seconds=110),
            ),
        ]

        bars = aggregate_bars_from_ticks(ticks, timeframe=BarTimeframe.MIN_1)
        assert len(bars) == 2

        # Bar 1 validation
        b1 = bars[0]
        assert b1.symbol == "AAPL"
        assert b1.open == 150.0
        assert b1.high == 152.0
        assert b1.low == 149.0
        assert b1.close == 151.0
        assert b1.volume == 50.0
        assert b1.trades_count == 4
        assert b1.high >= b1.open and b1.high >= b1.close
        assert b1.low <= b1.open and b1.low <= b1.close

        # Bar 2 validation
        b2 = bars[1]
        assert b2.open == 151.5
        assert b2.high == 153.0
        assert b2.low == 150.5
        assert b2.close == 150.5
        assert b2.volume == 65.0
        assert b2.trades_count == 3

    def test_streaming_bar_aggregator(self) -> None:
        agg = BarAggregator("NVDA", timeframe="1m")
        base_time = datetime(2026, 9, 6, 14, 0, 0, tzinfo=timezone.utc)

        t1 = MarketTick(
            symbol="NVDA",
            price=120.0,
            size=5.0,
            side=Side.BUY,
            timestamp=base_time + timedelta(seconds=5),
        )
        t2 = MarketTick(
            symbol="NVDA",
            price=125.0,
            size=10.0,
            side=Side.BUY,
            timestamp=base_time + timedelta(seconds=30),
        )
        # t3 crosses into the next minute
        t3 = MarketTick(
            symbol="NVDA",
            price=124.0,
            size=8.0,
            side=Side.SELL,
            timestamp=base_time + timedelta(seconds=65),
        )

        b1 = agg.update(t1)
        assert b1 is None
        b2 = agg.update(t2)
        assert b2 is None

        completed_bar = agg.update(t3)
        assert completed_bar is not None
        assert completed_bar.symbol == "NVDA"
        assert completed_bar.open == 120.0
        assert completed_bar.high == 125.0
        assert completed_bar.close == 125.0
        assert completed_bar.volume == 15.0

        # Flush partial minute
        flushed = agg.flush()
        assert flushed is not None
        assert flushed.open == 124.0
        assert flushed.close == 124.0
        assert flushed.volume == 8.0

        # Further flush is empty
        assert agg.flush() is None


class TestMarketDataBufferAnalytics:
    """Test MarketDataBuffer quantitative metrics, depth summaries, and execution impact."""

    def test_top_quote_and_depth_snapshot(self) -> None:
        feed = SyntheticMarketFeed(seed=123)
        buf = MarketDataBuffer(capacity=1000)

        # Before any data
        assert buf.compute_top_quote("AAPL") is None
        assert buf.compute_market_depth_snapshot("AAPL") is None

        ob = feed.generate_order_book("AAPL", depth=5)
        buf.push_order_book(ob)
        tick = feed.generate_tick("AAPL")
        buf.push_tick(tick)

        quote = buf.compute_top_quote("AAPL")
        assert isinstance(quote, Quote)
        assert quote.symbol == "AAPL"
        assert quote.bid == ob.best_bid
        assert quote.ask == ob.best_ask
        assert quote.last_price == tick.price

        snapshot = buf.compute_market_depth_snapshot("AAPL", depth=3)
        assert isinstance(snapshot, MarketDepthSnapshot)
        assert snapshot.symbol == "AAPL"
        assert len(snapshot.bids) == 3
        assert len(snapshot.asks) == 3

    def test_effective_spread_and_order_flow_imbalance(self) -> None:
        buf = MarketDataBuffer(capacity=1000)
        feed = SyntheticMarketFeed(seed=100)

        ob = feed.generate_order_book("MSFT", depth=5)
        buf.push_order_book(ob)
        assert ob.mid_price is not None

        # Push buy tick
        tick_buy = MarketTick(symbol="MSFT", price=ob.mid_price + 0.05, size=20.0, side=Side.BUY)
        buf.push_tick(tick_buy)
        eff_spread = buf.compute_effective_spread("MSFT")
        assert eff_spread is not None
        assert pytest.approx(eff_spread, 0.001) == 0.10

        # Push sell tick
        tick_sell = MarketTick(symbol="MSFT", price=ob.mid_price - 0.05, size=10.0, side=Side.SELL)
        buf.push_tick(tick_sell)

        flow = buf.compute_order_flow_imbalance("MSFT")
        assert flow["buy_volume"] == 20.0
        assert flow["sell_volume"] == 10.0
        assert flow["net_flow"] == 10.0
        assert pytest.approx(flow["imbalance_ratio"], 0.001) == (10.0 / 30.0)

    def test_volume_profile(self) -> None:
        buf = MarketDataBuffer(capacity=1000)
        ticks = [
            MarketTick(symbol="NVDA", price=100.0, size=10.0, side=Side.BUY),
            MarketTick(symbol="NVDA", price=100.5, size=50.0, side=Side.BUY),
            MarketTick(symbol="NVDA", price=101.0, size=20.0, side=Side.SELL),
            MarketTick(symbol="NVDA", price=102.0, size=5.0, side=Side.SELL),
        ]
        for t in ticks:
            buf.push_tick(t)

        profile = buf.compute_volume_profile("NVDA", bins=5)
        assert "bins" in profile
        assert len(profile["bins"]) == 5
        assert profile["total_volume"] == 85.0
        assert profile["poc_price"] is not None

    def test_estimate_market_impact(self) -> None:
        feed = SyntheticMarketFeed(seed=777)
        buf = MarketDataBuffer(capacity=1000)
        ob = feed.generate_order_book("GOOGL", depth=10)
        buf.push_order_book(ob)

        # Market buy order
        buy_impact = buf.estimate_market_impact("GOOGL", size=25.0, side=Side.BUY)
        assert buy_impact["executed_size"] == 25.0
        assert buy_impact["avg_execution_price"] >= ob.best_ask
        assert buy_impact["slippage_bps"] >= 0.0
        assert buy_impact["unfilled_size"] == 0.0

        # Market sell order
        sell_impact = buf.estimate_market_impact("GOOGL", size=30.0, side=Side.SELL)
        assert sell_impact["executed_size"] == 30.0
        assert sell_impact["avg_execution_price"] <= ob.best_bid
        assert sell_impact["unfilled_size"] == 0.0

    def test_compute_risk_metrics_and_telemetry_summary(self) -> None:
        feed = SyntheticMarketFeed(seed=42)
        buf = MarketDataBuffer(capacity=1000)
        ticks = feed.generate_history("BTC/USD", n_points=80)
        for t in ticks:
            buf.push_tick(t)
        ob = feed.generate_order_book("BTC/USD", depth=5)
        buf.push_order_book(ob)

        risk = buf.compute_risk_metrics("BTC/USD")
        assert isinstance(risk, RiskMetrics)
        assert risk.symbol == "BTC/USD"
        assert risk.realized_volatility is not None and risk.realized_volatility > 0.0
        assert risk.max_drawdown <= 0.0
        assert risk.value_at_risk_95 is not None and risk.value_at_risk_95 < 0.0

        summary = buf.compute_telemetry_summary("BTC/USD")
        assert isinstance(summary, MarketTelemetrySummary)
        assert summary.symbol == "BTC/USD"
        assert summary.volume_24h is not None and summary.volume_24h > 0.0
        assert summary.market_state == MarketState.OPEN


class TestSyntheticMarketFeed:
    """Test SyntheticMarketFeed jump diffusion, asset classes, and time series generation."""

    def test_asset_class_detection(self) -> None:
        feed = SyntheticMarketFeed()
        assert feed.get_asset_class("AAPL") == AssetClass.EQUITY
        assert feed.get_asset_class("BTC/USD") == AssetClass.CRYPTO
        assert feed.get_asset_class("ETH/USD") == AssetClass.CRYPTO
        assert feed.get_asset_class("EUR/USD") == AssetClass.FX
        assert feed.get_asset_class("XAU/USD") == AssetClass.COMMODITY

    def test_timeframe_bars_generation(self) -> None:
        feed = SyntheticMarketFeed(seed=55)
        bars_1m = feed.generate_bars("AAPL", n_bars=10, timeframe=BarTimeframe.MIN_1)
        assert len(bars_1m) == 10
        assert bars_1m[0].timeframe == BarTimeframe.MIN_1

        bars_1h = feed.generate_bars("AAPL", n_bars=5, timeframe=BarTimeframe.HOUR_1)
        assert len(bars_1h) == 5
        assert bars_1h[0].timeframe == BarTimeframe.HOUR_1


class TestBenchmarkingEngine:
    """Test high-throughput benchmark execution and metrics recording."""

    def test_run_telemetry_benchmark(self) -> None:
        res = run_telemetry_benchmark(ticks_count=2000, symbol="NVDA", run_indicators=True)
        assert isinstance(res, BenchmarkRunResult)
        assert res.total_ticks == 2000
        assert res.duration_seconds > 0.0
        assert res.ticks_per_second > 1000.0  # Should easily exceed 1,000 ticks/sec
        assert res.latency_p50_us >= 0.0
        assert res.latency_p95_us >= res.latency_p50_us
        assert res.latency_p99_us >= res.latency_p95_us
        assert 0.0 <= res.buffer_utilization_pct <= 100.0
        assert "indicators" in res.metadata
        assert "sma_ms" in res.metadata["indicators"]
