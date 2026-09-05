"""High-throughput performance benchmarking engine for telemetry ingestion and analytics."""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from high_performance_model.indicators.series import (
    average_true_range,
    bollinger_bands,
    exponential_moving_average,
    macd,
    momentum,
    rate_of_change,
    relative_strength_index,
    simple_moving_average,
    stochastic_oscillator,
    volume_weighted_average_price,
)
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import BenchmarkRunResult


def run_telemetry_benchmark(
    ticks_count: int = 10_000,
    symbol: str = "NVDA",
    capacity: int = 100_000,
    run_indicators: bool = True,
    sample_latency_interval: int = 10,
) -> BenchmarkRunResult:
    """Execute high-throughput telemetry ingestion and indicator calculation benchmark."""
    feed = SyntheticMarketFeed(seed=42)
    buffer = MarketDataBuffer(capacity=max(capacity, ticks_count + 1000))

    latencies_ns: List[int] = []

    # Pre-generate ticks to isolate ingestion latency if desired or generate on fly
    # We measure push_tick latency
    t_start = time.perf_counter()

    for i in range(ticks_count):
        tick = feed.generate_tick(symbol=symbol)
        if i % sample_latency_interval == 0:
            t_tick_0 = time.perf_counter_ns()
            buffer.push_tick(tick)
            t_tick_1 = time.perf_counter_ns()
            latencies_ns.append(t_tick_1 - t_tick_0)
        else:
            buffer.push_tick(tick)

    t_end = time.perf_counter()
    duration = max(t_end - t_start, 1e-6)
    ticks_per_sec = float(round(ticks_count / duration, 2))

    latencies_us = np.array(latencies_ns, dtype=np.float64) / 1000.0
    p50 = float(round(float(np.percentile(latencies_us, 50)), 3)) if len(latencies_us) else 0.0
    p95 = float(round(float(np.percentile(latencies_us, 95)), 3)) if len(latencies_us) else 0.0
    p99 = float(round(float(np.percentile(latencies_us, 99)), 3)) if len(latencies_us) else 0.0

    buffered_count = len(buffer._ticks.get(symbol.upper(), []))
    utilization_pct = min(100.0, float(round((buffered_count / buffer.capacity) * 100.0, 2)))

    indicator_timings: Dict[str, float] = {}
    if run_indicators:
        prices = buffer.get_price_series(symbol, limit=min(ticks_count, 10_000))
        volumes = buffer.get_volume_series(symbol, limit=min(ticks_count, 10_000))
        highs = prices * 1.01
        lows = prices * 0.99

        t0 = time.perf_counter()
        _ = simple_moving_average(prices, 20)
        indicator_timings["sma_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = exponential_moving_average(prices, 20)
        indicator_timings["ema_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = relative_strength_index(prices, 14)
        indicator_timings["rsi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = macd(prices)
        indicator_timings["macd_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = bollinger_bands(prices, 20)
        indicator_timings["bollinger_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = average_true_range(highs, lows, prices, 14)
        indicator_timings["atr_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = volume_weighted_average_price(prices, volumes)
        indicator_timings["vwap_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = momentum(prices, 10)
        indicator_timings["momentum_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = rate_of_change(prices, 10)
        indicator_timings["roc_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        _ = stochastic_oscillator(highs, lows, prices, 14, 3, 3)
        indicator_timings["stochastic_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    return BenchmarkRunResult(
        total_ticks=ticks_count,
        duration_seconds=round(duration, 4),
        ticks_per_second=ticks_per_sec,
        latency_p50_us=p50,
        latency_p95_us=p95,
        latency_p99_us=p99,
        buffer_utilization_pct=utilization_pct,
        metadata={
            "symbol": symbol,
            "sample_latency_count": len(latencies_ns),
            "indicators": indicator_timings,
        },
    )
