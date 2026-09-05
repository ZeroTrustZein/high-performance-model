"""FinTech market telemetry ingestion, buffering, and synthetic generation."""

from high_performance_model.telemetry.benchmark import run_telemetry_benchmark
from high_performance_model.telemetry.buffer import (
    BarAggregator,
    MarketDataBuffer,
    aggregate_bars_from_ticks,
)
from high_performance_model.telemetry.generator import SyntheticMarketFeed

__all__ = [
    "BarAggregator",
    "MarketDataBuffer",
    "SyntheticMarketFeed",
    "aggregate_bars_from_ticks",
    "run_telemetry_benchmark",
]
