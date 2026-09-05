"""FinTech market telemetry ingestion, buffering, and synthetic generation."""

from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed

__all__ = ["MarketDataBuffer", "SyntheticMarketFeed"]
