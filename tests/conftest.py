"""Pytest fixtures for FinTech MCP server tests."""

import numpy as np
import pytest

from high_performance_model.protocol.server import MCPServer
from high_performance_model.server.app import create_fintech_mcp_server
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed


@pytest.fixture
def synthetic_feed() -> SyntheticMarketFeed:
    """Synthetic market data feed fixture."""
    return SyntheticMarketFeed(seed=123)


@pytest.fixture
def market_buffer(synthetic_feed: SyntheticMarketFeed) -> MarketDataBuffer:
    """Market data buffer pre-populated with ticks."""
    buf = MarketDataBuffer(capacity=5000)
    for tick in synthetic_feed.generate_history("AAPL", n_points=100):
        buf.push_tick(tick)
    ob = synthetic_feed.generate_order_book("AAPL", depth=5)
    buf.push_order_book(ob)
    return buf


@pytest.fixture
def sample_prices() -> np.ndarray:
    """Sample price series array."""
    np.random.seed(42)
    base = 100.0
    returns = np.random.normal(0.0005, 0.01, size=150)
    return base * np.cumprod(1.0 + returns)


@pytest.fixture
def mcp_server() -> MCPServer:
    """Configured FinTech MCP server instance."""
    return create_fintech_mcp_server(buffer_capacity=1000)
