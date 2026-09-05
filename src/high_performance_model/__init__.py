"""High-performance Model Context Protocol (MCP) server for FinTech market telemetry."""

from high_performance_model.protocol.server import MCPServer
from high_performance_model.server.app import create_fintech_mcp_server
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.types import (
    Bar,
    MarketTick,
    OrderBook,
    OrderBookLevel,
    Side,
    TechnicalIndicatorResult,
)

__version__ = "0.1.0"
__all__ = [
    "Bar",
    "MCPServer",
    "MarketDataBuffer",
    "MarketTick",
    "OrderBook",
    "OrderBookLevel",
    "Side",
    "TechnicalIndicatorResult",
    "create_fintech_mcp_server",
]
