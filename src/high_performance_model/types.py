"""Core domain types and data contracts for FinTech market telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Side(str, Enum):
    """Order / trade book side."""

    BUY = "buy"
    SELL = "sell"


class MarketTick(BaseModel):
    """High-frequency market tick telemetry record."""

    symbol: str
    price: float = Field(..., gt=0.0)
    size: float = Field(..., ge=0.0)
    side: Side
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert tick to dictionary with ISO timestamp."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        }


class OrderBookLevel(BaseModel):
    """Single price-quantity depth level in an order book."""

    price: float = Field(..., gt=0.0)
    quantity: float = Field(..., ge=0.0)
    orders_count: int = 1


class OrderBook(BaseModel):
    """Level-2 order book snapshot."""

    symbol: str
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def best_bid(self) -> Optional[float]:
        """Return highest bid price if present."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Return lowest ask price if present."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Return top-of-book bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return round(self.best_ask - self.best_bid, 6)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Return mid price between best bid and ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return round((self.best_bid + self.best_ask) / 2.0, 6)
        return None

    @property
    def order_book_imbalance(self) -> float:
        """Compute top-level depth imbalance ratio (-1.0 to +1.0)."""
        bid_vol = sum(level.quantity for level in self.bids[:5])
        ask_vol = sum(level.quantity for level in self.asks[:5])
        total_vol = bid_vol + ask_vol
        if total_vol == 0.0:
            return 0.0
        return round((bid_vol - ask_vol) / total_vol, 4)


class Bar(BaseModel):
    """Aggregated OHLCV candle bar."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    vwap: Optional[float] = None
    trades_count: int = 0


class TechnicalIndicatorResult(BaseModel):
    """Calculated technical indicator metrics."""

    symbol: str
    indicator: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    values: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
