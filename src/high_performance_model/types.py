"""Core domain types, enums, data contracts, and validators for FinTech market telemetry."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Side(str, Enum):
    """Order and trade execution side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Financial order execution types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    """Order time-in-force instructions."""

    GTC = "GTC"  # Good 'Til Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Good for Day
    GTD = "GTD"  # Good 'Til Date


class AssetClass(str, Enum):
    """Financial market asset classifications."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"
    INDEX = "index"
    OPTION = "option"
    FUTURE = "future"


class IndicatorType(str, Enum):
    """Supported technical analysis indicators."""

    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    ATR = "atr"
    VWAP = "vwap"
    MOMENTUM = "momentum"
    ROC = "roc"
    STOCHASTIC = "stochastic"


class BarTimeframe(str, Enum):
    """Candlestick aggregation intervals."""

    SEC_1 = "1s"
    SEC_5 = "5s"
    SEC_15 = "15s"
    SEC_30 = "30s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class MarketState(str, Enum):
    """Market session operating state."""

    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    CLOSED = "closed"
    HALTED = "halted"


class LiquidityTier(str, Enum):
    """Market asset liquidity categorization."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    ILLIQUID = "illiquid"


# Regex for valid tickers: e.g., AAPL, BTC/USD, EUR-USD, SPY.US
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_\-\.\/]{1,20}$")


def validate_symbol(symbol: str) -> str:
    """Validate and normalize ticker symbol representation."""
    if not isinstance(symbol, str):
        raise TypeError(f"Symbol must be a string, got {type(symbol).__name__}")
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Symbol cannot be empty")
    if not _SYMBOL_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid ticker symbol format: '{cleaned}'. Must contain 1-20 alphanumeric or ./-_ characters"
        )
    return cleaned


def timeframe_to_seconds(timeframe: Union[BarTimeframe, str]) -> int:
    """Convert bar timeframe enum or code string to total seconds."""
    raw = timeframe.value if isinstance(timeframe, BarTimeframe) else str(timeframe).strip().lower()
    mapping: Dict[str, int] = {
        "1s": 1,
        "5s": 5,
        "15s": 15,
        "30s": 30,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported timeframe: '{raw}'. Supported: {list(mapping.keys())}")
    return mapping[raw]


def compute_spread_bps(bid: float, ask: float) -> float:
    """Compute bid-ask spread in basis points (bps) relative to mid-price."""
    if bid <= 0.0 or ask <= 0.0:
        raise ValueError("Bid and ask prices must be strictly positive")
    if ask < bid:
        raise ValueError(f"Ask price ({ask}) cannot be strictly lower than bid price ({bid})")
    mid = (bid + ask) / 2.0
    return round(((ask - bid) / mid) * 10_000.0, 4)


def compute_microprice(bid: float, ask: float, bid_qty: float, ask_qty: float) -> float:
    """Compute order book microprice weighted by opposite side depth."""
    if bid <= 0.0 or ask <= 0.0:
        raise ValueError("Bid and ask prices must be strictly positive")
    if bid_qty < 0.0 or ask_qty < 0.0:
        raise ValueError("Quantities must be non-negative")
    total_qty = bid_qty + ask_qty
    if total_qty == 0.0:
        return round((bid + ask) / 2.0, 6)
    return round((bid * ask_qty + ask * bid_qty) / total_qty, 6)


class MarketTick(BaseModel):
    """High-frequency market tick telemetry record."""

    symbol: str
    price: float = Field(..., gt=0.0)
    size: float = Field(..., ge=0.0)
    side: Side
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = Field(default=0, ge=0)
    exchange: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    @property
    def notional(self) -> float:
        """Total trade notional value."""
        return round(self.price * self.size, 6)

    def is_uptick(self, previous: Optional[MarketTick]) -> Optional[bool]:
        """Determine if tick is an uptick, downtick, or zero-tick compared to predecessor."""
        if previous is None:
            return None
        if self.price > previous.price:
            return True
        if self.price < previous.price:
            return False
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert tick to dictionary with ISO timestamp."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "exchange": self.exchange,
            "conditions": list(self.conditions),
            "notional": self.notional,
        }


class OrderBookLevel(BaseModel):
    """Single price-quantity depth level in an order book."""

    price: float = Field(..., gt=0.0)
    quantity: float = Field(..., ge=0.0)
    orders_count: int = Field(default=1, ge=0)

    @property
    def notional(self) -> float:
        """Notional depth value at this price level."""
        return round(self.price * self.quantity, 6)

    def to_tuple(self) -> Tuple[float, float, int]:
        """Convert price level to compact (price, quantity, orders_count) tuple."""
        return (self.price, self.quantity, self.orders_count)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation."""
        return {
            "price": self.price,
            "quantity": self.quantity,
            "orders_count": self.orders_count,
            "notional": self.notional,
        }


class OrderBook(BaseModel):
    """Level-2 order book snapshot."""

    symbol: str
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = Field(default=0, ge=0)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

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
    def spread_bps(self) -> Optional[float]:
        """Return top-of-book bid-ask spread in basis points."""
        if self.best_bid is not None and self.best_ask is not None and self.mid_price:
            return round((self.best_ask - self.best_bid) / self.mid_price * 10_000.0, 4)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Return mid price between best bid and ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return round((self.best_bid + self.best_ask) / 2.0, 6)
        return None

    @property
    def microprice(self) -> Optional[float]:
        """Return volume-weighted microprice from top of book."""
        if self.best_bid is not None and self.best_ask is not None and self.bids and self.asks:
            return compute_microprice(
                bid=self.best_bid,
                ask=self.best_ask,
                bid_qty=self.bids[0].quantity,
                ask_qty=self.asks[0].quantity,
            )
        return self.mid_price

    @property
    def order_book_imbalance(self) -> float:
        """Compute top-level depth imbalance ratio (-1.0 to +1.0)."""
        bid_vol = sum(level.quantity for level in self.bids[:5])
        ask_vol = sum(level.quantity for level in self.asks[:5])
        total_vol = bid_vol + ask_vol
        if total_vol == 0.0:
            return 0.0
        return round((bid_vol - ask_vol) / total_vol, 4)

    @property
    def total_bid_volume(self) -> float:
        """Sum of all bid quantities."""
        return round(sum(level.quantity for level in self.bids), 6)

    @property
    def total_ask_volume(self) -> float:
        """Sum of all ask quantities."""
        return round(sum(level.quantity for level in self.asks), 6)

    @property
    def total_bid_notional(self) -> float:
        """Sum of all bid notionals."""
        return round(sum(level.notional for level in self.bids), 6)

    @property
    def total_ask_notional(self) -> float:
        """Sum of all ask notionals."""
        return round(sum(level.notional for level in self.asks), 6)

    @property
    def is_crossed(self) -> bool:
        """Whether the book is locked or crossed (best bid >= best ask)."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_bid >= self.best_ask
        return False

    def depth_summary(self, levels: int = 5) -> Dict[str, Any]:
        """Summarize order book depth up to N levels."""
        n = max(1, levels)
        return {
            "symbol": self.symbol,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "mid_price": self.mid_price,
            "microprice": self.microprice,
            "imbalance": self.order_book_imbalance,
            "bid_levels": len(self.bids[:n]),
            "ask_levels": len(self.asks[:n]),
            "total_bid_volume": round(sum(lvl.quantity for lvl in self.bids[:n]), 6),
            "total_ask_volume": round(sum(lvl.quantity for lvl in self.asks[:n]), 6),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert order book to dictionary."""
        return {
            "symbol": self.symbol,
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "mid_price": self.mid_price,
            "microprice": self.microprice,
            "imbalance": self.order_book_imbalance,
        }


class Bar(BaseModel):
    """Aggregated OHLCV candle bar."""

    symbol: str
    open: float = Field(..., gt=0.0)
    high: float = Field(..., gt=0.0)
    low: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    volume: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeframe: BarTimeframe = Field(default=BarTimeframe.MIN_1)
    vwap: Optional[float] = None
    trades_count: int = Field(default=0, ge=0)
    turnover: Optional[float] = None

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    @model_validator(mode="after")
    def _validate_candle_geometry(self) -> Bar:
        """Validate standard candlestick price relationships."""
        if self.high < self.low:
            raise ValueError(
                f"High price ({self.high}) cannot be strictly lower than Low price ({self.low})"
            )
        # Allow tiny float precision epsilon for numerical calculations
        epsilon = 1e-4
        if self.high < max(self.open, self.close) - epsilon:
            raise ValueError(
                f"High price ({self.high}) must be >= open ({self.open}) and close ({self.close})"
            )
        if self.low > min(self.open, self.close) + epsilon:
            raise ValueError(
                f"Low price ({self.low}) must be <= open ({self.open}) and close ({self.close})"
            )
        return self

    @property
    def range(self) -> float:
        """Total price range of the candle."""
        return round(self.high - self.low, 6)

    @property
    def body(self) -> float:
        """Absolute body size of the candle."""
        return round(abs(self.close - self.open), 6)

    @property
    def is_bullish(self) -> bool:
        """Whether close is greater than or equal to open."""
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        """Whether close is lower than open."""
        return self.close < self.open

    @property
    def typical_price(self) -> float:
        """Typical price (HLC / 3)."""
        return round((self.high + self.low + self.close) / 3.0, 6)

    def to_dict(self) -> Dict[str, Any]:
        """Convert candle bar to dictionary with ISO timestamp."""
        return {
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "vwap": self.vwap,
            "trades_count": self.trades_count,
            "turnover": self.turnover,
            "range": self.range,
            "body": self.body,
            "is_bullish": self.is_bullish,
            "typical_price": self.typical_price,
        }


class Quote(BaseModel):
    """Top-of-book market quote representation."""

    symbol: str
    bid: float = Field(..., gt=0.0)
    ask: float = Field(..., gt=0.0)
    bid_size: float = Field(..., ge=0.0)
    ask_size: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_price: Optional[float] = Field(default=None, gt=0.0)
    last_size: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    @model_validator(mode="after")
    def _validate_quote(self) -> Quote:
        if self.ask < self.bid:
            raise ValueError(
                f"Ask price ({self.ask}) cannot be strictly lower than Bid price ({self.bid})"
            )
        return self

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return round(self.ask - self.bid, 6)

    @property
    def mid_price(self) -> float:
        """Mid price."""
        return round((self.bid + self.ask) / 2.0, 6)

    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        return compute_spread_bps(self.bid, self.ask)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation."""
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "timestamp": self.timestamp.isoformat(),
            "last_price": self.last_price,
            "last_size": self.last_size,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
        }


class MarketDepthSnapshot(BaseModel):
    """Snapshot data contract of order book depth for tool responses."""

    symbol: str
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    spread: Optional[float] = None
    mid_price: Optional[float] = None
    imbalance: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "symbol": self.symbol,
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
            "spread": self.spread,
            "mid_price": self.mid_price,
            "imbalance": self.imbalance,
            "timestamp": self.timestamp.isoformat(),
        }


class IndicatorConfig(BaseModel):
    """Configuration parameters for technical indicator computation."""

    indicator: IndicatorType
    period: int = Field(default=20, gt=0)
    fast_period: int = Field(default=12, gt=0)
    slow_period: int = Field(default=26, gt=0)
    signal_period: int = Field(default=9, gt=0)
    num_std: float = Field(default=2.0, gt=0.0)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_macd_params(self) -> IndicatorConfig:
        if self.indicator == IndicatorType.MACD and self.fast_period >= self.slow_period:
            raise ValueError(
                f"MACD fast_period ({self.fast_period}) must be strictly less than slow_period ({self.slow_period})"
            )
        return self


class TechnicalIndicatorResult(BaseModel):
    """Calculated technical indicator metrics."""

    symbol: str
    indicator: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    values: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    def get_value(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Safely fetch metric value from result dictionary."""
        return self.values.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary with ISO timestamp."""
        return {
            "symbol": self.symbol,
            "indicator": self.indicator,
            "timestamp": self.timestamp.isoformat(),
            "values": dict(self.values),
            "metadata": dict(self.metadata),
        }


class RiskMetrics(BaseModel):
    """Quantitative portfolio and market telemetry risk analytics."""

    symbol: str
    realized_volatility: float = Field(..., ge=0.0)
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = Field(..., le=0.0)
    value_at_risk_95: Optional[float] = None
    expected_shortfall_95: Optional[float] = None
    sample_size: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert risk metrics to dictionary."""
        return {
            "symbol": self.symbol,
            "realized_volatility": self.realized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "value_at_risk_95": self.value_at_risk_95,
            "expected_shortfall_95": self.expected_shortfall_95,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


class MarketTelemetrySummary(BaseModel):
    """Comprehensive single-asset market telemetry state summary."""

    symbol: str
    last_price: float = Field(..., gt=0.0)
    last_size: float = Field(..., ge=0.0)
    last_side: Side
    volume_24h: Optional[float] = Field(default=None, ge=0.0)
    vwap: Optional[float] = Field(default=None, gt=0.0)
    high_24h: Optional[float] = Field(default=None, gt=0.0)
    low_24h: Optional[float] = Field(default=None, gt=0.0)
    spread: Optional[float] = Field(default=None, ge=0.0)
    change_24h_pct: Optional[float] = None
    market_state: MarketState = Field(default=MarketState.OPEN)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "last_size": self.last_size,
            "last_side": self.last_side.value,
            "volume_24h": self.volume_24h,
            "vwap": self.vwap,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "spread": self.spread,
            "change_24h_pct": self.change_24h_pct,
            "market_state": self.market_state.value,
            "timestamp": self.timestamp.isoformat(),
        }


class TradeExecution(BaseModel):
    """Simulated trade execution match record."""

    trade_id: str
    symbol: str
    price: float = Field(..., gt=0.0)
    size: float = Field(..., gt=0.0)
    side: Side
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    order_type: OrderType = Field(default=OrderType.LIMIT)
    maker_order_id: Optional[str] = None
    taker_order_id: Optional[str] = None
    fee: float = Field(default=0.0, ge=0.0)

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return validate_symbol(v)

    @property
    def notional(self) -> float:
        """Total execution notional value."""
        return round(self.price * self.size, 6)

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "order_type": self.order_type.value,
            "maker_order_id": self.maker_order_id,
            "taker_order_id": self.taker_order_id,
            "fee": self.fee,
            "notional": self.notional,
        }


class BenchmarkRunResult(BaseModel):
    """Performance benchmark execution metrics."""

    total_ticks: int = Field(..., ge=0)
    duration_seconds: float = Field(..., gt=0.0)
    ticks_per_second: float = Field(..., ge=0.0)
    latency_p50_us: float = Field(..., ge=0.0)
    latency_p95_us: float = Field(..., ge=0.0)
    latency_p99_us: float = Field(..., ge=0.0)
    buffer_utilization_pct: float = Field(..., ge=0.0, le=100.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark metrics to dictionary."""
        return {
            "total_ticks": self.total_ticks,
            "duration_seconds": round(self.duration_seconds, 6),
            "ticks_per_second": round(self.ticks_per_second, 2),
            "latency_p50_us": round(self.latency_p50_us, 2),
            "latency_p95_us": round(self.latency_p95_us, 2),
            "latency_p99_us": round(self.latency_p99_us, 2),
            "buffer_utilization_pct": round(self.buffer_utilization_pct, 2),
        }
