"""Execution domain models: orders, positions, and portfolio state."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from high_performance_model.types import OrderType, Side, Symbol, TimeInForce, validate_symbol


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SimulatedOrder(BaseModel):
    """Simulated trading order data contract."""

    order_id: str = Field(default_factory=lambda: f"ord_{uuid4().hex[:8]}")
    symbol: Symbol
    side: Side
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(..., gt=0.0)
    price: Optional[float] = Field(default=None, gt=0.0)
    stop_price: Optional[float] = Field(default=None, gt=0.0)
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = Field(default=0.0, ge=0.0)
    average_fill_price: Optional[float] = Field(default=None, ge=0.0)
    fee_paid: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def remaining_quantity(self) -> float:
        """Unfilled quantity remaining on order."""
        return round(max(0.0, self.quantity - self.filled_quantity), 6)

    @property
    def is_active(self) -> bool:
        """Check whether order is still open for matching."""
        return self.status in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "fee_paid": self.fee_paid,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Position(BaseModel):
    """Real-time position tracking for a single asset."""

    symbol: Symbol
    quantity: float = 0.0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss based on current market price."""
        if abs(self.quantity) < 1e-9 or self.current_price <= 0.0:
            return 0.0
        return round(self.quantity * (self.current_price - self.average_entry_price), 4)

    @property
    def market_value(self) -> float:
        """Total current market value of holding."""
        return round(abs(self.quantity) * self.current_price, 4)

    def update_market_price(self, price: float) -> None:
        """Update mark-to-market price."""
        if price > 0.0:
            self.current_price = round(price, 4)

    def apply_fill(self, fill_side: Side, fill_qty: float, fill_price: float) -> float:
        """Incorporate execution fill into position, returning any realized PnL."""
        if fill_qty <= 0.0:
            return 0.0

        pnl_realized = 0.0

        if fill_side == Side.BUY:
            if self.quantity >= 0:
                # Adding to long position
                new_qty = self.quantity + fill_qty
                total_cost = (self.quantity * self.average_entry_price) + (fill_qty * fill_price)
                self.average_entry_price = round(total_cost / new_qty, 4)
                self.quantity = round(new_qty, 6)
            else:
                # Covering short position
                closing_qty = min(abs(self.quantity), fill_qty)
                pnl_realized = (self.average_entry_price - fill_price) * closing_qty
                self.realized_pnl = round(self.realized_pnl + pnl_realized, 4)
                rem_short = self.quantity + closing_qty

                if abs(rem_short) < 1e-9:
                    excess_buy = fill_qty - closing_qty
                    if excess_buy > 0:
                        self.quantity = round(excess_buy, 6)
                        self.average_entry_price = round(fill_price, 4)
                    else:
                        self.quantity = 0.0
                        self.average_entry_price = 0.0
                else:
                    self.quantity = round(rem_short, 6)

        else:  # Side.SELL
            if self.quantity <= 0:
                # Adding to short position
                new_qty = self.quantity - fill_qty
                total_cost = (abs(self.quantity) * self.average_entry_price) + (
                    fill_qty * fill_price
                )
                self.average_entry_price = round(total_cost / abs(new_qty), 4)
                self.quantity = round(new_qty, 6)
            else:
                # Closing long position
                closing_qty = min(self.quantity, fill_qty)
                pnl_realized = (fill_price - self.average_entry_price) * closing_qty
                self.realized_pnl = round(self.realized_pnl + pnl_realized, 4)
                rem_long = self.quantity - closing_qty

                if rem_long <= 1e-9:
                    excess_sell = fill_qty - closing_qty
                    if excess_sell > 0:
                        self.quantity = round(-excess_sell, 6)
                        self.average_entry_price = round(fill_price, 4)
                    else:
                        self.quantity = 0.0
                        self.average_entry_price = 0.0
                else:
                    self.quantity = round(rem_long, 6)

        return round(pnl_realized, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.market_value,
        }


class Portfolio(BaseModel):
    """Aggregated portfolio and cash tracking."""

    initial_cash: float = 100_000.0
    cash_balance: float = 100_000.0
    positions: Dict[str, Position] = Field(default_factory=dict)
    total_realized_pnl: float = 0.0
    total_fees_paid: float = 0.0

    @property
    def total_unrealized_pnl(self) -> float:
        """Combined unrealized PnL across all active positions."""
        return round(sum(p.unrealized_pnl for p in self.positions.values()), 4)

    @property
    def total_portfolio_value(self) -> float:
        """Cash balance plus market value of active positions."""
        long_val = sum(p.market_value for p in self.positions.values() if p.quantity > 0)
        return round(self.cash_balance + long_val + self.total_unrealized_pnl, 4)

    @property
    def total_return_pct(self) -> float:
        """Percentage return on initial cash."""
        if self.initial_cash <= 0:
            return 0.0
        return round(
            ((self.total_portfolio_value - self.initial_cash) / self.initial_cash) * 100.0, 4
        )

    def get_or_create_position(self, symbol: str) -> Position:
        """Retrieve existing position or create a clean one."""
        clean_sym = validate_symbol(symbol)
        if clean_sym not in self.positions:
            self.positions[clean_sym] = Position(symbol=clean_sym)
        return self.positions[clean_sym]

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio status to dictionary."""
        return {
            "initial_cash": self.initial_cash,
            "cash_balance": round(self.cash_balance, 4),
            "total_portfolio_value": self.total_portfolio_value,
            "total_realized_pnl": round(self.total_realized_pnl, 4),
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_fees_paid": round(self.total_fees_paid, 4),
            "total_return_pct": self.total_return_pct,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }
