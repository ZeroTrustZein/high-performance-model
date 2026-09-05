"""High-throughput simulated execution and order matching engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from high_performance_model.execution.models import (
    OrderStatus,
    Portfolio,
    SimulatedOrder,
)
from high_performance_model.types import (
    MarketTick,
    OrderBook,
    OrderType,
    Side,
    TimeInForce,
    TradeExecution,
    validate_symbol,
)


class ExecutionSimulator:
    """Matches simulated orders against L2 depth and market ticks with fee modeling."""

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        maker_fee_bps: float = 1.0,
        taker_fee_bps: float = 5.0,
    ) -> None:
        self.portfolio = portfolio or Portfolio()
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.orders: Dict[str, SimulatedOrder] = {}
        self.trades: List[TradeExecution] = []
        self._latest_books: Dict[str, OrderBook] = {}

    def update_order_book(self, order_book: OrderBook) -> None:
        """Cache latest order book for execution matching."""
        self._latest_books[order_book.symbol] = order_book

    def submit_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        auto_match: bool = True,
    ) -> SimulatedOrder:
        """Submit a simulated order and attempt matching if market order."""
        sym = validate_symbol(symbol)
        order = SimulatedOrder(
            symbol=sym,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.PENDING,
        )
        self.orders[order.order_id] = order

        if auto_match and order_type == OrderType.MARKET:
            ob = self._latest_books.get(sym)
            if ob:
                self.match_market_order(order, ob)

        return order

    def cancel_order(self, order_id: str) -> Optional[SimulatedOrder]:
        """Cancel an open simulated order."""
        order = self.orders.get(order_id)
        if not order:
            return None
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
        return order

    def _record_fill(
        self,
        symbol: str,
        side: Side,
        fill_qty: float,
        fill_price: float,
        order_type: OrderType,
        fee_bps: float,
        timestamp: datetime,
        maker_order_id: Optional[str] = None,
        taker_order_id: Optional[str] = None,
    ) -> TradeExecution:
        """Record trade execution and update portfolio cash, fees, and position state."""
        cost = fill_qty * fill_price
        fee = round(cost * (fee_bps / 10_000.0), 6)

        trade = TradeExecution(
            trade_id=f"trd_{uuid4().hex[:8]}",
            symbol=symbol,
            price=fill_price,
            size=round(fill_qty, 6),
            side=side,
            order_type=order_type,
            maker_order_id=maker_order_id,
            taker_order_id=taker_order_id,
            fee=fee,
            timestamp=timestamp,
        )
        self.trades.append(trade)

        pos = self.portfolio.get_or_create_position(symbol)
        pnl_realized = pos.apply_fill(side, fill_qty, fill_price)
        pos.update_market_price(fill_price)

        if side == Side.BUY:
            self.portfolio.cash_balance -= cost + fee
        else:
            self.portfolio.cash_balance += cost - fee

        self.portfolio.total_fees_paid += fee
        self.portfolio.total_realized_pnl += pnl_realized
        return trade

    def match_market_order(
        self, order: SimulatedOrder, order_book: OrderBook
    ) -> List[TradeExecution]:
        """Walk L2 order book levels and execute market order fills."""
        if not order.is_active or order.remaining_quantity <= 0:
            return []

        levels = order_book.asks if order.side == Side.BUY else order_book.bids
        if not levels:
            return []

        trades: List[TradeExecution] = []
        remaining = order.remaining_quantity
        total_filled = 0.0
        total_cost = 0.0

        for level in levels:
            fill_qty = min(remaining, level.quantity)
            if fill_qty <= 0:
                continue

            fill_price = level.price
            trade = self._record_fill(
                symbol=order.symbol,
                side=order.side,
                fill_qty=fill_qty,
                fill_price=fill_price,
                order_type=OrderType.MARKET,
                fee_bps=self.taker_fee_bps,
                timestamp=datetime.now(timezone.utc),
                taker_order_id=order.order_id,
            )
            trades.append(trade)

            total_filled += fill_qty
            total_cost += fill_qty * fill_price
            remaining -= fill_qty

            if remaining <= 1e-9:
                break

        if total_filled > 0:
            order.filled_quantity = round(order.filled_quantity + total_filled, 6)
            order.fee_paid = round(order.fee_paid + sum(t.fee for t in trades), 6)

            prev_cost = (order.average_fill_price or 0.0) * (order.filled_quantity - total_filled)
            order.average_fill_price = round((prev_cost + total_cost) / order.filled_quantity, 4)

            if order.remaining_quantity <= 1e-9:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            order.updated_at = datetime.now(timezone.utc)

        return trades

    def on_tick(self, tick: MarketTick) -> List[TradeExecution]:
        """Incorporate market tick: update positions mark-to-market and process limit orders."""
        sym = tick.symbol
        # Update mark-to-market
        if sym in self.portfolio.positions:
            self.portfolio.positions[sym].update_market_price(tick.price)

        trades: List[TradeExecution] = []
        # Check active limit and stop orders
        for order in list(self.orders.values()):
            if not order.is_active or order.symbol != sym:
                continue

            if order.order_type == OrderType.LIMIT and order.price is not None:
                can_fill = False
                if order.side == Side.BUY and tick.price <= order.price:
                    can_fill = True
                elif order.side == Side.SELL and tick.price >= order.price:
                    can_fill = True

                if can_fill:
                    fill_qty = order.remaining_quantity
                    fill_price = order.price
                    trade = self._record_fill(
                        symbol=sym,
                        side=order.side,
                        fill_qty=fill_qty,
                        fill_price=fill_price,
                        order_type=OrderType.LIMIT,
                        fee_bps=self.maker_fee_bps,
                        timestamp=tick.timestamp,
                        maker_order_id=order.order_id,
                    )
                    trades.append(trade)

                    order.filled_quantity = order.quantity
                    order.average_fill_price = fill_price
                    order.fee_paid += trade.fee
                    order.status = OrderStatus.FILLED
                    order.updated_at = datetime.now(timezone.utc)

            elif order.order_type == OrderType.STOP and order.stop_price is not None:
                # Stop loss triggered: convert to market order
                triggered = False
                if order.side == Side.SELL and tick.price <= order.stop_price:
                    triggered = True
                elif order.side == Side.BUY and tick.price >= order.stop_price:
                    triggered = True

                if triggered:
                    order.order_type = OrderType.MARKET
                    ob = self._latest_books.get(sym)
                    if ob:
                        trades.extend(self.match_market_order(order, ob))

        return trades

    def get_open_orders(self, symbol: Optional[str] = None) -> List[SimulatedOrder]:
        """List currently pending or partially filled orders."""
        orders = [o for o in self.orders.values() if o.is_active]
        if symbol:
            sym = validate_symbol(symbol)
            orders = [o for o in orders if o.symbol == sym]
        return orders

    def get_order_history(self, symbol: Optional[str] = None) -> List[SimulatedOrder]:
        """List all historical orders."""
        orders = list(self.orders.values())
        if symbol:
            sym = validate_symbol(symbol)
            orders = [o for o in orders if o.symbol == sym]
        return orders

    def get_trade_history(self, symbol: Optional[str] = None) -> List[TradeExecution]:
        """List all executed trades."""
        trades = self.trades
        if symbol:
            sym = validate_symbol(symbol)
            trades = [t for t in trades if t.symbol == sym]
        return trades

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retrieve full portfolio and position status."""
        return self.portfolio.to_dict()
