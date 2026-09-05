"""Comprehensive test suite for simulated order execution, L2 matching, and portfolio tracking."""

from __future__ import annotations

import pytest

from high_performance_model.execution.engine import ExecutionSimulator
from high_performance_model.execution.models import (
    OrderStatus,
    Portfolio,
    Position,
    SimulatedOrder,
)
from high_performance_model.types import (
    MarketTick,
    OrderBook,
    OrderBookLevel,
    OrderType,
    Side,
)


class TestExecutionModels:
    """Test SimulatedOrder, Position, and Portfolio domain logic."""

    def test_simulated_order_invariants(self) -> None:
        order = SimulatedOrder(
            symbol="AAPL",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=150.0,
        )
        assert order.is_active
        assert order.remaining_quantity == 100.0
        assert order.status == OrderStatus.PENDING

        d = order.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["order_type"] == "limit"
        assert d["price"] == 150.0

    def test_position_long_and_short_pnl(self) -> None:
        pos = Position(symbol="NVDA")
        assert pos.quantity == 0.0
        assert pos.unrealized_pnl == 0.0

        # Buy 10 @ 100
        pnl = pos.apply_fill(Side.BUY, 10.0, 100.0)
        assert pnl == 0.0
        assert pos.quantity == 10.0
        assert pos.average_entry_price == 100.0

        # Add 10 @ 110 -> average entry is 105
        pos.apply_fill(Side.BUY, 10.0, 110.0)
        assert pos.quantity == 20.0
        assert pos.average_entry_price == 105.0

        # Mark-to-market at 120
        pos.update_market_price(120.0)
        assert pos.unrealized_pnl == 20.0 * (120.0 - 105.0)  # 300.0
        assert pos.market_value == 2400.0

        # Sell 15 @ 125 -> closes 15 from 105 entry, realizes (125 - 105) * 15 = 300
        pnl_realized = pos.apply_fill(Side.SELL, 15.0, 125.0)
        assert pytest.approx(pnl_realized, 0.001) == 300.0
        assert pos.quantity == 5.0
        assert pos.average_entry_price == 105.0

    def test_portfolio_metrics(self) -> None:
        portfolio = Portfolio(initial_cash=50_000.0, cash_balance=50_000.0)
        pos = portfolio.get_or_create_position("MSFT")
        pos.apply_fill(Side.BUY, 10.0, 400.0)
        portfolio.cash_balance -= 4000.0

        pos.update_market_price(420.0)
        # Unrealized PnL: 10 * 20 = 200
        assert portfolio.total_unrealized_pnl == 200.0
        # Portfolio value: cash (46000) + long market value (4200) + unrealized (already in market value)
        assert portfolio.total_portfolio_value > 50_000.0


class TestExecutionSimulator:
    """Test ExecutionSimulator matching against order books and market ticks."""

    @pytest.fixture
    def setup_book(self) -> OrderBook:
        bids = [
            OrderBookLevel(price=100.0, quantity=10.0),
            OrderBookLevel(price=99.0, quantity=20.0),
            OrderBookLevel(price=98.0, quantity=50.0),
        ]
        asks = [
            OrderBookLevel(price=101.0, quantity=10.0),
            OrderBookLevel(price=102.0, quantity=20.0),
            OrderBookLevel(price=103.0, quantity=50.0),
        ]
        return OrderBook(symbol="AAPL", bids=bids, asks=asks)

    def test_market_buy_multi_level_fill_and_slippage(self, setup_book: OrderBook) -> None:
        sim = ExecutionSimulator()
        sim.update_order_book(setup_book)

        # Buy 25 units: takes 10 @ 101, 15 @ 102
        # Avg price: (10*101 + 15*102) / 25 = (1010 + 1530) / 25 = 2540 / 25 = 101.6
        order = sim.submit_order(
            symbol="AAPL", side=Side.BUY, quantity=25.0, order_type=OrderType.MARKET
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 25.0
        assert order.average_fill_price == 101.6
        assert order.fee_paid > 0.0

        # Check trades
        trades = sim.get_trade_history("AAPL")
        assert len(trades) == 2
        assert trades[0].price == 101.0
        assert trades[0].size == 10.0
        assert trades[1].price == 102.0
        assert trades[1].size == 15.0

        # Check position
        pos = sim.portfolio.positions["AAPL"]
        assert pos.quantity == 25.0
        assert pos.average_entry_price == 101.6

    def test_limit_order_execution_on_ticks(self) -> None:
        sim = ExecutionSimulator()

        # Submit limit buy order at 95.0
        order = sim.submit_order(
            symbol="GOOGL",
            side=Side.BUY,
            quantity=10.0,
            order_type=OrderType.LIMIT,
            price=95.0,
        )
        assert order.status == OrderStatus.PENDING

        # Tick at 96.0 -> does not fill
        tick_high = MarketTick(symbol="GOOGL", price=96.0, size=5.0, side=Side.SELL)
        fills_1 = sim.on_tick(tick_high)
        assert len(fills_1) == 0
        assert order.status == OrderStatus.PENDING

        # Tick at 94.5 -> fills!
        tick_low = MarketTick(symbol="GOOGL", price=94.5, size=5.0, side=Side.SELL)
        fills_2 = sim.on_tick(tick_low)
        assert len(fills_2) == 1
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10.0
        assert order.average_fill_price == 95.0

    def test_stop_loss_trigger(self, setup_book: OrderBook) -> None:
        sim = ExecutionSimulator()
        sim.update_order_book(setup_book)

        # Place stop loss to sell if price drops to or below 99.5
        order = sim.submit_order(
            symbol="AAPL",
            side=Side.SELL,
            quantity=5.0,
            order_type=OrderType.STOP,
            stop_price=99.5,
        )
        assert order.status == OrderStatus.PENDING

        # Tick above stop -> untouched
        sim.on_tick(MarketTick(symbol="AAPL", price=100.0, size=1.0, side=Side.BUY))
        assert order.status == OrderStatus.PENDING

        # Tick at 99.0 -> triggers stop and matches against bids (best bid 100.0)
        sim.on_tick(MarketTick(symbol="AAPL", price=99.0, size=1.0, side=Side.SELL))
        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == 100.0

    def test_order_cancellation(self) -> None:
        sim = ExecutionSimulator()
        order = sim.submit_order("MSFT", Side.BUY, 10.0, order_type=OrderType.LIMIT, price=300.0)
        assert order.is_active

        cancelled = sim.cancel_order(order.order_id)
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED
        assert not cancelled.is_active

        # Non-existent order cancellation returns None
        assert sim.cancel_order("fake_id") is None
