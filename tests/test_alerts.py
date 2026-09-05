"""Comprehensive test suite for real-time market alerting and notification subsystems."""

from __future__ import annotations

from high_performance_model.alerts.engine import AlertEngine
from high_performance_model.alerts.models import (
    AlertEvent,
    AlertRule,
    AlertType,
)
from high_performance_model.types import (
    MarketTick,
    OrderBook,
    OrderBookLevel,
    Side,
)


class TestAlertModels:
    """Test AlertRule and AlertEvent domain models."""

    def test_alert_rule_defaults_and_serialization(self) -> None:
        rule = AlertRule(
            symbol="AAPL",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=250.0,
            message="Apple above 250",
        )
        assert rule.symbol == "AAPL"
        assert not rule.triggered
        assert rule.one_shot is True
        d = rule.to_dict()
        assert d["alert_type"] == "price_above"
        assert d["threshold"] == 250.0


class TestAlertEngine:
    """Test real-time market condition monitoring and triggers."""

    def test_create_and_delete_rule(self) -> None:
        engine = AlertEngine()
        rule = engine.create_rule("NVDA", AlertType.PRICE_ABOVE, 150.0)
        assert rule.alert_id in engine.rules
        assert len(engine.get_active_rules()) == 1

        deleted = engine.delete_rule(rule.alert_id)
        assert deleted is True
        assert len(engine.get_active_rules()) == 0
        assert engine.delete_rule("unknown_id") is False

    def test_price_above_and_below_triggers(self) -> None:
        engine = AlertEngine()
        engine.create_rule("AAPL", AlertType.PRICE_ABOVE, 200.0, one_shot=True)
        engine.create_rule("AAPL", AlertType.PRICE_BELOW, 180.0, one_shot=True)

        # Tick at 190 -> no triggers
        events_1 = engine.on_tick(MarketTick(symbol="AAPL", price=190.0, size=10.0, side=Side.BUY))
        assert len(events_1) == 0

        # Tick at 205 -> triggers price_above
        events_2 = engine.on_tick(MarketTick(symbol="AAPL", price=205.0, size=10.0, side=Side.BUY))
        assert len(events_2) == 1
        assert events_2[0].alert_type == AlertType.PRICE_ABOVE
        assert events_2[0].current_value == 205.0

        # Further tick at 210 -> does not trigger because one_shot is True
        events_3 = engine.on_tick(MarketTick(symbol="AAPL", price=210.0, size=10.0, side=Side.BUY))
        assert len(events_3) == 0

        # Tick at 175 -> triggers price_below
        events_4 = engine.on_tick(MarketTick(symbol="AAPL", price=175.0, size=10.0, side=Side.SELL))
        assert len(events_4) == 1
        assert events_4[0].alert_type == AlertType.PRICE_BELOW

    def test_volume_spike_trigger(self) -> None:
        engine = AlertEngine()
        engine.create_rule("MSFT", AlertType.VOLUME_SPIKE, 1000.0)

        assert len(engine.on_tick(MarketTick(symbol="MSFT", price=400.0, size=50.0, side=Side.BUY))) == 0
        events = engine.on_tick(MarketTick(symbol="MSFT", price=400.0, size=1500.0, side=Side.BUY))
        assert len(events) == 1
        assert events[0].alert_type == AlertType.VOLUME_SPIKE

    def test_order_book_spread_and_imbalance_triggers(self) -> None:
        engine = AlertEngine()
        engine.create_rule("BTC/USD", AlertType.SPREAD_WIDER_THAN, 10.0)
        engine.create_rule("BTC/USD", AlertType.IMBALANCE_SPIKE, 0.5)

        # Normal tight book
        tight_ob = OrderBook(
            symbol="BTC/USD",
            bids=[OrderBookLevel(price=60000.0, quantity=10.0)],
            asks=[OrderBookLevel(price=60001.0, quantity=10.0)],
        )
        assert len(engine.on_order_book(tight_ob)) == 0

        # Wide book with heavy imbalance
        # Bids: 100 qty @ 59990.0; Asks: 5 qty @ 60020.0 (spread = 30.0 > 10.0, imbalance > 0.5)
        wide_ob = OrderBook(
            symbol="BTC/USD",
            bids=[OrderBookLevel(price=59990.0, quantity=100.0)],
            asks=[OrderBookLevel(price=60020.0, quantity=5.0)],
        )
        events = engine.on_order_book(wide_ob)
        assert len(events) == 2
        types = {e.alert_type for e in events}
        assert AlertType.SPREAD_WIDER_THAN in types
        assert AlertType.IMBALANCE_SPIKE in types

    def test_listener_subscription(self) -> None:
        engine = AlertEngine()
        engine.create_rule("AAPL", AlertType.PRICE_ABOVE, 100.0)

        received: list[AlertEvent] = []

        def _callback(evt: AlertEvent) -> None:
            received.append(evt)

        engine.subscribe(_callback)
        engine.on_tick(MarketTick(symbol="AAPL", price=105.0, size=1.0, side=Side.BUY))
        assert len(received) == 1
        assert received[0].current_value == 105.0

        # Unsubscribe
        engine.unsubscribe(_callback)
        engine.create_rule("AAPL", AlertType.PRICE_ABOVE, 110.0)
        engine.on_tick(MarketTick(symbol="AAPL", price=115.0, size=1.0, side=Side.BUY))
        assert len(received) == 1
