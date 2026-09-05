"""Real-time market condition monitoring and alert evaluation engine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from high_performance_model.alerts.models import (
    AlertEvent,
    AlertRule,
    AlertType,
)
from high_performance_model.types import (
    MarketTick,
    OrderBook,
    validate_symbol,
)

logger = logging.getLogger("high_performance_model.alerts.engine")


class AlertEngine:
    """Evaluates market conditions against registered alert thresholds in real-time."""

    def __init__(self) -> None:
        self.rules: Dict[str, AlertRule] = {}
        self.events: List[AlertEvent] = []
        self._listeners: List[Callable[[AlertEvent], None]] = []

    def create_rule(
        self,
        symbol: str,
        alert_type: AlertType,
        threshold: float,
        one_shot: bool = True,
        message: Optional[str] = None,
    ) -> AlertRule:
        """Register a new monitoring rule."""
        sym = validate_symbol(symbol)
        rule = AlertRule(
            symbol=sym,
            alert_type=alert_type,
            threshold=threshold,
            one_shot=one_shot,
            message=message,
        )
        self.rules[rule.alert_id] = rule
        return rule

    def delete_rule(self, alert_id: str) -> bool:
        """Remove a rule by ID."""
        if alert_id in self.rules:
            del self.rules[alert_id]
            return True
        return False

    def get_active_rules(self, symbol: Optional[str] = None) -> List[AlertRule]:
        """List active untriggered or recurring rules."""
        active = [r for r in self.rules.values() if not r.triggered or not r.one_shot]
        if symbol:
            sym = validate_symbol(symbol)
            active = [r for r in active if r.symbol == sym]
        return active

    def get_event_history(self, symbol: Optional[str] = None) -> List[AlertEvent]:
        """List fired alert events."""
        evts = self.events
        if symbol:
            sym = validate_symbol(symbol)
            evts = [e for e in evts if e.symbol == sym]
        return evts

    def subscribe(self, callback: Callable[[AlertEvent], None]) -> None:
        """Register a callback for fired alert events."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[AlertEvent], None]) -> None:
        """Remove a registered alert callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _fire_alert(
        self, rule: AlertRule, current_value: float, custom_msg: Optional[str] = None
    ) -> AlertEvent:
        """Mark rule triggered and notify subscribers."""
        now = datetime.now(timezone.utc)
        rule.triggered = True
        rule.triggered_at = now
        rule.trigger_count += 1

        msg = (
            custom_msg
            or rule.message
            or (
                f"Alert {rule.alert_type.value} on {rule.symbol}: "
                f"current {current_value} breached threshold {rule.threshold}"
            )
        )

        event = AlertEvent(
            alert_id=rule.alert_id,
            symbol=rule.symbol,
            alert_type=rule.alert_type,
            threshold=rule.threshold,
            current_value=current_value,
            message=msg,
            timestamp=now,
        )
        self.events.append(event)

        for listener in self._listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.warning("Error in alert listener callback: %s", exc)

        return event

    def on_tick(self, tick: MarketTick) -> List[AlertEvent]:
        """Evaluate tick price and size against active rules."""
        sym = tick.symbol
        fired: List[AlertEvent] = []

        for rule in list(self.rules.values()):
            if rule.symbol != sym:
                continue
            if rule.triggered and rule.one_shot:
                continue

            if rule.alert_type == AlertType.PRICE_ABOVE and tick.price >= rule.threshold:
                fired.append(self._fire_alert(rule, tick.price))
            elif rule.alert_type == AlertType.PRICE_BELOW and tick.price <= rule.threshold:
                fired.append(self._fire_alert(rule, tick.price))
            elif rule.alert_type == AlertType.VOLUME_SPIKE and tick.size >= rule.threshold:
                fired.append(self._fire_alert(rule, tick.size))

        return fired

    def on_order_book(self, order_book: OrderBook) -> List[AlertEvent]:
        """Evaluate spread and order book imbalance against active rules."""
        sym = order_book.symbol
        fired: List[AlertEvent] = []

        for rule in list(self.rules.values()):
            if rule.symbol != sym:
                continue
            if rule.triggered and rule.one_shot:
                continue

            if rule.alert_type == AlertType.SPREAD_WIDER_THAN:
                spread = order_book.spread
                if spread is not None and spread >= rule.threshold:
                    fired.append(self._fire_alert(rule, spread))

            elif rule.alert_type == AlertType.IMBALANCE_SPIKE:
                imb = abs(order_book.order_book_imbalance)
                if imb >= rule.threshold:
                    fired.append(self._fire_alert(rule, imb))

        return fired

    def clear(self) -> None:
        """Reset all rules and events."""
        self.rules.clear()
        self.events.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Summarize alert engine state."""
        return {
            "total_rules": len(self.rules),
            "active_rules": len(self.get_active_rules()),
            "total_events_fired": len(self.events),
            "recent_events": [e.to_dict() for e in self.events[-10:]],
        }
