"""Domain models for market telemetry alerts and event notifications."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from high_performance_model.types import Symbol


class AlertType(str, Enum):
    """Supported market telemetry alert types."""

    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    SPREAD_WIDER_THAN = "spread_wider_than"
    IMBALANCE_SPIKE = "imbalance_spike"
    VOLUME_SPIKE = "volume_spike"


class AlertRule(BaseModel):
    """Definition of a threshold-based market alert."""

    alert_id: str = Field(default_factory=lambda: f"alt_{uuid4().hex[:8]}")
    symbol: Symbol
    alert_type: AlertType
    threshold: float
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    trigger_count: int = 0
    one_shot: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert rule to dictionary."""
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "trigger_count": self.trigger_count,
            "one_shot": self.one_shot,
            "created_at": self.created_at.isoformat(),
            "message": self.message,
        }


class AlertEvent(BaseModel):
    """Fired alert event instance."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:8]}")
    alert_id: str
    symbol: Symbol
    alert_type: AlertType
    threshold: float
    current_value: float
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert event to dictionary."""
        return {
            "event_id": self.event_id,
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }
