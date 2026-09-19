"""Data contracts and domain models for quantitative risk and Monte Carlo stress testing."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from high_performance_model.types import Symbol


class RiskLevel(str, Enum):
    """Qualitative classification of portfolio/asset risk."""

    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class RiskRatioResult(BaseModel):
    """Calculated risk-adjusted performance ratios."""

    symbol: Symbol
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    downside_deviation: float = Field(..., ge=0.0)
    annualized_return: float
    annualized_volatility: float = Field(..., ge=0.0)
    risk_free_rate: float
    sample_size: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize risk ratio result to dictionary."""
        return {
            "symbol": self.symbol,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "downside_deviation": self.downside_deviation,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "risk_free_rate": self.risk_free_rate,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp.isoformat(),
        }


class MonteCarloConfig(BaseModel):
    """Configuration parameters for Monte Carlo stochastic simulations."""

    n_simulations: int = Field(default=5000, ge=10, le=100000)
    horizon_days: int = Field(default=30, ge=1, le=365)
    time_steps_per_day: int = Field(default=1, ge=1, le=24)
    confidence_levels: List[float] = Field(default_factory=lambda: [0.90, 0.95, 0.99])
    random_seed: Optional[int] = None
    jump_diffusion: bool = False
    jump_intensity: float = Field(default=0.1, ge=0.0)
    jump_mean: float = 0.0
    jump_std: float = Field(default=0.05, ge=0.0)


class MonteCarloStressResult(BaseModel):
    """Aggregated output from Monte Carlo price and return trajectory simulations."""

    symbol: Symbol
    initial_price: float = Field(..., gt=0.0)
    horizon_days: int = Field(..., ge=1)
    n_simulations: int = Field(..., ge=1)
    expected_final_price: float
    median_final_price: float
    worst_case_drawdown: float = Field(..., le=0.0)
    probability_of_loss: float = Field(..., ge=0.0, le=1.0)
    var_by_confidence: Dict[str, float] = Field(default_factory=dict)
    cvar_by_confidence: Dict[str, float] = Field(default_factory=dict)
    percentile_trajectories: Dict[str, List[float]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Monte Carlo result to dictionary."""
        return {
            "symbol": self.symbol,
            "initial_price": self.initial_price,
            "horizon_days": self.horizon_days,
            "n_simulations": self.n_simulations,
            "expected_final_price": self.expected_final_price,
            "median_final_price": self.median_final_price,
            "worst_case_drawdown": self.worst_case_drawdown,
            "probability_of_loss": self.probability_of_loss,
            "var_by_confidence": dict(self.var_by_confidence),
            "cvar_by_confidence": dict(self.cvar_by_confidence),
            "percentile_trajectories": {k: list(v) for k, v in self.percentile_trajectories.items()},
            "timestamp": self.timestamp.isoformat(),
        }


class StressScenario(BaseModel):
    """Macroeconomic or market shock stress scenario specification."""

    name: str
    description: str
    price_shock_pct: float = Field(..., description="Instantaneous price shock percentage (-0.20 = -20%)")
    volatility_multiplier: float = Field(default=1.5, gt=0.0)
    liquidity_haircut_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    correlation_spike: float = Field(default=0.2, ge=0.0, le=1.0)


class StressTestResult(BaseModel):
    """Outcome of applying a stress scenario to a position or market series."""

    symbol: Symbol
    scenario_name: str
    initial_price: float = Field(..., gt=0.0)
    stressed_price: float = Field(..., gt=0.0)
    pnl_shock_pct: float
    stressed_volatility: float = Field(..., ge=0.0)
    stressed_var_95: float
    risk_level: RiskLevel
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stress test result to dictionary."""
        return {
            "symbol": self.symbol,
            "scenario_name": self.scenario_name,
            "initial_price": self.initial_price,
            "stressed_price": self.stressed_price,
            "pnl_shock_pct": self.pnl_shock_pct,
            "stressed_volatility": self.stressed_volatility,
            "stressed_var_95": self.stressed_var_95,
            "risk_level": self.risk_level.value,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskTelemetrySnapshot(BaseModel):
    """Streaming risk telemetry point emitted by the risk engine."""

    symbol: Symbol
    ratios: RiskRatioResult
    monte_carlo_var_95: Optional[float] = None
    stress_impacts: List[StressTestResult] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize risk telemetry snapshot."""
        return {
            "symbol": self.symbol,
            "ratios": self.ratios.to_dict(),
            "monte_carlo_var_95": self.monte_carlo_var_95,
            "stress_impacts": [s.to_dict() for s in self.stress_impacts],
            "risk_level": self.risk_level.value,
            "timestamp": self.timestamp.isoformat(),
        }
