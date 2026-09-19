"""Stress testing scenarios and extreme event impact analyzers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from high_performance_model.risk.models import RiskLevel, StressScenario, StressTestResult


class StressTestingSuite:
    """Pre-built and custom historical/hypothetical market stress scenarios."""

    STANDARD_SCENARIOS: Dict[str, StressScenario] = {
        "2008_financial_crisis": StressScenario(
            name="2008 Financial Crisis",
            description="Lehman collapse credit contagion and systemic deleveraging",
            price_shock_pct=-0.35,
            volatility_multiplier=2.8,
            liquidity_haircut_pct=0.25,
            correlation_spike=0.45,
        ),
        "covid_liquidity_shock": StressScenario(
            name="COVID-19 Liquidity Shock",
            description="March 2020 sudden volatility explosion and margin calls",
            price_shock_pct=-0.25,
            volatility_multiplier=2.5,
            liquidity_haircut_pct=0.20,
            correlation_spike=0.40,
        ),
        "tech_flash_crash": StressScenario(
            name="Tech Flash Crash",
            description="Algorithmic order book cascade and intraday liquidity vacuum",
            price_shock_pct=-0.12,
            volatility_multiplier=1.8,
            liquidity_haircut_pct=0.15,
            correlation_spike=0.25,
        ),
        "stagflation_rate_spike": StressScenario(
            name="Stagflation & Rate Hike Shock",
            description="Aggressive central bank surprise tightening and equity repricing",
            price_shock_pct=-0.18,
            volatility_multiplier=1.6,
            liquidity_haircut_pct=0.10,
            correlation_spike=0.20,
        ),
        "crypto_black_swan": StressScenario(
            name="Crypto Contagion Black Swan",
            description="Major exchange or protocol failure causing cross-asset liquidation",
            price_shock_pct=-0.45,
            volatility_multiplier=3.2,
            liquidity_haircut_pct=0.35,
            correlation_spike=0.60,
        ),
    }

    @classmethod
    def list_scenarios(cls) -> List[StressScenario]:
        """Return list of standard available stress scenarios."""
        return list(cls.STANDARD_SCENARIOS.values())

    @classmethod
    def get_scenario(cls, key: str) -> Optional[StressScenario]:
        """Retrieve a standard scenario by lookup key."""
        return cls.STANDARD_SCENARIOS.get(key)

    @classmethod
    def evaluate_scenario(
        cls,
        symbol: str,
        initial_price: float,
        current_volatility: float,
        scenario: StressScenario,
    ) -> StressTestResult:
        """Apply stress scenario parameters and calculate immediate stressed market risk."""
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")

        stressed_price = round(initial_price * (1.0 + scenario.price_shock_pct), 4)
        stressed_vol = round(current_volatility * scenario.volatility_multiplier, 4)

        # Parametric 95% 1-day VaR under stressed volatility (1.64485 * stressed_daily_vol)
        daily_stressed_vol = stressed_vol / math.sqrt(252.0)
        stressed_var_95 = round(1.64485 * daily_stressed_vol, 4)

        # Risk level determination based on total price shock and stressed volatility
        abs_shock = abs(scenario.price_shock_pct)
        if abs_shock >= 0.30 or stressed_vol >= 0.80:
            risk_lvl = RiskLevel.CRITICAL
        elif abs_shock >= 0.20 or stressed_vol >= 0.50:
            risk_lvl = RiskLevel.HIGH
        elif abs_shock >= 0.10 or stressed_vol >= 0.30:
            risk_lvl = RiskLevel.ELEVATED
        else:
            risk_lvl = RiskLevel.MODERATE

        return StressTestResult(
            symbol=symbol,
            scenario_name=scenario.name,
            initial_price=initial_price,
            stressed_price=stressed_price,
            pnl_shock_pct=scenario.price_shock_pct,
            stressed_volatility=stressed_vol,
            stressed_var_95=stressed_var_95,
            risk_level=risk_lvl,
        )
