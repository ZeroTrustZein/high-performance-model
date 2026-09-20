"""Vectorized algorithms for Sharpe, Sortino, Calmar, and downside risk metrics."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np

from high_performance_model.indicators.series import max_drawdown
from high_performance_model.risk.models import RiskRatioResult


def calculate_downside_deviation(
    returns: Union[np.ndarray, Sequence[float]],
    target_return: float = 0.0,
) -> float:
    """Calculate downside deviation below a target return threshold."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    downside = np.minimum(0.0, arr - target_return)
    dev = float(math.sqrt(np.mean(downside**2)))
    return float(round(dev, 6))


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.045,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: Period returns array.
        risk_free_rate: Annualized risk-free benchmark rate (e.g. 0.045 for 4.5%).
        periods_per_year: Number of periods per annual cycle (default: 252 trading days).
        annualize: Whether to annualize mean and standard deviation.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year if annualize else risk_free_rate
    mean_excess = float(np.mean(arr - rf_per_period))
    ratio = mean_excess / std
    if annualize:
        ratio *= math.sqrt(periods_per_year)

    return float(round(ratio, 4))


def calculate_sortino_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    risk_free_rate: float = 0.045,
    target_return: float = 0.0,
    periods_per_year: float = 252.0,
    annualize: bool = True,
) -> float:
    """Calculate annualized Sortino Ratio focusing strictly on downside deviation.

    Args:
        returns: Period returns array.
        risk_free_rate: Annualized risk-free benchmark rate.
        target_return: Minimal acceptable return (MAR).
        periods_per_year: Number of periods per year.
        annualize: Whether to annualize the metric.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    target_per_period = target_return / periods_per_year if annualize else target_return
    downside_dev = calculate_downside_deviation(arr, target_return=target_per_period)
    if downside_dev < 1e-12:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year if annualize else risk_free_rate
    mean_excess = float(np.mean(arr - rf_per_period))
    ratio = mean_excess / downside_dev
    if annualize:
        ratio *= math.sqrt(periods_per_year)

    return float(round(ratio, 4))


def calculate_calmar_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    max_drawdown_pct: float,
    periods_per_year: float = 252.0,
) -> Optional[float]:
    """Calculate Calmar Ratio (Annualized Return / Absolute Max Drawdown)."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2 or abs(max_drawdown_pct) < 1e-6:
        return None

    annualized_ret = float(np.mean(arr) * periods_per_year)
    return float(round(annualized_ret / abs(max_drawdown_pct), 4))


def calculate_omega_ratio(
    returns: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.0,
) -> Optional[float]:
    """Calculate Omega Ratio (probability weighted ratio of gains vs losses)."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return None

    gains = arr[arr > threshold] - threshold
    losses = threshold - arr[arr < threshold]
    sum_losses = float(np.sum(losses))
    if sum_losses < 1e-12:
        return None

    sum_gains = float(np.sum(gains))
    return float(round(sum_gains / sum_losses, 4))


def compute_comprehensive_risk_ratios(
    symbol: str,
    returns: Union[np.ndarray, Sequence[float]],
    prices: Optional[Union[np.ndarray, Sequence[float]]] = None,
    risk_free_rate: float = 0.045,
    periods_per_year: float = 252.0,
) -> RiskRatioResult:
    """Compute complete package of risk-adjusted ratios for a return stream."""
    arr = np.asarray(returns, dtype=np.float64)
    n = len(arr)

    if n < 2:
        return RiskRatioResult(
            symbol=symbol,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            downside_deviation=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            risk_free_rate=risk_free_rate,
            sample_size=n,
        )

    sharpe = calculate_sharpe_ratio(
        arr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    sortino = calculate_sortino_ratio(
        arr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    downside_dev = calculate_downside_deviation(arr, target_return=0.0)
    ann_ret = float(round(float(np.mean(arr) * periods_per_year), 6))
    ann_vol = float(round(float(np.std(arr, ddof=1) * math.sqrt(periods_per_year)), 6))
    omega = calculate_omega_ratio(arr)

    calmar: Optional[float] = None
    if prices is not None:
        p_arr = np.asarray(prices, dtype=np.float64)
        if len(p_arr) > 1:
            mdd, _ = max_drawdown(p_arr)
            calmar = calculate_calmar_ratio(
                arr, max_drawdown_pct=mdd, periods_per_year=periods_per_year
            )

    return RiskRatioResult(
        symbol=symbol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        downside_deviation=round(downside_dev, 6),
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        risk_free_rate=risk_free_rate,
        sample_size=n,
    )
