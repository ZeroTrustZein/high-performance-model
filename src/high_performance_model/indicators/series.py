"""Vectorized technical indicators and quantitative risk analytics implemented with NumPy."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np


def simple_moving_average(data: Union[np.ndarray, list[float]], period: int = 20) -> np.ndarray:
    """Calculate Simple Moving Average (SMA)."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < period or period <= 0:
        return np.full_like(arr, np.nan)

    weights = np.ones(period) / period
    sma = np.convolve(arr, weights, mode="valid")
    # Pad leading values with NaN
    padding = np.full(period - 1, np.nan)
    return np.concatenate([padding, sma])


def exponential_moving_average(
    data: Union[np.ndarray, list[float]], period: int = 20, alpha: Union[float, None] = None
) -> np.ndarray:
    """Calculate Exponential Moving Average (EMA) with optional custom smoothing alpha."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0 or period <= 0:
        return np.full_like(arr, np.nan)

    smoothing = alpha if alpha is not None else (2.0 / (period + 1.0))
    ema = np.empty(n, dtype=np.float64)
    ema[0] = arr[0]

    for i in range(1, n):
        ema[i] = smoothing * arr[i] + (1.0 - smoothing) * ema[i - 1]

    if n < period:
        return ema
    return ema


def relative_strength_index(data: Union[np.ndarray, list[float]], period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index (RSI)."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n <= period or period <= 0:
        return np.full_like(arr, np.nan)

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(n, np.nan)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def macd(
    data: Union[np.ndarray, list[float]],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD line, Signal line, and Histogram."""
    arr = np.asarray(data, dtype=np.float64)
    ema_fast = exponential_moving_average(arr, period=fast_period)
    ema_slow = exponential_moving_average(arr, period=slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, period=signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    data: Union[np.ndarray, list[float]],
    period: int = 20,
    num_std: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Calculate Bollinger Bands (upper, middle, lower, bandwidth)."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n < period or period <= 0:
        nan_arr = np.full_like(arr, np.nan)
        return {"upper": nan_arr, "middle": nan_arr, "lower": nan_arr, "bandwidth": nan_arr}

    sma = simple_moving_average(arr, period=period)
    rolling_std = np.full(n, np.nan)

    for i in range(period - 1, n):
        rolling_std[i] = np.std(arr[i - period + 1 : i + 1])

    upper = sma + (rolling_std * num_std)
    lower = sma - (rolling_std * num_std)
    bandwidth = np.where(sma != 0.0, (upper - lower) / sma, 0.0)

    return {
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "bandwidth": bandwidth,
    }


def average_true_range(
    high: Union[np.ndarray, list[float]],
    low: Union[np.ndarray, list[float]],
    close: Union[np.ndarray, list[float]],
    period: int = 14,
) -> np.ndarray:
    """Calculate Average True Range (ATR) volatility metric."""
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)

    n = len(close_arr)
    if n < 2:
        return np.full_like(close_arr, np.nan)

    prev_c = np.roll(close_arr, 1)
    prev_c[0] = close_arr[0]

    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - prev_c)
    tr3 = np.abs(low_arr - prev_c)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    return exponential_moving_average(true_range, period=period)


def volume_weighted_average_price(
    prices: Union[np.ndarray, list[float]],
    volumes: Union[np.ndarray, list[float]],
) -> float:
    """Calculate single-window VWAP."""
    p = np.asarray(prices, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    total_v = np.sum(v)
    if total_v <= 0:
        return float(p[-1]) if len(p) > 0 else 0.0
    return float(round(np.sum(p * v) / total_v, 4))


def momentum(data: Union[np.ndarray, list[float]], period: int = 10) -> np.ndarray:
    """Calculate absolute Price Momentum (P_t - P_{t-n})."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    res = np.full(n, np.nan, dtype=np.float64)
    if n <= period or period <= 0:
        return res
    res[period:] = arr[period:] - arr[:-period]
    return res


def rate_of_change(data: Union[np.ndarray, list[float]], period: int = 10) -> np.ndarray:
    """Calculate Rate of Change percentage ((P_t - P_{t-n}) / P_{t-n} * 100)."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    res = np.full(n, np.nan, dtype=np.float64)
    if n <= period or period <= 0:
        return res
    prev = arr[:-period]
    diff = arr[period:] - prev
    res[period:] = np.where(prev != 0.0, (diff / prev) * 100.0, 0.0)
    return res


def stochastic_oscillator(
    high: Union[np.ndarray, list[float]],
    low: Union[np.ndarray, list[float]],
    close: Union[np.ndarray, list[float]],
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Stochastic Oscillator (%K and %D lines)."""
    h_arr = np.asarray(high, dtype=np.float64)
    l_arr = np.asarray(low, dtype=np.float64)
    c_arr = np.asarray(close, dtype=np.float64)
    n = len(c_arr)

    k_line = np.full(n, np.nan, dtype=np.float64)
    d_line = np.full(n, np.nan, dtype=np.float64)

    if n < k_period or k_period <= 0 or d_period <= 0:
        return k_line, d_line

    raw_k = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        lowest_low = np.min(l_arr[i - k_period + 1 : i + 1])
        highest_high = np.max(h_arr[i - k_period + 1 : i + 1])
        rng = highest_high - lowest_low
        if rng > 0:
            raw_k[i] = ((c_arr[i] - lowest_low) / rng) * 100.0
        else:
            raw_k[i] = 50.0

    if smooth_k > 1:
        # Smooth %K
        for i in range(k_period + smooth_k - 2, n):
            window = raw_k[i - smooth_k + 1 : i + 1]
            if not np.any(np.isnan(window)):
                k_line[i] = np.mean(window)
    else:
        k_line = raw_k

    # Calculate %D as SMA of %K
    for i in range(n):
        if i >= d_period - 1:
            window = k_line[i - d_period + 1 : i + 1]
            if not np.any(np.isnan(window)):
                d_line[i] = np.mean(window)

    return k_line, d_line


def rolling_z_score(data: Union[np.ndarray, list[float]], period: int = 20) -> np.ndarray:
    """Calculate Rolling Z-Score ((X_t - Mean_t) / Std_t)."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    z = np.full(n, np.nan, dtype=np.float64)
    if n < period or period <= 1:
        return z

    sma = simple_moving_average(arr, period=period)
    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        std = np.std(window)
        if std > 1e-12:
            z[i] = (arr[i] - sma[i]) / std
        else:
            z[i] = 0.0
    return z


def realized_volatility(
    data: Union[np.ndarray, list[float]],
    period: Optional[int] = None,
    annualize: bool = True,
    trading_periods: float = 252.0 * 24.0 * 60.0,
) -> float:
    """Calculate realized volatility from price log returns."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    window = arr[-period:] if period is not None and period < len(arr) else arr
    if len(window) < 2:
        return 0.0

    log_returns = np.diff(np.log(window))
    vol = float(np.std(log_returns))
    if annualize:
        vol *= math.sqrt(trading_periods)
    return float(round(vol, 6))


def sharpe_ratio(
    returns: Union[np.ndarray, list[float]],
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: float = 252.0 * 24.0 * 60.0,
) -> float:
    """Calculate Sharpe Ratio for a return series."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    std = float(np.std(arr))
    if std < 1e-12:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year if annualize else risk_free_rate
    excess = arr - rf_per_period
    ratio = float(np.mean(excess) / std)
    if annualize:
        ratio *= math.sqrt(periods_per_year)
    return float(round(ratio, 4))


def sortino_ratio(
    returns: Union[np.ndarray, list[float]],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualize: bool = True,
    periods_per_year: float = 252.0 * 24.0 * 60.0,
) -> float:
    """Calculate Sortino Ratio focusing on downside risk deviation."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    target_per_period = target_return / periods_per_year if annualize else target_return
    downside = np.minimum(0.0, arr - target_per_period)
    downside_dev = float(math.sqrt(np.mean(downside**2)))
    if downside_dev < 1e-12:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year if annualize else risk_free_rate
    excess = float(np.mean(arr - rf_per_period))
    ratio = excess / downside_dev
    if annualize:
        ratio *= math.sqrt(periods_per_year)
    return float(round(ratio, 4))


def max_drawdown(data: Union[np.ndarray, list[float]]) -> Tuple[float, np.ndarray]:
    """Calculate Maximum Drawdown and full drawdown curve."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, np.empty(0, dtype=np.float64)

    running_max = np.maximum.accumulate(arr)
    drawdowns = np.where(running_max > 0.0, (arr - running_max) / running_max, 0.0)
    mdd = float(np.min(drawdowns))
    return float(round(mdd, 6)), drawdowns


def value_at_risk(
    returns: Union[np.ndarray, list[float]],
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk (VaR) at specified confidence level (e.g. 0.95)."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    alpha = 1.0 - confidence_level
    if method == "historical":
        var = float(np.percentile(arr, alpha * 100.0))
    elif method == "parametric":
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        # Standard normal inverse CDF approximation for common confidence levels
        z_dict = {0.90: 1.28155, 0.95: 1.64485, 0.99: 2.32635, 0.999: 3.09023}
        z = z_dict.get(round(confidence_level, 3), 1.64485)
        var = mean - (z * std)
    else:
        raise ValueError(f"Unsupported VaR method: {method}. Use 'historical' or 'parametric'.")

    return float(round(var, 6))


def expected_shortfall(
    returns: Union[np.ndarray, list[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Expected Shortfall (CVaR) - conditional expectation of loss beyond VaR."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    var = value_at_risk(arr, confidence_level=confidence_level, method="historical")
    beyond_var = arr[arr <= var]
    if len(beyond_var) == 0:
        return var
    return float(round(float(np.mean(beyond_var)), 6))
