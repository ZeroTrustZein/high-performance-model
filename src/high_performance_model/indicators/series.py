"""Vectorized technical indicators implemented with NumPy."""

from __future__ import annotations

from typing import Dict, Tuple, Union

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
