"""Comprehensive unit tests for vectorized technical indicators and quantitative risk analytics."""

from __future__ import annotations

import numpy as np
import pytest

from high_performance_model.indicators.series import (
    average_true_range,
    bollinger_bands,
    expected_shortfall,
    exponential_moving_average,
    macd,
    max_drawdown,
    momentum,
    rate_of_change,
    realized_volatility,
    relative_strength_index,
    rolling_z_score,
    sharpe_ratio,
    simple_moving_average,
    sortino_ratio,
    stochastic_oscillator,
    value_at_risk,
    volume_weighted_average_price,
)


@pytest.fixture
def trending_prices() -> np.ndarray:
    """Generate linear uptrending price series."""
    return np.linspace(100.0, 200.0, 100)


@pytest.fixture
def cyclical_prices() -> np.ndarray:
    """Generate sinusoidal oscillating price series."""
    x = np.linspace(0, 4 * np.pi, 120)
    return 100.0 + 10.0 * np.sin(x)


class TestMovingAverages:
    """Test Simple and Exponential Moving Average algorithms."""

    def test_sma_basic_and_padding(self, trending_prices: np.ndarray) -> None:
        period = 10
        sma = simple_moving_average(trending_prices, period=period)
        assert len(sma) == len(trending_prices)
        assert np.all(np.isnan(sma[: period - 1]))
        assert not np.isnan(sma[period - 1])
        # On linear sequence 100..200, SMA at index 9 is mean of 0..9
        expected = np.mean(trending_prices[:period])
        assert pytest.approx(sma[period - 1], 0.001) == expected

    def test_sma_edge_cases(self) -> None:
        data = [10.0, 20.0, 30.0]
        # Period larger than data length
        sma_large = simple_moving_average(data, period=10)
        assert np.all(np.isnan(sma_large))
        # Period <= 0
        sma_zero = simple_moving_average(data, period=0)
        assert np.all(np.isnan(sma_zero))
        # Empty data
        sma_empty = simple_moving_average([], period=5)
        assert len(sma_empty) == 0

    def test_ema_smoothing(self, trending_prices: np.ndarray) -> None:
        ema = exponential_moving_average(trending_prices, period=10)
        assert len(ema) == len(trending_prices)
        assert not np.isnan(ema[0])
        assert ema[0] == trending_prices[0]
        # Custom alpha
        ema_custom = exponential_moving_average(trending_prices, period=10, alpha=0.5)
        assert len(ema_custom) == len(trending_prices)

    def test_ema_edge_cases(self) -> None:
        assert len(exponential_moving_average([], period=10)) == 0
        zero_period = exponential_moving_average([1.0, 2.0], period=0)
        assert np.all(np.isnan(zero_period))


class TestOscillatorsAndBands:
    """Test RSI, MACD, Bollinger Bands, and Stochastic Oscillators."""

    def test_rsi_all_gains(self) -> None:
        prices = np.linspace(10.0, 100.0, 30)
        rsi = relative_strength_index(prices, period=14)
        # All gains -> RSI should be 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0
        assert np.all(valid_rsi == 100.0)

    def test_rsi_all_losses(self) -> None:
        prices = np.linspace(100.0, 10.0, 30)
        rsi = relative_strength_index(prices, period=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0
        assert np.all(valid_rsi == 0.0)

    def test_rsi_cyclical_bounded(self, cyclical_prices: np.ndarray) -> None:
        rsi = relative_strength_index(cyclical_prices, period=14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0
        assert np.all((valid_rsi >= 0.0) & (valid_rsi <= 100.0))

    def test_macd_relationship(self, cyclical_prices: np.ndarray) -> None:
        macd_line, signal_line, hist = macd(
            cyclical_prices, fast_period=12, slow_period=26, signal_period=9
        )
        assert len(macd_line) == len(cyclical_prices)
        assert len(signal_line) == len(cyclical_prices)
        assert len(hist) == len(cyclical_prices)
        # Histogram must equal macd_line - signal_line
        np.testing.assert_allclose(hist, macd_line - signal_line, rtol=1e-5, atol=1e-5)

    def test_bollinger_bands_invariants(self, cyclical_prices: np.ndarray) -> None:
        bb = bollinger_bands(cyclical_prices, period=20, num_std=2.0)
        valid_idx = ~np.isnan(bb["upper"])
        assert np.all(bb["upper"][valid_idx] >= bb["middle"][valid_idx])
        assert np.all(bb["middle"][valid_idx] >= bb["lower"][valid_idx])
        # Bandwidth check
        expected_bw = (bb["upper"][valid_idx] - bb["lower"][valid_idx]) / bb["middle"][valid_idx]
        np.testing.assert_allclose(bb["bandwidth"][valid_idx], expected_bw, rtol=1e-5)

    def test_bollinger_bands_insufficient_data(self) -> None:
        bb = bollinger_bands([10.0, 20.0], period=10)
        assert np.all(np.isnan(bb["upper"]))
        assert np.all(np.isnan(bb["lower"]))

    def test_stochastic_oscillator(self, cyclical_prices: np.ndarray) -> None:
        highs = cyclical_prices + 2.0
        lows = cyclical_prices - 2.0
        k, d = stochastic_oscillator(
            highs, lows, cyclical_prices, k_period=14, d_period=3, smooth_k=3
        )
        assert len(k) == len(cyclical_prices)
        assert len(d) == len(cyclical_prices)

        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        assert len(valid_k) > 0
        assert len(valid_d) > 0
        assert np.all((valid_k >= 0.0) & (valid_k <= 100.0))
        assert np.all((valid_d >= 0.0) & (valid_d <= 100.0))

    def test_stochastic_oscillator_flat_market(self) -> None:
        flat = np.full(30, 100.0)
        k, d = stochastic_oscillator(flat, flat, flat, k_period=14, d_period=3)
        valid_k = k[~np.isnan(k)]
        assert np.all(valid_k == 50.0)


class TestMomentumAndVolatility:
    """Test Momentum, Rate of Change, ATR, VWAP, and Z-Score."""

    def test_momentum(self) -> None:
        data = [10.0, 12.0, 15.0, 18.0, 22.0]
        mom = momentum(data, period=2)
        assert np.isnan(mom[0]) and np.isnan(mom[1])
        assert mom[2] == 15.0 - 10.0
        assert mom[3] == 18.0 - 12.0
        assert mom[4] == 22.0 - 15.0

    def test_momentum_edge_cases(self) -> None:
        assert np.all(np.isnan(momentum([1.0, 2.0], period=5)))
        assert np.all(np.isnan(momentum([1.0, 2.0], period=0)))

    def test_rate_of_change(self) -> None:
        data = [100.0, 110.0, 120.0, 130.0]
        roc = rate_of_change(data, period=1)
        assert np.isnan(roc[0])
        assert pytest.approx(roc[1], 0.001) == 10.0
        assert pytest.approx(roc[2], 0.001) == (120.0 - 110.0) / 110.0 * 100.0

    def test_rate_of_change_zero_division(self) -> None:
        data = [0.0, 10.0, 20.0]
        roc = rate_of_change(data, period=1)
        assert roc[1] == 0.0  # Division by 0 fallback

    def test_average_true_range(self) -> None:
        highs = np.array([105.0, 108.0, 110.0, 107.0, 109.0])
        lows = np.array([95.0, 97.0, 99.0, 96.0, 98.0])
        closes = np.array([100.0, 102.0, 105.0, 101.0, 104.0])
        atr = average_true_range(highs, lows, closes, period=3)
        assert len(atr) == len(closes)
        assert not np.isnan(atr[-1])
        assert atr[-1] > 0.0

    def test_atr_insufficient_length(self) -> None:
        atr = average_true_range([100.0], [90.0], [95.0], period=5)
        assert np.all(np.isnan(atr))

    def test_volume_weighted_average_price(self) -> None:
        prices = [100.0, 102.0, 104.0]
        volumes = [10.0, 20.0, 10.0]
        # (100*10 + 102*20 + 104*10) / 40 = (1000 + 2040 + 1040) / 40 = 4080 / 40 = 102.0
        vwap = volume_weighted_average_price(prices, volumes)
        assert vwap == 102.0

    def test_vwap_zero_volume(self) -> None:
        assert volume_weighted_average_price([100.0], [0.0]) == 100.0
        assert volume_weighted_average_price([], []) == 0.0

    def test_rolling_z_score(self) -> None:
        np.random.seed(42)
        noise = np.random.normal(0, 1, 100)
        z = rolling_z_score(noise, period=20)
        assert len(z) == 100
        assert np.all(np.isnan(z[:19]))
        valid_z = z[19:]
        # Standardized z-score on normal data should mostly be between -3 and 3
        assert np.all(np.abs(valid_z) < 5.0)

    def test_rolling_z_score_flat(self) -> None:
        flat = np.full(30, 50.0)
        z = rolling_z_score(flat, period=10)
        valid_z = z[9:]
        assert np.all(valid_z == 0.0)


class TestQuantitativeRiskAnalytics:
    """Test Realized Volatility, Sharpe, Sortino, Drawdowns, and VaR/CVaR."""

    def test_realized_volatility(self) -> None:
        prices = [100.0, 100.0, 100.0, 100.0]
        assert realized_volatility(prices) == 0.0
        assert realized_volatility([100.0]) == 0.0

        np.random.seed(42)
        drift = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        vol = realized_volatility(drift, annualize=True)
        assert vol > 0.0

    def test_sharpe_ratio(self) -> None:
        # Zero returns
        assert sharpe_ratio([0.0, 0.0, 0.0]) == 0.0
        # Positive returns
        returns = np.array([0.01, 0.015, 0.02, 0.005, 0.012])
        sr = sharpe_ratio(returns, risk_free_rate=0.0, annualize=False)
        assert sr > 0.0

    def test_sortino_ratio(self) -> None:
        # Strictly positive returns -> no downside deviation -> 0.0
        positive_returns = np.array([0.01, 0.02, 0.03, 0.015])
        assert sortino_ratio(positive_returns, annualize=False) == 0.0

        # Mixed returns with downside
        mixed = np.array([0.02, -0.01, 0.03, -0.015, 0.01])
        sortino = sortino_ratio(mixed, risk_free_rate=0.0, annualize=False)
        assert isinstance(sortino, float)

    def test_max_drawdown(self) -> None:
        # Upward only
        mdd, series = max_drawdown([100.0, 105.0, 110.0, 115.0])
        assert mdd == 0.0
        assert np.all(series == 0.0)

        # 50% drop
        mdd, series = max_drawdown([100.0, 150.0, 75.0, 120.0])
        assert pytest.approx(mdd, 0.001) == -0.5
        assert pytest.approx(series[2], 0.001) == -0.5

        # Empty data
        mdd_empty, s_empty = max_drawdown([])
        assert mdd_empty == 0.0
        assert len(s_empty) == 0

    def test_value_at_risk_and_expected_shortfall(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 500)

        var_hist = value_at_risk(returns, confidence_level=0.95, method="historical")
        var_param = value_at_risk(returns, confidence_level=0.95, method="parametric")
        assert var_hist < 0.0
        assert var_param < 0.0

        cvar = expected_shortfall(returns, confidence_level=0.95)
        # Expected shortfall (loss magnitude in tail) must be <= VaR threshold
        assert cvar <= var_hist

        with pytest.raises(ValueError, match="Unsupported VaR method"):
            value_at_risk(returns, method="monte_carlo")

    def test_var_cvar_empty(self) -> None:
        assert value_at_risk([], confidence_level=0.95) == 0.0
        assert expected_shortfall([], confidence_level=0.95) == 0.0
