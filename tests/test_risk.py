"""Comprehensive unit and integration test suite for quantitative risk and stress testing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from high_performance_model.risk.metrics import (
    calculate_calmar_ratio,
    calculate_downside_deviation,
    calculate_omega_ratio,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    compute_comprehensive_risk_ratios,
)
from high_performance_model.risk.models import (
    MonteCarloConfig,
    MonteCarloStressResult,
    RiskLevel,
    RiskTelemetrySnapshot,
    StressScenario,
)
from high_performance_model.risk.monte_carlo import MonteCarloStressTester
from high_performance_model.risk.stress import StressTestingSuite
from high_performance_model.telemetry.generator import SyntheticMarketFeed

# =====================================================================
# 1. Quantitative Risk Metrics Tests
# =====================================================================


class TestRiskMetricsSuite:
    """Test suite covering downside deviation, Sharpe, Sortino, Calmar, and Omega ratios."""

    def test_downside_deviation_empty(self) -> None:
        assert calculate_downside_deviation([]) == 0.0
        assert calculate_downside_deviation(np.array([], dtype=np.float64)) == 0.0

    def test_downside_deviation_all_gains(self) -> None:
        # All returns strictly above target return (0.0)
        returns = [0.01, 0.02, 0.03, 0.015]
        assert calculate_downside_deviation(returns, target_return=0.0) == 0.0

    def test_downside_deviation_mixed_returns(self) -> None:
        returns = [-0.02, 0.01, -0.04, 0.03, -0.01]
        target = 0.0
        # downside returns below target: -0.02, -0.04, -0.01
        # squared downside: 0.0004, 0.0016, 0.0001
        # mean across all 5 returns: (0.0004 + 0 + 0.0016 + 0 + 0.0001) / 5 = 0.00042
        # sqrt(0.00042) ~= 0.0204939...
        expected = round(math.sqrt((0.0004 + 0.0016 + 0.0001) / 5.0), 6)
        actual = calculate_downside_deviation(returns, target_return=target)
        assert pytest.approx(actual, rel=1e-5) == expected

    def test_downside_deviation_custom_target(self) -> None:
        returns = [0.01, 0.02, 0.03]
        # target return 0.02: returns below target are 0.01 (diff -0.01)
        # squared: (-0.01)^2 = 0.0001; mean = 0.0001 / 3; sqrt ~= 0.005774
        actual = calculate_downside_deviation(returns, target_return=0.02)
        expected = round(math.sqrt((0.01 - 0.02) ** 2 / 3.0), 6)
        assert pytest.approx(actual, rel=1e-5) == expected

    def test_sharpe_ratio_insufficient_samples(self) -> None:
        assert calculate_sharpe_ratio([]) == 0.0
        assert calculate_sharpe_ratio([0.05]) == 0.0

    def test_sharpe_ratio_zero_volatility(self) -> None:
        # Standard deviation is zero
        assert calculate_sharpe_ratio([0.02, 0.02, 0.02, 0.02]) == 0.0

    def test_sharpe_ratio_annualized_and_period(self) -> None:
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008]
        rf = 0.045
        periods = 252.0

        # Annualized
        ann_sharpe = calculate_sharpe_ratio(
            returns, risk_free_rate=rf, periods_per_year=periods, annualize=True
        )
        assert isinstance(ann_sharpe, float)
        assert ann_sharpe != 0.0

        # Non-annualized
        period_sharpe = calculate_sharpe_ratio(
            returns, risk_free_rate=rf, periods_per_year=periods, annualize=False
        )
        assert isinstance(period_sharpe, float)
        assert period_sharpe != 0.0
        assert ann_sharpe != period_sharpe

    def test_sharpe_ratio_custom_risk_free_rate(self) -> None:
        returns = [0.01, 0.02, 0.015, 0.012]
        sharpe_zero_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_high_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.10)
        assert sharpe_zero_rf > sharpe_high_rf

    def test_sortino_ratio_insufficient_samples(self) -> None:
        assert calculate_sortino_ratio([]) == 0.0
        assert calculate_sortino_ratio([0.05]) == 0.0

    def test_sortino_ratio_zero_downside(self) -> None:
        # All positive returns well above risk-free rate
        assert (
            calculate_sortino_ratio([0.08, 0.09, 0.10], risk_free_rate=0.0, target_return=0.0)
            == 0.0
        )

    def test_sortino_ratio_annualized_and_period(self) -> None:
        returns = [0.02, -0.01, 0.015, -0.02, 0.03, -0.005]
        ann_sortino = calculate_sortino_ratio(returns, annualize=True)
        period_sortino = calculate_sortino_ratio(returns, annualize=False)
        assert isinstance(ann_sortino, float)
        assert isinstance(period_sortino, float)
        assert ann_sortino != period_sortino

    def test_sortino_ratio_custom_target(self) -> None:
        returns = [0.01, 0.02, -0.01, 0.015]
        s1 = calculate_sortino_ratio(returns, target_return=0.0)
        s2 = calculate_sortino_ratio(returns, target_return=0.02)
        assert s1 != s2

    def test_calmar_ratio_insufficient_samples(self) -> None:
        assert calculate_calmar_ratio([], max_drawdown_pct=-0.1) is None
        assert calculate_calmar_ratio([0.05], max_drawdown_pct=-0.1) is None

    def test_calmar_ratio_zero_or_tiny_drawdown(self) -> None:
        returns = [0.01, 0.02, 0.01]
        assert calculate_calmar_ratio(returns, max_drawdown_pct=0.0) is None
        assert calculate_calmar_ratio(returns, max_drawdown_pct=1e-8) is None

    def test_calmar_ratio_valid(self) -> None:
        returns = [0.001, 0.002, 0.001, 0.0015]
        mdd = -0.15
        calmar = calculate_calmar_ratio(returns, max_drawdown_pct=mdd, periods_per_year=252.0)
        assert calmar is not None
        ann_return = float(np.mean(returns) * 252.0)
        expected = round(ann_return / abs(mdd), 4)
        assert calmar == expected

        # Positive mdd input should yield identical result due to abs()
        calmar_pos = calculate_calmar_ratio(returns, max_drawdown_pct=0.15, periods_per_year=252.0)
        assert calmar_pos == calmar

    def test_omega_ratio_empty(self) -> None:
        assert calculate_omega_ratio([]) is None
        assert calculate_omega_ratio(np.array([])) is None

    def test_omega_ratio_no_losses(self) -> None:
        # All returns > 0 -> sum of losses is 0 -> None
        assert calculate_omega_ratio([0.01, 0.02, 0.03], threshold=0.0) is None

    def test_omega_ratio_valid(self) -> None:
        returns = [0.02, -0.01, 0.04, -0.02]
        # Gains: 0.02 + 0.04 = 0.06
        # Losses: 0.01 + 0.02 = 0.03
        # Omega: 0.06 / 0.03 = 2.0
        omega = calculate_omega_ratio(returns, threshold=0.0)
        assert omega == 2.0

    def test_omega_ratio_custom_threshold(self) -> None:
        returns = [0.03, 0.01, -0.01, 0.02]
        omega = calculate_omega_ratio(returns, threshold=0.015)
        assert omega is not None
        assert isinstance(omega, float)

    def test_compute_comprehensive_risk_ratios_small_sample(self) -> None:
        res_empty = compute_comprehensive_risk_ratios("AAPL", [])
        assert res_empty.symbol == "AAPL"
        assert res_empty.sharpe_ratio == 0.0
        assert res_empty.sortino_ratio == 0.0
        assert res_empty.sample_size == 0

        res_single = compute_comprehensive_risk_ratios("AAPL", [0.02])
        assert res_single.symbol == "AAPL"
        assert res_single.sharpe_ratio == 0.0
        assert res_single.sortino_ratio == 0.0
        assert res_single.sample_size == 1

    def test_compute_comprehensive_risk_ratios_without_prices(self) -> None:
        returns = [0.01, -0.005, 0.015, -0.008, 0.012, -0.002]
        res = compute_comprehensive_risk_ratios("MSFT", returns)
        assert res.symbol == "MSFT"
        assert res.sample_size == 6
        assert res.sharpe_ratio != 0.0
        assert res.sortino_ratio != 0.0
        assert res.downside_deviation > 0.0
        assert res.annualized_return != 0.0
        assert res.annualized_volatility > 0.0
        assert res.calmar_ratio is None  # No prices passed

    def test_compute_comprehensive_risk_ratios_with_prices(self) -> None:
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        prices = [100.0, 101.0, 98.98, 101.95, 100.93, 102.95]
        res = compute_comprehensive_risk_ratios("NVDA", returns, prices=prices)
        assert res.symbol == "NVDA"
        assert res.calmar_ratio is not None
        assert isinstance(res.calmar_ratio, float)
        assert res.omega_ratio is not None

    def test_compute_comprehensive_risk_ratios_single_price(self) -> None:
        returns = [0.01, -0.02, 0.03]
        prices = [100.0]  # Only 1 price
        res = compute_comprehensive_risk_ratios("GOOGL", returns, prices=prices)
        assert res.calmar_ratio is None  # len(prices) <= 1, so calmar is None

    def test_compute_comprehensive_risk_ratios_custom_parameters(self) -> None:
        returns = [0.005, -0.002, 0.008, -0.003]
        res = compute_comprehensive_risk_ratios(
            "TSLA", returns, risk_free_rate=0.03, periods_per_year=365.0
        )
        assert res.symbol == "TSLA"
        assert res.risk_free_rate == 0.03


# =====================================================================
# 2. Monte Carlo Stress Tester Engine Tests
# =====================================================================


class TestMonteCarloStressTester:
    """Test suite covering geometric Brownian motion and Merton jump diffusion simulations."""

    def test_init_defaults_and_custom_config(self) -> None:
        tester_default = MonteCarloStressTester()
        assert tester_default.config.n_simulations == 5000
        assert tester_default.config.horizon_days == 30

        custom_cfg = MonteCarloConfig(n_simulations=100, horizon_days=10, random_seed=123)
        tester_custom = MonteCarloStressTester(config=custom_cfg)
        assert tester_custom.config.n_simulations == 100
        assert tester_custom.config.horizon_days == 10
        assert tester_custom.config.random_seed == 123

    def test_simulate_paths_invalid_initial_price(self) -> None:
        tester = MonteCarloStressTester()
        with pytest.raises(ValueError, match="initial_price must be positive"):
            tester.simulate_paths(initial_price=0.0, drift=0.05, volatility=0.2)

        with pytest.raises(ValueError, match="initial_price must be positive"):
            tester.simulate_paths(initial_price=-50.0, drift=0.05, volatility=0.2)

    def test_simulate_paths_shape_and_initial_price(self) -> None:
        cfg = MonteCarloConfig(
            n_simulations=200, horizon_days=15, time_steps_per_day=2, random_seed=42
        )
        tester = MonteCarloStressTester(config=cfg)

        s0 = 150.0
        paths = tester.simulate_paths(initial_price=s0, drift=0.08, volatility=0.25)
        # Shape: (n_simulations, horizon_days * time_steps_per_day + 1)
        expected_steps = 15 * 2 + 1
        assert paths.shape == (200, expected_steps)
        # All paths start at initial price
        assert np.all(paths[:, 0] == s0)
        # Prices remain positive
        assert np.all(paths > 0.0)

    def test_simulate_paths_reproducibility_with_seed(self) -> None:
        cfg1 = MonteCarloConfig(n_simulations=50, horizon_days=5, random_seed=999)
        cfg2 = MonteCarloConfig(n_simulations=50, horizon_days=5, random_seed=999)
        tester1 = MonteCarloStressTester(cfg1)
        tester2 = MonteCarloStressTester(cfg2)

        paths1 = tester1.simulate_paths(100.0, 0.05, 0.20)
        paths2 = tester2.simulate_paths(100.0, 0.05, 0.20)
        np.testing.assert_allclose(paths1, paths2)

    def test_simulate_paths_jump_diffusion(self) -> None:
        cfg = MonteCarloConfig(
            n_simulations=300,
            horizon_days=20,
            jump_diffusion=True,
            jump_intensity=1.5,
            jump_mean=-0.05,
            jump_std=0.10,
            random_seed=42,
        )
        tester = MonteCarloStressTester(config=cfg)
        paths = tester.simulate_paths(initial_price=100.0, drift=0.05, volatility=0.15)
        assert paths.shape == (300, 21)
        assert np.all(paths > 0.0)

    def test_simulate_paths_custom_horizon_and_simulations(self) -> None:
        cfg = MonteCarloConfig(n_simulations=50, horizon_days=10, random_seed=42)
        tester = MonteCarloStressTester(config=cfg)

        # Override default horizon and n_simulations in method call
        paths = tester.simulate_paths(100.0, 0.05, 0.20, horizon_days=5, n_simulations=25)
        assert paths.shape == (25, 6)

    def test_run_stress_test_metrics(self) -> None:
        cfg = MonteCarloConfig(n_simulations=500, horizon_days=20, random_seed=77)
        tester = MonteCarloStressTester(config=cfg)

        res = tester.run_stress_test(
            symbol="BTC/USD",
            initial_price=50000.0,
            drift=0.10,
            volatility=0.60,
        )

        assert isinstance(res, MonteCarloStressResult)
        assert res.symbol == "BTC/USD"
        assert res.initial_price == 50000.0
        assert res.horizon_days == 20
        assert res.n_simulations == 500
        assert res.expected_final_price > 0.0
        assert res.median_final_price > 0.0
        assert res.worst_case_drawdown <= 0.0
        assert 0.0 <= res.probability_of_loss <= 1.0

    def test_run_stress_test_var_cvar_consistency(self) -> None:
        cfg = MonteCarloConfig(
            n_simulations=1000,
            horizon_days=30,
            confidence_levels=[0.90, 0.95, 0.99],
            random_seed=101,
        )
        tester = MonteCarloStressTester(config=cfg)
        res = tester.run_stress_test("ETH/USD", initial_price=3000.0, drift=0.05, volatility=0.50)

        assert "90%" in res.var_by_confidence
        assert "95%" in res.var_by_confidence
        assert "99%" in res.var_by_confidence

        # 99% VaR should represent a worse (more negative or lower) return than 90% VaR
        assert res.var_by_confidence["99%"] <= res.var_by_confidence["90%"]

        # CVaR (expected tail loss) must be <= VaR (more severe loss)
        for cl in ["90%", "95%", "99%"]:
            assert res.cvar_by_confidence[cl] <= res.var_by_confidence[cl]

    def test_run_stress_test_percentile_trajectories(self) -> None:
        cfg = MonteCarloConfig(n_simulations=200, horizon_days=10, random_seed=42)
        tester = MonteCarloStressTester(config=cfg)
        res = tester.run_stress_test("SPY", initial_price=500.0, drift=0.08, volatility=0.15)

        for p in ["p5", "p25", "p50", "p75", "p95"]:
            assert p in res.percentile_trajectories
            traj = res.percentile_trajectories[p]
            assert len(traj) == 11
            assert traj[0] == 500.0

        # Monotonicity check across percentiles at horizon end
        assert res.percentile_trajectories["p5"][-1] <= res.percentile_trajectories["p50"][-1]
        assert res.percentile_trajectories["p50"][-1] <= res.percentile_trajectories["p95"][-1]


# =====================================================================
# 3. Stress Testing Suite Tests
# =====================================================================


class TestStressTestingSuite:
    """Test suite covering standard and custom market stress scenario evaluations."""

    def test_list_scenarios(self) -> None:
        scenarios = StressTestingSuite.list_scenarios()
        assert len(scenarios) == 5
        names = [s.name for s in scenarios]
        assert "2008 Financial Crisis" in names
        assert "COVID-19 Liquidity Shock" in names
        assert "Tech Flash Crash" in names
        assert "Stagflation & Rate Hike Shock" in names
        assert "Crypto Contagion Black Swan" in names

    def test_get_scenario_valid_and_invalid(self) -> None:
        sc = StressTestingSuite.get_scenario("2008_financial_crisis")
        assert sc is not None
        assert sc.name == "2008 Financial Crisis"
        assert sc.price_shock_pct == -0.35
        assert sc.volatility_multiplier == 2.8

        invalid = StressTestingSuite.get_scenario("non_existent_key")
        assert invalid is None

    def test_evaluate_scenario_invalid_initial_price(self) -> None:
        sc = StressTestingSuite.get_scenario("tech_flash_crash")
        assert sc is not None
        with pytest.raises(ValueError, match="initial_price must be positive"):
            StressTestingSuite.evaluate_scenario("AAPL", 0.0, 0.25, sc)
        with pytest.raises(ValueError, match="initial_price must be positive"):
            StressTestingSuite.evaluate_scenario("AAPL", -10.0, 0.25, sc)

    def test_evaluate_scenario_critical_risk(self) -> None:
        # Abs shock >= 0.30 or stressed_vol >= 0.80 -> CRITICAL
        crypto_sc = StressTestingSuite.get_scenario("crypto_black_swan")
        assert crypto_sc is not None
        res = StressTestingSuite.evaluate_scenario("BTC/USD", 60000.0, 0.50, crypto_sc)
        assert res.risk_level == RiskLevel.CRITICAL
        assert res.stressed_price == round(60000.0 * (1.0 - 0.45), 4)
        assert res.stressed_volatility == round(0.50 * 3.2, 4)

    def test_evaluate_scenario_high_risk(self) -> None:
        # Abs shock >= 0.20 or stressed_vol >= 0.50 -> HIGH
        covid_sc = StressTestingSuite.get_scenario("covid_liquidity_shock")
        assert covid_sc is not None
        res = StressTestingSuite.evaluate_scenario("SPY", 400.0, 0.15, covid_sc)
        assert res.risk_level == RiskLevel.HIGH
        assert res.stressed_price == round(400.0 * (1.0 - 0.25), 4)

    def test_evaluate_scenario_elevated_risk(self) -> None:
        # Abs shock >= 0.10 or stressed_vol >= 0.30 -> ELEVATED
        flash_sc = StressTestingSuite.get_scenario("tech_flash_crash")
        assert flash_sc is not None
        res = StressTestingSuite.evaluate_scenario("QQQ", 350.0, 0.12, flash_sc)
        assert res.risk_level == RiskLevel.ELEVATED
        assert res.stressed_price == round(350.0 * (1.0 - 0.12), 4)

    def test_evaluate_scenario_moderate_risk(self) -> None:
        # Minor scenario with small shock and small volatility
        mild_scenario = StressScenario(
            name="Mild Pullback",
            description="Routine healthy consolidation",
            price_shock_pct=-0.04,
            volatility_multiplier=1.1,
        )
        res = StressTestingSuite.evaluate_scenario("JNJ", 160.0, 0.10, mild_scenario)
        assert res.risk_level == RiskLevel.MODERATE
        assert res.stressed_price == round(160.0 * (1.0 - 0.04), 4)

    def test_evaluate_scenario_calculations(self) -> None:
        sc = StressScenario(
            name="Test Scenario",
            description="Testing calculations",
            price_shock_pct=-0.15,
            volatility_multiplier=2.0,
        )
        initial_price = 200.0
        current_vol = 0.20
        res = StressTestingSuite.evaluate_scenario("AAPL", initial_price, current_vol, sc)

        assert res.symbol == "AAPL"
        assert res.scenario_name == "Test Scenario"
        assert res.initial_price == 200.0
        assert res.stressed_price == 170.0
        assert res.pnl_shock_pct == -0.15
        assert res.stressed_volatility == 0.40
        # VaR = 1.64485 * (0.40 / sqrt(252))
        expected_var = round(1.64485 * (0.40 / math.sqrt(252.0)), 4)
        assert res.stressed_var_95 == expected_var
        assert res.risk_level == RiskLevel.ELEVATED


# =====================================================================
# 4. Risk Integration and End-to-End Pipeline Tests
# =====================================================================


class TestRiskIntegrationAndPipeline:
    """Test full integration between synthetic feed, buffer, analytics, and telemetry snapshots."""

    def test_end_to_end_risk_workflow(self) -> None:
        feed = SyntheticMarketFeed(symbols=["AAPL"], seed=42)
        ticks = feed.generate_history("AAPL", n_points=120)
        prices = [t.price for t in ticks]
        returns = np.diff(prices) / prices[:-1]

        # 1. Compute comprehensive risk ratios
        ratios = compute_comprehensive_risk_ratios(
            symbol="AAPL",
            returns=returns,
            prices=prices,
            risk_free_rate=0.045,
        )
        assert ratios.symbol == "AAPL"
        assert ratios.sample_size == len(returns)
        assert ratios.sharpe_ratio != 0.0
        assert ratios.sortino_ratio != 0.0
        assert ratios.calmar_ratio is not None

        # 2. Run Monte Carlo stress simulation
        mc_cfg = MonteCarloConfig(n_simulations=500, horizon_days=20, random_seed=42)
        tester = MonteCarloStressTester(config=mc_cfg)
        vol = float(np.std(returns, ddof=1) * math.sqrt(252.0))
        drift = float(np.mean(returns) * 252.0)
        mc_result = tester.run_stress_test(
            symbol="AAPL",
            initial_price=prices[-1],
            drift=drift,
            volatility=vol,
        )
        assert mc_result.symbol == "AAPL"
        assert mc_result.n_simulations == 500
        assert "95%" in mc_result.var_by_confidence

        # 3. Evaluate multi-scenario stress tests
        stress_results = []
        for scenario in StressTestingSuite.list_scenarios()[:3]:
            stress_res = StressTestingSuite.evaluate_scenario(
                symbol="AAPL",
                initial_price=prices[-1],
                current_volatility=vol,
                scenario=scenario,
            )
            stress_results.append(stress_res)

        assert len(stress_results) == 3

        # 4. Package into streaming RiskTelemetrySnapshot
        snapshot = RiskTelemetrySnapshot(
            symbol="AAPL",
            ratios=ratios,
            monte_carlo_var_95=mc_result.var_by_confidence.get("95%"),
            stress_impacts=stress_results,
            risk_level=RiskLevel.MODERATE,
        )
        assert snapshot.symbol == "AAPL"
        assert snapshot.monte_carlo_var_95 is not None
        assert len(snapshot.stress_impacts) == 3

        # 5. Dict serialization and round-trip
        data = snapshot.to_dict()
        assert data["symbol"] == "AAPL"
        assert "ratios" in data
        assert "monte_carlo_var_95" in data
        assert len(data["stress_impacts"]) == 3

        json_str = snapshot.model_dump_json()
        restored = RiskTelemetrySnapshot.model_validate_json(json_str)
        assert restored.symbol == "AAPL"
        assert restored.ratios.sharpe_ratio == ratios.sharpe_ratio
        assert len(restored.stress_impacts) == 3
