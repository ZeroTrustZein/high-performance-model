"""High-performance vectorized Monte Carlo simulation and stress testing engine."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from high_performance_model.risk.models import MonteCarloConfig, MonteCarloStressResult


class MonteCarloStressTester:
    """Vectorized Monte Carlo engine for simulating price paths and stress testing portfolios."""

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self.config = config or MonteCarloConfig()

    def simulate_paths(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        horizon_days: Optional[int] = None,
        n_simulations: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate geometric Brownian motion paths (shape: [n_simulations, total_steps + 1]).

        Args:
            initial_price: Starting asset price (S0).
            drift: Annualized expected return (mu).
            volatility: Annualized return volatility (sigma).
            horizon_days: Number of calendar/trading days to simulate.
            n_simulations: Total number of Monte Carlo paths.

        Returns:
            2D numpy array of simulated price trajectories.
        """
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")

        days = horizon_days or self.config.horizon_days
        n_sims = n_simulations or self.config.n_simulations
        steps_per_day = self.config.time_steps_per_day
        total_steps = days * steps_per_day

        dt = 1.0 / (252.0 * steps_per_day)

        rng = np.random.default_rng(self.config.random_seed)

        # Vectorized standard normal increments
        z = rng.standard_normal((n_sims, total_steps))

        # Geometric Brownian Motion drift and diffusion components
        mu = drift
        sigma = max(volatility, 1e-4)

        drift_step = (mu - 0.5 * (sigma**2)) * dt
        diffusion_step = sigma * math.sqrt(dt) * z

        log_returns = drift_step + diffusion_step

        # Optional Merton Jump Diffusion
        if self.config.jump_diffusion and self.config.jump_intensity > 0:
            poisson_lambda = self.config.jump_intensity * dt
            jumps = rng.poisson(poisson_lambda, size=(n_sims, total_steps))
            jump_sizes = rng.normal(
                self.config.jump_mean, self.config.jump_std, size=(n_sims, total_steps)
            )
            log_returns += jumps * jump_sizes

        # Accumulate cumulative log returns
        cum_returns = np.zeros((n_sims, total_steps + 1), dtype=np.float64)
        cum_returns[:, 1:] = np.cumsum(log_returns, axis=1)

        paths = initial_price * np.exp(cum_returns)
        return paths

    def run_stress_test(
        self,
        symbol: str,
        initial_price: float,
        drift: float,
        volatility: float,
        horizon_days: Optional[int] = None,
        n_simulations: Optional[int] = None,
    ) -> MonteCarloStressResult:
        """Run full Monte Carlo stress testing workflow and produce structured analytics."""
        days = horizon_days or self.config.horizon_days
        n_sims = n_simulations or self.config.n_simulations

        paths = self.simulate_paths(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            horizon_days=days,
            n_simulations=n_sims,
        )

        final_prices = paths[:, -1]
        returns_to_horizon = (final_prices - initial_price) / initial_price

        expected_final = float(round(float(np.mean(final_prices)), 4))
        median_final = float(round(float(np.median(final_prices)), 4))
        prob_loss = float(round(float(np.mean(returns_to_horizon < 0.0)), 4))

        # Compute max drawdown across each trajectory
        cum_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - cum_max) / cum_max
        worst_dd = float(round(float(np.min(drawdowns)), 4))

        # Value at Risk and Expected Shortfall per confidence level
        var_by_conf: Dict[str, float] = {}
        cvar_by_conf: Dict[str, float] = {}

        if self.config.confidence_levels:
            alphas = [(1.0 - cl) * 100.0 for cl in self.config.confidence_levels]
            var_values = np.percentile(returns_to_horizon, alphas)
            for cl, var_ret in zip(self.config.confidence_levels, var_values):
                tail_losses = returns_to_horizon[returns_to_horizon <= var_ret]
                cvar_ret = float(np.mean(tail_losses)) if len(tail_losses) > 0 else float(var_ret)
                key = f"{int(cl * 100)}%"
                var_by_conf[key] = float(round(float(var_ret), 4))
                cvar_by_conf[key] = float(round(cvar_ret, 4))

        # Key percentile trajectories for visualization
        percentile_ranks = [5, 25, 50, 75, 95]
        trajectories = np.percentile(paths, percentile_ranks, axis=0)
        percentiles_dict: Dict[str, List[float]] = {
            f"p{p}": [float(round(val, 2)) for val in traj]
            for p, traj in zip(percentile_ranks, trajectories)
        }

        return MonteCarloStressResult(
            symbol=symbol,
            initial_price=initial_price,
            horizon_days=days,
            n_simulations=n_sims,
            expected_final_price=expected_final,
            median_final_price=median_final,
            worst_case_drawdown=worst_dd,
            probability_of_loss=prob_loss,
            var_by_confidence=var_by_conf,
            cvar_by_confidence=cvar_by_conf,
            percentile_trajectories=percentiles_dict,
        )
