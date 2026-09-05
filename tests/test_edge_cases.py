"""Comprehensive edge-case and boundary condition test suite.

Covers:
- Vectorized Indicators with empty arrays, division-by-zero, flat series, single items
- MarketDataBuffer boundary handling, empty states, ring-buffer overflow, volume profile POC
- ExecutionSimulator & Position reversals, short selling, stop-loss buy/sell, cancellations
- Historical Replay & Persistence corrupt files, unsupported formats, empty datasets
- AlertEngine edge cases, listener resilience, invalid rules
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from high_performance_model.alerts.engine import AlertEngine
from high_performance_model.alerts.models import AlertType
from high_performance_model.execution.engine import ExecutionSimulator
from high_performance_model.execution.models import OrderStatus, Position
from high_performance_model.indicators.series import (
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
)
from high_performance_model.storage.persistence import (
    load_bars_from_jsonl,
    load_buffer_snapshot,
    load_ticks_from_csv,
    load_ticks_from_jsonl,
    save_bars_to_jsonl,
    save_ticks_to_csv,
    save_ticks_to_jsonl,
)
from high_performance_model.storage.replay import HistoricalReplayEngine
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.types import (
    MarketTick,
    OrderBook,
    OrderBookLevel,
    OrderType,
    Side,
)


class TestIndicatorEdgeCases:
    """Mathematical boundary conditions, zero divisions, and small-sample guards."""

    def test_moving_averages_boundary(self) -> None:
        empty = np.array([], dtype=np.float64)
        single = np.array([100.0], dtype=np.float64)

        # SMA
        assert np.all(np.isnan(simple_moving_average(empty, period=10)))
        assert np.all(np.isnan(simple_moving_average(single, period=0)))
        assert np.isnan(simple_moving_average(single, period=5)[0])

        # EMA
        assert len(exponential_moving_average(empty, period=10)) == 0
        assert np.all(np.isnan(exponential_moving_average(single, period=-1)))
        # n < period should still calculate smoothed series
        res_short = exponential_moving_average(single, period=5, alpha=0.5)
        assert res_short[0] == 100.0

    def test_rsi_boundary(self) -> None:
        # Array length <= period
        short = np.array([10.0, 11.0, 12.0])
        assert np.all(np.isnan(relative_strength_index(short, period=14)))
        assert np.all(np.isnan(relative_strength_index(short, period=-1)))

        # Flat prices (zero gains and zero losses)
        flat = np.full(30, 100.0)
        rsi_flat = relative_strength_index(flat, period=14)
        assert rsi_flat[14] == 100.0  # avg_loss == 0

    def test_macd_boundary(self) -> None:
        short = np.array([10.0, 12.0, 14.0])
        m_line, s_line, hist = macd(short, fast_period=12, slow_period=26, signal_period=9)
        assert len(m_line) == 3
        assert len(s_line) == 3
        assert len(hist) == 3

    def test_bollinger_bands_boundary(self) -> None:
        short = np.array([100.0, 105.0])
        res = bollinger_bands(short, period=10)
        assert np.all(np.isnan(res["upper"]))
        assert np.all(np.isnan(res["middle"]))
        assert np.all(np.isnan(res["lower"]))

        # Period <= 1 or num_std <= 0
        res_invalid = bollinger_bands(short, period=0)
        assert np.all(np.isnan(res_invalid["middle"]))

    def test_stochastic_oscillator_boundary(self) -> None:
        high_arr = np.array([10.0, 12.0])
        low_arr = np.array([8.0, 9.0])
        close_arr = np.array([9.0, 11.0])

        # n < k_period
        k, d = stochastic_oscillator(high_arr, low_arr, close_arr, k_period=14)
        assert np.all(np.isnan(k))
        assert np.all(np.isnan(d))

        # smooth_k <= 1 branch
        highs = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
        lows = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        closes = np.array([9.0, 11.0, 14.0, 13.0, 15.0])
        k_nosmooth, _ = stochastic_oscillator(highs, lows, closes, k_period=3, smooth_k=1)
        assert not np.isnan(k_nosmooth[-1])

    def test_momentum_and_roc_boundary(self) -> None:
        data = np.array([10.0, 20.0, 30.0])
        assert np.all(np.isnan(momentum(data, period=5)))
        assert np.all(np.isnan(momentum(data, period=0)))

        assert np.all(np.isnan(rate_of_change(data, period=5)))
        assert np.all(np.isnan(rate_of_change(data, period=0)))

        # Zero denominator in ROC
        zero_prev = np.array([0.0, 10.0, 20.0, 30.0])
        roc_zero = rate_of_change(zero_prev, period=2)
        assert not np.isinf(roc_zero[2])

    def test_rolling_z_score_boundary(self) -> None:
        short = np.array([1.0, 2.0])
        assert np.all(np.isnan(rolling_z_score(short, period=10)))
        assert np.all(np.isnan(rolling_z_score(short, period=1)))

        # Flat values: std < 1e-12 returns 0.0
        flat = np.full(25, 50.0)
        z = rolling_z_score(flat, period=10)
        assert z[-1] == 0.0

    def test_risk_metrics_boundary(self) -> None:
        # realized_volatility
        assert realized_volatility([], period=10) == 0.0
        assert realized_volatility([100.0], period=10) == 0.0
        # annualize False
        vol_raw = realized_volatility([100.0, 101.0, 102.0], annualize=False)
        assert vol_raw >= 0.0

        # sharpe_ratio
        assert sharpe_ratio([]) == 0.0
        assert sharpe_ratio([0.01]) == 0.0
        # flat returns: std < 1e-12
        assert sharpe_ratio([0.05, 0.05, 0.05]) == 0.0
        assert sharpe_ratio([0.01, 0.02, 0.03], annualize=False) != 0.0

        # sortino_ratio
        assert sortino_ratio([]) == 0.0
        assert sortino_ratio([0.01]) == 0.0
        # only positive returns (downside deviation is zero)
        assert sortino_ratio([0.05, 0.06, 0.07]) == 0.0
        assert sortino_ratio([0.05, -0.02, 0.03], annualize=False) != 0.0

        # max_drawdown
        mdd_empty, curve = max_drawdown([])
        assert mdd_empty == 0.0 and len(curve) == 0
        # Monotonically increasing
        mdd_up, _ = max_drawdown([10.0, 20.0, 30.0, 40.0])
        assert mdd_up == 0.0

        # value_at_risk & expected_shortfall
        assert value_at_risk([]) == 0.0
        var_param = value_at_risk([-0.02, 0.01, -0.05, 0.03], method="parametric", confidence_level=0.90)
        assert isinstance(var_param, float)

        with pytest.raises(ValueError, match="Unsupported VaR method"):
            value_at_risk([-0.01, 0.02], method="invalid_method")

        assert expected_shortfall([]) == 0.0
        # Single return where beyond_var falls back to var
        es_single = expected_shortfall([-0.02])
        assert es_single == -0.02


class TestBufferEdgeCases:
    """Buffer capacity, invalid queries, empty states, and microstructural fallbacks."""

    def test_buffer_capacity_validation(self) -> None:
        with pytest.raises(ValueError, match="capacity must be at least 1"):
            MarketDataBuffer(capacity=0)

    def test_buffer_empty_symbol_queries(self) -> None:
        buf = MarketDataBuffer(capacity=100)
        assert buf.get_ticks("NONEXISTENT") == []
        assert buf.get_bars("NONEXISTENT") == []
        assert buf.get_latest_tick("NONEXISTENT") is None
        assert buf.get_latest_order_book("NONEXISTENT") is None
        assert len(buf.get_price_series("NONEXISTENT")) == 0
        assert len(buf.get_volume_series("NONEXISTENT")) == 0
        assert buf.compute_vwap("NONEXISTENT") is None
        assert buf.compute_realized_volatility("NONEXISTENT") is None
        assert buf.compute_top_quote("NONEXISTENT") is None
        assert buf.compute_market_depth_snapshot("NONEXISTENT") is None
        assert buf.compute_effective_spread("NONEXISTENT") is None

        ofi = buf.compute_order_flow_imbalance("NONEXISTENT")
        assert ofi["buy_volume"] == 0.0 and ofi["imbalance_ratio"] == 0.0

        vp = buf.compute_volume_profile("NONEXISTENT")
        assert vp["bins"] == [] and vp["poc_price"] is None

        impact = buf.estimate_market_impact("NONEXISTENT", size=10.0, side=Side.BUY)
        assert impact["executed_size"] == 0.0 and impact["unfilled_size"] == 10.0

        assert buf.compute_risk_metrics("NONEXISTENT") is None
        assert buf.compute_telemetry_summary("NONEXISTENT") is None

    def test_ring_buffer_eviction(self) -> None:
        buf = MarketDataBuffer(capacity=10)
        for i in range(25):
            buf.push_tick(
                MarketTick(
                    symbol="AAPL",
                    price=100.0 + i,
                    size=1.0,
                    side=Side.BUY,
                    sequence=i + 1,
                )
            )

        ticks = buf.get_ticks("AAPL", limit=50)
        assert len(ticks) == 10
        assert ticks[0].sequence == 16
        assert ticks[-1].sequence == 25

    def test_compute_vwap_zero_volume(self) -> None:
        buf = MarketDataBuffer(capacity=10)
        buf.push_tick(MarketTick(symbol="AAPL", price=150.0, size=0.0, side=Side.BUY))
        vwap = buf.compute_vwap("AAPL")
        assert vwap == 150.0

    def test_volume_profile_flat_prices(self) -> None:
        buf = MarketDataBuffer(capacity=10)
        for _ in range(5):
            buf.push_tick(MarketTick(symbol="AAPL", price=150.0, size=10.0, side=Side.BUY))
        vp = buf.compute_volume_profile("AAPL", bins=5)
        assert len(vp["bins"]) == 1
        assert vp["poc_price"] == 150.0
        assert vp["total_volume"] == 50.0

    def test_market_impact_unfilled_and_insufficient_depth(self) -> None:
        buf = MarketDataBuffer(capacity=10)
        ob = OrderBook(
            symbol="AAPL",
            bids=[OrderBookLevel(price=100.0, quantity=5.0)],
            asks=[OrderBookLevel(price=101.0, quantity=5.0)],
        )
        buf.push_order_book(ob)

        # Request 20 shares when only 5 are available
        impact = buf.estimate_market_impact("AAPL", size=20.0, side=Side.BUY)
        assert impact["executed_size"] == 5.0
        assert impact["unfilled_size"] == 15.0
        assert impact["avg_execution_price"] == 101.0

        # Empty asks
        ob_empty_asks = OrderBook(symbol="AAPL", bids=[OrderBookLevel(price=100.0, quantity=5.0)], asks=[])
        buf.push_order_book(ob_empty_asks)
        impact_no_asks = buf.estimate_market_impact("AAPL", size=5.0, side=Side.BUY)
        assert impact_no_asks["executed_size"] == 0.0

    def test_buffer_clear(self) -> None:
        buf = MarketDataBuffer(capacity=10)
        buf.push_tick(MarketTick(symbol="AAPL", price=150.0, size=1.0, side=Side.BUY))
        assert len(buf.symbols) == 1
        buf.clear()
        assert len(buf.symbols) == 0
        assert buf.get_ticks("AAPL") == []


class TestExecutionAndPositionEdgeCases:
    """Short selling, position flips, zero quantity fills, and stop triggers."""

    def test_position_zero_quantity_fill(self) -> None:
        pos = Position(symbol="AAPL")
        pnl = pos.apply_fill(Side.BUY, 0.0, 150.0)
        assert pnl == 0.0
        assert pos.quantity == 0.0

    def test_position_negative_market_price_update(self) -> None:
        pos = Position(symbol="AAPL", current_price=100.0)
        pos.update_market_price(-50.0)
        assert pos.current_price == 100.0

    def test_position_short_flip_to_long(self) -> None:
        pos = Position(symbol="AAPL")
        # Short 10 @ 100
        pnl1 = pos.apply_fill(Side.SELL, 10.0, 100.0)
        assert pnl1 == 0.0
        assert pos.quantity == -10.0
        assert pos.average_entry_price == 100.0

        # Add to short: Short 5 more @ 110
        pos.apply_fill(Side.SELL, 5.0, 110.0)
        assert pos.quantity == -15.0
        assert pos.average_entry_price == round((10 * 100 + 5 * 110) / 15, 4)

        # Mark price at 90 (in profit)
        pos.update_market_price(90.0)
        assert pos.unrealized_pnl > 0

        # Buy 20 @ 90 -> covers 15 short (pnl = (entry - 90)*15) and opens 5 long @ 90
        pnl_cover = pos.apply_fill(Side.BUY, 20.0, 90.0)
        assert pnl_cover > 0
        assert pos.quantity == 5.0
        assert pos.average_entry_price == 90.0

    def test_position_long_flip_to_short(self) -> None:
        pos = Position(symbol="AAPL")
        # Long 10 @ 100
        pos.apply_fill(Side.BUY, 10.0, 100.0)
        # Sell 15 @ 120 -> covers 10 long (pnl = 200) and opens 5 short @ 120
        pnl = pos.apply_fill(Side.SELL, 15.0, 120.0)
        assert pnl == 200.0
        assert pos.quantity == -5.0
        assert pos.average_entry_price == 120.0

    def test_simulator_cancel_order_edge_cases(self) -> None:
        sim = ExecutionSimulator()
        # Cancel non-existent
        assert sim.cancel_order("bad_id") is None

        # Cancel active order
        ord1 = sim.submit_order("AAPL", Side.BUY, 10.0, OrderType.LIMIT, price=50.0)
        assert ord1.is_active
        cancelled = sim.cancel_order(ord1.order_id)
        assert cancelled is not None and cancelled.status == OrderStatus.CANCELLED

        # Cancel already cancelled
        second_cancel = sim.cancel_order(ord1.order_id)
        assert second_cancel is not None and second_cancel.status == OrderStatus.CANCELLED

    def test_simulator_get_orders_and_trades_filtering(self) -> None:
        sim = ExecutionSimulator()
        ord_aapl = sim.submit_order("AAPL", Side.BUY, 10.0, OrderType.LIMIT, price=50.0)
        ord_nvda = sim.submit_order("NVDA", Side.BUY, 10.0, OrderType.LIMIT, price=50.0)

        # Filter by symbol
        aapl_orders = sim.get_open_orders(symbol="AAPL")
        assert len(aapl_orders) == 1
        assert aapl_orders[0].order_id == ord_aapl.order_id

        all_open = sim.get_open_orders()
        assert len(all_open) == 2

        all_hist = sim.get_order_history(symbol="NVDA")
        assert len(all_hist) == 1
        assert all_hist[0].order_id == ord_nvda.order_id

    def test_simulator_stop_loss_buy_trigger(self) -> None:
        sim = ExecutionSimulator()
        ob = OrderBook(
            symbol="AAPL",
            bids=[OrderBookLevel(price=100.0, quantity=10.0)],
            asks=[OrderBookLevel(price=102.0, quantity=10.0)],
        )
        sim.update_order_book(ob)

        # Stop buy order triggered when price breaks above 105.0
        stop_buy = sim.submit_order(
            "AAPL",
            Side.BUY,
            5.0,
            OrderType.STOP,
            stop_price=105.0,
        )
        assert stop_buy.is_active

        # Market tick below stop: no trigger
        sim.on_tick(MarketTick(symbol="AAPL", price=104.0, size=1.0, side=Side.BUY))
        assert stop_buy.is_active

        # Market tick rises to 105.5: triggers stop buy
        trades = sim.on_tick(MarketTick(symbol="AAPL", price=105.5, size=1.0, side=Side.BUY))
        assert len(trades) > 0
        assert stop_buy.status == OrderStatus.FILLED


class TestStorageAndReplayEdgeCases:
    """Empty files, invalid formats, seek operations, and corrupted records."""

    def test_persistence_empty_collections(self, tmp_path: Path) -> None:
        jsonl_ticks = tmp_path / "empty_ticks.jsonl"
        save_ticks_to_jsonl([], jsonl_ticks)
        assert load_ticks_from_jsonl(jsonl_ticks) == []

        jsonl_bars = tmp_path / "empty_bars.jsonl"
        save_bars_to_jsonl([], jsonl_bars)
        assert load_bars_from_jsonl(jsonl_bars) == []

        csv_ticks = tmp_path / "empty_ticks.csv"
        save_ticks_to_csv([], csv_ticks)
        assert load_ticks_from_csv(csv_ticks) == []

    def test_persistence_blank_lines_in_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "blank_lines.jsonl"
        f.write_text("\n\n   \n", encoding="utf-8")
        assert load_ticks_from_jsonl(f) == []

    def test_load_snapshot_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt_snapshot.json"
        bad_file.write_text("{corrupt: json}", encoding="utf-8")
        with pytest.raises(Exception):
            load_buffer_snapshot(bad_file)

    def test_replay_unsupported_file_extension(self, tmp_path: Path) -> None:
        unsupported = tmp_path / "data.unsupported"
        unsupported.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            HistoricalReplayEngine(source_file=unsupported)

    def test_replay_seek_and_reset(self, tmp_path: Path) -> None:
        # Create 5 ticks
        ticks = [
            MarketTick(symbol="AAPL", price=100.0 + i, size=1.0, side=Side.BUY, sequence=i + 1)
            for i in range(5)
        ]
        f = tmp_path / "seek_test.jsonl"
        save_ticks_to_jsonl(ticks, f)

        engine = HistoricalReplayEngine(source_file=f)
        assert engine.total_ticks == 5

        # Seek to valid index
        engine.seek(2)
        assert engine.cursor == 2

        # Seek out-of-bounds clamps to boundaries
        engine.seek(100)
        assert engine.cursor == 5
        engine.seek(-10)
        assert engine.cursor == 0

        engine.reset()
        assert engine.cursor == 0


class TestAlertsEdgeCases:
    """Alert deletion, listener failures, and spread/imbalance triggers."""

    def test_alert_engine_deletion_non_existent(self) -> None:
        engine = AlertEngine()
        assert engine.delete_rule("fake_rule_id") is False

    def test_alert_listener_exception_resilience(self) -> None:
        engine = AlertEngine()
        engine.create_rule("AAPL", AlertType.PRICE_ABOVE, 100.0)

        # Faulty listener
        def _bad_listener(_evt):
            raise RuntimeError("Boom!")

        received = []
        def _good_listener(evt):
            received.append(evt)

        engine.subscribe(_bad_listener)
        engine.subscribe(_good_listener)

        # Dispatch tick: engine should not crash, good listener still receives event
        engine.on_tick(MarketTick(symbol="AAPL", price=105.0, size=1.0, side=Side.BUY))
        assert len(received) == 1

    def test_alert_spread_and_imbalance_triggers(self) -> None:
        engine = AlertEngine()
        r_spread = engine.create_rule("AAPL", AlertType.SPREAD_WIDER_THAN, 2.0)
        r_imb = engine.create_rule("AAPL", AlertType.IMBALANCE_SPIKE, 0.7)

        # Order book with spread 3.0 and imbalance 0.8
        ob = OrderBook(
            symbol="AAPL",
            bids=[OrderBookLevel(price=100.0, quantity=90.0)],
            asks=[OrderBookLevel(price=103.0, quantity=10.0)],
        )
        engine.on_order_book(ob)

        events = engine.get_event_history("AAPL")
        assert len(events) == 2
        assert any(e.alert_id == r_spread.alert_id for e in events)
        assert any(e.alert_id == r_imb.alert_id for e in events)
