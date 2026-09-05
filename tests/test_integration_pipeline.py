"""End-to-end integration test suite for FinTech MCP Server.

Validates the full lifecycle:
Synthetic Feed -> Ring-Buffer Ingestion & Bar Aggregation -> Microstructure Analytics ->
Vectorized Indicators -> Multi-Condition Alerts -> L2 Execution & Portfolio Tracking ->
Disk Persistence & Historical Replay -> Full MCP Protocol Communication over JSON-RPC.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from high_performance_model.alerts.engine import AlertEngine
from high_performance_model.alerts.models import AlertType
from high_performance_model.execution.engine import ExecutionSimulator
from high_performance_model.execution.models import OrderStatus
from high_performance_model.indicators.series import (
    average_true_range,
    bollinger_bands,
    exponential_moving_average,
    macd,
    momentum,
    rate_of_change,
    relative_strength_index,
    simple_moving_average,
    stochastic_oscillator,
    volume_weighted_average_price,
)
from high_performance_model.protocol.transports import MemoryTransport
from high_performance_model.server.app import create_fintech_mcp_server
from high_performance_model.storage.persistence import (
    load_buffer_snapshot,
    save_buffer_snapshot,
    save_ticks_to_jsonl,
)
from high_performance_model.storage.replay import HistoricalReplayEngine
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import (
    BarTimeframe,
    OrderType,
    Side,
)


class TestFullMarketLifecycle:
    """End-to-end lifecycle integration covering ingestion, analytics, execution, and protocol."""

    def test_end_to_end_market_pipeline(self, tmp_path: Path) -> None:
        symbols = ["AAPL", "NVDA", "BTC/USD"]
        feed = SyntheticMarketFeed(symbols=symbols, seed=42)
        buffer = MarketDataBuffer(capacity=500)

        # 1. Ingestion of 200 ticks per symbol and order book generation
        all_ticks = []
        for sym in symbols:
            ticks = feed.generate_history(sym, n_points=200)
            all_ticks.extend(ticks)
            for t in ticks:
                buffer.push_tick(t)
            ob = feed.generate_order_book(sym, depth=10)
            buffer.push_order_book(ob)

        assert len(buffer.symbols) == 3
        for sym in symbols:
            stored_ticks = buffer.get_ticks(sym, limit=300)
            assert len(stored_ticks) == 200
            assert buffer.get_latest_tick(sym) is not None
            assert buffer.get_latest_order_book(sym) is not None

        # 2. Aggregations and Microstructural metrics
        for sym in symbols:
            bars = buffer.aggregate_and_store_bars(sym, timeframe=BarTimeframe.MIN_1)
            assert len(bars) >= 1

            vwap = buffer.compute_vwap(sym)
            assert vwap is not None and vwap > 0

            vol = buffer.compute_realized_volatility(sym)
            assert vol is not None and vol >= 0

            quote = buffer.compute_top_quote(sym)
            assert quote is not None
            assert quote.bid < quote.ask

            depth = buffer.compute_market_depth_snapshot(sym, depth=5)
            assert depth is not None
            assert len(depth.bids) <= 5

            eff_spread = buffer.compute_effective_spread(sym)
            assert eff_spread is not None and eff_spread >= 0

            ofi = buffer.compute_order_flow_imbalance(sym)
            assert "buy_volume" in ofi and "sell_volume" in ofi

            v_prof = buffer.compute_volume_profile(sym, bins=8)
            assert len(v_prof["bins"]) == 8
            assert v_prof["poc_price"] is not None

            impact_buy = buffer.estimate_market_impact(sym, size=10.0, side=Side.BUY)
            assert impact_buy["executed_size"] > 0
            assert impact_buy["avg_execution_price"] is not None

            summary = buffer.compute_telemetry_summary(sym)
            assert summary is not None
            assert summary.symbol == sym

        # 3. Vectorized Indicators on Ingested Series
        aapl_prices = buffer.get_price_series("AAPL", limit=200)
        aapl_vols = buffer.get_volume_series("AAPL", limit=200)
        assert len(aapl_prices) == 200

        sma = simple_moving_average(aapl_prices, period=20)
        ema = exponential_moving_average(aapl_prices, period=20)
        rsi = relative_strength_index(aapl_prices, period=14)
        macd_line, sig_line, hist = macd(aapl_prices)
        bb = bollinger_bands(aapl_prices, period=20)
        atr_vals = average_true_range(aapl_prices * 1.01, aapl_prices * 0.99, aapl_prices, period=14)
        vwap_val = volume_weighted_average_price(aapl_prices, aapl_vols)
        mom = momentum(aapl_prices, period=10)
        roc = rate_of_change(aapl_prices, period=10)
        k_stoch, d_stoch = stochastic_oscillator(
            aapl_prices * 1.01, aapl_prices * 0.99, aapl_prices, 14, 3, 3
        )

        assert not np.isnan(sma[-1])
        assert not np.isnan(ema[-1])
        assert 0.0 <= rsi[-1] <= 100.0
        assert not np.isnan(macd_line[-1])
        assert bb["upper"][-1] >= bb["middle"][-1] >= bb["lower"][-1]
        assert atr_vals[-1] > 0
        assert vwap_val > 0
        assert not np.isnan(mom[-1])
        assert not np.isnan(roc[-1])
        assert 0.0 <= k_stoch[-1] <= 100.0

        # 4. Event-Driven Alert Monitoring
        alert_engine = AlertEngine()
        latest_aapl = buffer.get_latest_tick("AAPL")
        assert latest_aapl is not None

        # Add price above and below rules
        r_above = alert_engine.create_rule(
            "AAPL", AlertType.PRICE_ABOVE, latest_aapl.price * 0.99, message="Breakout High"
        )
        r_below = alert_engine.create_rule(
            "AAPL", AlertType.PRICE_BELOW, latest_aapl.price * 1.01, message="Support Dip"
        )
        r_vol = alert_engine.create_rule(
            "AAPL", AlertType.VOLUME_SPIKE, 0.01, message="Volume alert"
        )

        fired_events = []
        alert_engine.subscribe(lambda evt: fired_events.append(evt))

        # Evaluate tick against engine
        alert_engine.on_tick(latest_aapl)
        assert len(fired_events) >= 2
        assert any(e.alert_id == r_above.alert_id for e in fired_events)
        assert any(e.alert_id == r_below.alert_id for e in fired_events)
        assert any(e.alert_id == r_vol.alert_id for e in fired_events)

        # 5. Simulated Order Execution & Portfolio Transitions
        sim = ExecutionSimulator()
        aapl_ob = buffer.get_latest_order_book("AAPL")
        assert aapl_ob is not None
        assert aapl_ob.best_ask is not None
        sim.update_order_book(aapl_ob)

        # (a) Market Buy 50 shares
        buy_order = sim.submit_order("AAPL", Side.BUY, 50.0, OrderType.MARKET)
        assert buy_order.status == OrderStatus.FILLED
        assert buy_order.filled_quantity == 50.0
        assert sim.portfolio.cash_balance < sim.portfolio.initial_cash
        assert sim.portfolio.positions["AAPL"].quantity == 50.0

        # (b) Limit Sell 25 shares above market
        limit_order = sim.submit_order(
            "AAPL",
            Side.SELL,
            25.0,
            OrderType.LIMIT,
            price=aapl_ob.best_ask + 5.0,
        )
        assert limit_order.is_active

        # Feed tick that triggers limit sell
        trigger_tick = feed.generate_tick("AAPL")
        trigger_tick.price = aapl_ob.best_ask + 6.0
        sim.on_tick(trigger_tick)
        assert limit_order.status == OrderStatus.FILLED
        assert sim.portfolio.positions["AAPL"].quantity == 25.0
        assert sim.portfolio.total_realized_pnl != 0.0

        # (c) Sell remaining 25 and short 10 shares in one market order
        sim.update_order_book(aapl_ob)
        reversal_order = sim.submit_order("AAPL", Side.SELL, 35.0, OrderType.MARKET)
        assert reversal_order.status == OrderStatus.FILLED
        assert sim.portfolio.positions["AAPL"].quantity == -10.0  # Now short 10 shares

        # 6. Persistence to JSONL and Snapshot Restore
        jsonl_path = tmp_path / "stream_history.jsonl"
        save_ticks_to_jsonl(all_ticks, jsonl_path)
        assert jsonl_path.exists()

        snapshot_path = tmp_path / "buffer_snapshot.json"
        meta = save_buffer_snapshot(buffer, snapshot_path)
        assert meta["total_ticks"] == 600
        assert meta["symbols_count"] == 3

        restored_buffer = load_buffer_snapshot(snapshot_path)
        assert restored_buffer.symbols == ["AAPL", "BTC/USD", "NVDA"]
        assert len(restored_buffer.get_ticks("AAPL", limit=300)) == 200

        # 7. Historical Replay Verification
        replay_engine = HistoricalReplayEngine(source_file=jsonl_path)
        replayed_ticks = []

        async def _run_replay():
            async for tick in replay_engine.stream_ticks(speed_multiplier=0.0):
                replayed_ticks.append(tick)

        asyncio.run(_run_replay())
        assert len(replayed_ticks) == 600
        assert replayed_ticks[0].symbol in symbols

        # 8. MCP Server Handshake and Protocol Verification over MemoryTransport
        server = create_fintech_mcp_server(warm_up_tickers=["AAPL", "NVDA"])

        async def _run_mcp_session():
            transport = MemoryTransport()

            # Initialize
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": "integration-client", "version": "1.0.0"}},
            }
            await transport.feed_input(json.dumps(init_payload))

            # Query tools
            tools_payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            await transport.feed_input(json.dumps(tools_payload))

            # Call order execution tool
            exec_payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "submit_simulated_order",
                    "arguments": {"symbol": "AAPL", "side": "buy", "quantity": 15.0},
                },
            }
            await transport.feed_input(json.dumps(exec_payload))

            # Read portfolio state resource
            res_payload = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {"uri": "portfolio://state"},
            }
            await transport.feed_input(json.dumps(res_payload))

            await transport.feed_eof()
            await server.run(transport)

            # Assert responses
            resp1_raw = await transport.read_output()
            assert resp1_raw is not None
            resp1 = json.loads(resp1_raw)
            assert resp1["id"] == 1
            assert "serverInfo" in resp1["result"]

            resp2_raw = await transport.read_output()
            assert resp2_raw is not None
            resp2 = json.loads(resp2_raw)
            assert resp2["id"] == 2
            tools = [t["name"] for t in resp2["result"]["tools"]]
            assert "submit_simulated_order" in tools

            resp3_raw = await transport.read_output()
            assert resp3_raw is not None
            resp3 = json.loads(resp3_raw)
            assert resp3["id"] == 3
            exec_res = json.loads(resp3["result"]["content"][0]["text"])
            assert exec_res["status"] == "filled"
            assert exec_res["filled_quantity"] == 15.0

            resp4_raw = await transport.read_output()
            assert resp4_raw is not None
            resp4 = json.loads(resp4_raw)
            assert resp4["id"] == 4
            port_content = json.loads(resp4["result"]["contents"][0]["text"])
            assert "AAPL" in port_content["positions"]

        asyncio.run(_run_mcp_session())


class TestMultiAssetConcurrentPipeline:
    """Validate multi-ticker isolation, stop orders, and cross-asset handling."""

    def test_concurrent_symbols_execution(self) -> None:
        feed = SyntheticMarketFeed(symbols=["AAPL", "BTC/USD", "EUR/USD"], seed=100)
        sim = ExecutionSimulator()

        # Generate order books for all symbols
        for sym in ["AAPL", "BTC/USD", "EUR/USD"]:
            ob = feed.generate_order_book(sym, depth=5)
            sim.update_order_book(ob)

        # Place orders across distinct asset classes
        ord_aapl = sim.submit_order("AAPL", Side.BUY, 10.0, OrderType.MARKET)
        ord_btc = sim.submit_order("BTC/USD", Side.BUY, 0.5, OrderType.MARKET)
        ord_eur = sim.submit_order("EUR/USD", Side.SELL, 20.0, OrderType.MARKET)

        assert ord_aapl.status == OrderStatus.FILLED
        assert ord_btc.status == OrderStatus.FILLED
        assert ord_eur.status == OrderStatus.FILLED

        # Verify portfolio holds all 3 positions cleanly
        positions = sim.portfolio.positions
        assert len(positions) == 3
        assert positions["AAPL"].quantity == 10.0
        assert positions["BTC/USD"].quantity == 0.5
        assert positions["EUR/USD"].quantity == -20.0

        # Test stop-loss sell trigger on market downturn
        aapl_price = positions["AAPL"].average_entry_price
        stop_order = sim.submit_order(
            "AAPL",
            Side.SELL,
            10.0,
            OrderType.STOP,
            stop_price=aapl_price * 0.95,
        )
        assert stop_order.is_active

        # Tick drops below stop price
        drop_tick = feed.generate_tick("AAPL")
        drop_tick.price = aapl_price * 0.94
        sim.on_tick(drop_tick)

        # Stop loss should have triggered and filled
        assert stop_order.status == OrderStatus.FILLED
        assert positions["AAPL"].quantity == 0.0
