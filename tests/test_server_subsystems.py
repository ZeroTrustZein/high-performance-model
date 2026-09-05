"""Comprehensive test suite for server subsystem integrations: execution, alerting, persistence, resources, and prompts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from high_performance_model.protocol.models import JsonRpcRequest
from high_performance_model.server.app import create_fintech_mcp_server


class TestServerSubsystemToolsAndResources:
    """Verify newly integrated subsystem tools, resources, and prompts."""

    @pytest.fixture
    def server(self):
        return create_fintech_mcp_server(buffer_capacity=1000, warm_up_tickers=["AAPL", "NVDA"])

    def test_simulated_order_tools(self, server) -> None:
        async def _run():
            # 1. Submit market buy order
            order_req = JsonRpcRequest(
                id=1,
                method="tools/call",
                params={
                    "name": "submit_simulated_order",
                    "arguments": {
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 10.0,
                        "order_type": "market",
                    },
                },
            )
            order_resp = await server.handle_request(order_req)
            assert order_resp.id == 1
            order_data = json.loads(order_resp.result["content"][0]["text"])
            assert order_data["symbol"] == "AAPL"
            assert order_data["status"] == "filled"
            assert order_data["filled_quantity"] == 10.0
            assert order_data["average_fill_price"] > 0

            # 2. Submit limit order
            limit_req = JsonRpcRequest(
                id=2,
                method="tools/call",
                params={
                    "name": "submit_simulated_order",
                    "arguments": {
                        "symbol": "NVDA",
                        "side": "buy",
                        "quantity": 5.0,
                        "order_type": "limit",
                        "price": 100.0,
                    },
                },
            )
            limit_resp = await server.handle_request(limit_req)
            limit_data = json.loads(limit_resp.result["content"][0]["text"])
            assert limit_data["status"] == "pending"
            ord_id = limit_data["order_id"]

            # 3. Cancel limit order
            cancel_req = JsonRpcRequest(
                id=3,
                method="tools/call",
                params={
                    "name": "cancel_simulated_order",
                    "arguments": {"order_id": ord_id},
                },
            )
            cancel_resp = await server.handle_request(cancel_req)
            cancel_data = json.loads(cancel_resp.result["content"][0]["text"])
            assert cancel_data["status"] == "cancelled"

            # 4. Query portfolio state
            port_req = JsonRpcRequest(
                id=4, method="tools/call", params={"name": "get_portfolio_state"}
            )
            port_resp = await server.handle_request(port_req)
            port_data = json.loads(port_resp.result["content"][0]["text"])
            assert "cash_balance" in port_data
            assert "positions" in port_data
            assert "AAPL" in port_data["positions"]

        asyncio.run(_run())

    def test_alerting_tools_and_resources(self, server) -> None:
        async def _run():
            # Create alert
            create_req = JsonRpcRequest(
                id=10,
                method="tools/call",
                params={
                    "name": "create_market_alert",
                    "arguments": {
                        "symbol": "AAPL",
                        "alert_type": "price_above",
                        "threshold": 300.0,
                        "message": "AAPL break 300",
                    },
                },
            )
            create_resp = await server.handle_request(create_req)
            assert create_resp.id == 10
            rule_data = json.loads(create_resp.result["content"][0]["text"])
            assert rule_data["symbol"] == "AAPL"
            assert rule_data["alert_type"] == "price_above"

            # List alerts
            list_req = JsonRpcRequest(
                id=11, method="tools/call", params={"name": "list_market_alerts"}
            )
            list_resp = await server.handle_request(list_req)
            alerts_data = json.loads(list_resp.result["content"][0]["text"])
            assert len(alerts_data["active_rules"]) >= 1

            # Read resource alerts://active
            res_req = JsonRpcRequest(
                id=12, method="resources/read", params={"uri": "alerts://active"}
            )
            res_resp = await server.handle_request(res_req)
            content = json.loads(res_resp.result["contents"][0]["text"])
            assert "active_rules" in content

        asyncio.run(_run())

    def test_persistence_snapshot_tools(self, server, tmp_path: Path) -> None:
        async def _run():
            snapshot_path = str(tmp_path / "server_snap.json")

            # Export snapshot
            export_req = JsonRpcRequest(
                id=20,
                method="tools/call",
                params={
                    "name": "export_buffer_snapshot",
                    "arguments": {"file_path": snapshot_path},
                },
            )
            export_resp = await server.handle_request(export_req)
            assert export_resp.id == 20
            export_data = json.loads(export_resp.result["content"][0]["text"])
            assert export_data["total_ticks"] > 0

            # Import snapshot
            import_req = JsonRpcRequest(
                id=21,
                method="tools/call",
                params={
                    "name": "import_buffer_snapshot",
                    "arguments": {"file_path": snapshot_path},
                },
            )
            import_resp = await server.handle_request(import_req)
            import_data = json.loads(import_resp.result["content"][0]["text"])
            assert import_data["status"] == "success"

        asyncio.run(_run())

    def test_portfolio_and_execution_resources(self, server) -> None:
        async def _run():
            # Read portfolio://state
            port_res = await server.handle_request(
                JsonRpcRequest(id=30, method="resources/read", params={"uri": "portfolio://state"})
            )
            port_content = json.loads(port_res.result["contents"][0]["text"])
            assert "total_portfolio_value" in port_content

            # Read execution://history
            exec_res = await server.handle_request(
                JsonRpcRequest(
                    id=31, method="resources/read", params={"uri": "execution://history"}
                )
            )
            exec_content = json.loads(exec_res.result["contents"][0]["text"])
            assert "trades" in exec_content

            # Read market://telemetry/snapshot
            market_res = await server.handle_request(
                JsonRpcRequest(
                    id=32, method="resources/read", params={"uri": "market://telemetry/snapshot"}
                )
            )
            market_content = json.loads(market_res.result["contents"][0]["text"])
            assert "AAPL" in market_content
            assert "price" in market_content["AAPL"]

            # Read invalid resource
            bad_res = await server.handle_request(
                JsonRpcRequest(id=33, method="resources/read", params={"uri": "unknown://res"})
            )
            assert bad_res.error is not None
            assert bad_res.error.code == -32602

            # Check new prompt evaluate_trading_opportunity
            prompt_res = await server.handle_request(
                JsonRpcRequest(
                    id=34,
                    method="prompts/get",
                    params={
                        "name": "evaluate_trading_opportunity",
                        "arguments": {"symbol": "AAPL", "side": "buy"},
                    },
                )
            )
            assert "messages" in prompt_res.result
            assert "AAPL" in prompt_res.result["messages"][0]["content"]["text"]

            # Check invalid prompt
            bad_prompt = await server.handle_request(
                JsonRpcRequest(
                    id=35,
                    method="prompts/get",
                    params={"name": "unknown_prompt"},
                )
            )
            assert bad_prompt.error is not None
            assert bad_prompt.error.code == -32602

        asyncio.run(_run())

    def test_market_depth_and_indicators_tools(self, server) -> None:
        async def _run():
            # 1. get_market_quote
            quote_req = JsonRpcRequest(
                id=40,
                method="tools/call",
                params={"name": "get_market_quote", "arguments": {"symbol": "AAPL"}},
            )
            quote_resp = await server.handle_request(quote_req)
            assert quote_resp.id == 40
            quote_data = json.loads(quote_resp.result["content"][0]["text"])
            assert quote_data["symbol"] == "AAPL"
            assert "last_price" in quote_data

            # 2. get_order_book_depth
            depth_req = JsonRpcRequest(
                id=41,
                method="tools/call",
                params={
                    "name": "get_order_book_depth",
                    "arguments": {"symbol": "NVDA", "depth": 8},
                },
            )
            depth_resp = await server.handle_request(depth_req)
            assert depth_resp.id == 41
            depth_data = json.loads(depth_resp.result["content"][0]["text"])
            assert depth_data["symbol"] == "NVDA"
            assert len(depth_data["bids"]) <= 8
            assert "spread" in depth_data
            assert "imbalance" in depth_data

            # 3. calculate_technical_indicators with all indicator branches
            all_inds = [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bollinger",
                "atr",
                "vwap",
                "momentum",
                "roc",
                "stochastic",
            ]
            ind_req = JsonRpcRequest(
                id=42,
                method="tools/call",
                params={
                    "name": "calculate_technical_indicators",
                    "arguments": {"symbol": "AAPL", "indicators": all_inds},
                },
            )
            ind_resp = await server.handle_request(ind_req)
            assert ind_resp.id == 42
            ind_data = json.loads(ind_resp.result["content"][0]["text"])
            assert ind_data["symbol"] == "AAPL"
            assert "sma_20" in ind_data
            assert "ema_20" in ind_data
            assert "rsi_14" in ind_data
            assert "macd" in ind_data
            assert "bollinger" in ind_data
            assert "atr_14" in ind_data
            assert "vwap" in ind_data
            assert "momentum_10" in ind_data
            assert "roc_10" in ind_data
            assert "stochastic" in ind_data

        asyncio.run(_run())

    def test_analytics_and_impact_tools(self, server) -> None:
        async def _run():
            # 1. compute_risk_metrics
            risk_req = JsonRpcRequest(
                id=50,
                method="tools/call",
                params={"name": "compute_risk_metrics", "arguments": {"symbol": "AAPL"}},
            )
            risk_resp = await server.handle_request(risk_req)
            assert risk_resp.id == 50
            risk_data = json.loads(risk_resp.result["content"][0]["text"])
            assert risk_data["symbol"] == "AAPL"
            assert "realized_volatility_annualized" in risk_data
            assert "sharpe_ratio_estimate" in risk_data
            assert "value_at_risk_95" in risk_data

            # 2. get_volume_profile
            vp_req = JsonRpcRequest(
                id=51,
                method="tools/call",
                params={"name": "get_volume_profile", "arguments": {"symbol": "AAPL", "bins": 5}},
            )
            vp_resp = await server.handle_request(vp_req)
            assert vp_resp.id == 51
            vp_data = json.loads(vp_resp.result["content"][0]["text"])
            assert vp_data["symbol"] == "AAPL"
            assert "bins" in vp_data
            assert "poc_price" in vp_data

            # 3. estimate_market_impact (BUY and SELL)
            impact_buy_req = JsonRpcRequest(
                id=52,
                method="tools/call",
                params={
                    "name": "estimate_market_impact",
                    "arguments": {"symbol": "AAPL", "size": 10.0, "side": "buy"},
                },
            )
            impact_buy_resp = await server.handle_request(impact_buy_req)
            assert impact_buy_resp.id == 52
            impact_buy_data = json.loads(impact_buy_resp.result["content"][0]["text"])
            assert impact_buy_data["symbol"] == "AAPL"
            assert impact_buy_data["requested_size"] == 10.0

            impact_sell_req = JsonRpcRequest(
                id=53,
                method="tools/call",
                params={
                    "name": "estimate_market_impact",
                    "arguments": {"symbol": "AAPL", "size": 5.0, "side": "sell"},
                },
            )
            impact_sell_resp = await server.handle_request(impact_sell_req)
            assert impact_sell_resp.id == 53
            impact_sell_data = json.loads(impact_sell_resp.result["content"][0]["text"])
            assert impact_sell_data["symbol"] == "AAPL"

        asyncio.run(_run())

    def test_protocol_and_error_handling(self, server) -> None:
        async def _run():
            # 1. Notification: initialized
            notif = JsonRpcRequest(method="notifications/initialized")
            resp_notif = await server.handle_request(notif)
            assert resp_notif is None

            # 2. Ping
            ping_req = JsonRpcRequest(id=60, method="ping")
            ping_resp = await server.handle_request(ping_req)
            assert ping_resp.id == 60
            assert ping_resp.result == {}

            # 3. Unknown method
            bad_method = JsonRpcRequest(id=61, method="non_existent_method")
            resp_bad_method = await server.handle_request(bad_method)
            assert resp_bad_method.error is not None
            assert resp_bad_method.error.code == -32601

            # 4. Unknown tool
            bad_tool = JsonRpcRequest(
                id=62,
                method="tools/call",
                params={"name": "non_existent_tool", "arguments": {}},
            )
            resp_bad_tool = await server.handle_request(bad_tool)
            assert resp_bad_tool.error is not None
            assert resp_bad_tool.error.code == -32602

            # 5. Snapshot import failure path
            bad_import_req = JsonRpcRequest(
                id=63,
                method="tools/call",
                params={
                    "name": "import_buffer_snapshot",
                    "arguments": {"file_path": "non_existent_snapshot_path_123.json"},
                },
            )
            resp_import = await server.handle_request(bad_import_req)
            assert resp_import.id == 63
            import_data = json.loads(resp_import.result["content"][0]["text"])
            assert import_data["status"] == "error"

        asyncio.run(_run())
