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
            port_req = JsonRpcRequest(id=4, method="tools/call", params={"name": "get_portfolio_state"})
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
            list_req = JsonRpcRequest(id=11, method="tools/call", params={"name": "list_market_alerts"})
            list_resp = await server.handle_request(list_req)
            alerts_data = json.loads(list_resp.result["content"][0]["text"])
            assert len(alerts_data["active_rules"]) >= 1

            # Read resource alerts://active
            res_req = JsonRpcRequest(id=12, method="resources/read", params={"uri": "alerts://active"})
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
                params={"name": "export_buffer_snapshot", "arguments": {"file_path": snapshot_path}},
            )
            export_resp = await server.handle_request(export_req)
            assert export_resp.id == 20
            export_data = json.loads(export_resp.result["content"][0]["text"])
            assert export_data["total_ticks"] > 0

            # Import snapshot
            import_req = JsonRpcRequest(
                id=21,
                method="tools/call",
                params={"name": "import_buffer_snapshot", "arguments": {"file_path": snapshot_path}},
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
                JsonRpcRequest(id=31, method="resources/read", params={"uri": "execution://history"})
            )
            exec_content = json.loads(exec_res.result["contents"][0]["text"])
            assert "trades" in exec_content

            # Check new prompt evaluate_trading_opportunity
            prompt_res = await server.handle_request(
                JsonRpcRequest(
                    id=32,
                    method="prompts/get",
                    params={
                        "name": "evaluate_trading_opportunity",
                        "arguments": {"symbol": "AAPL", "side": "buy"},
                    },
                )
            )
            assert "messages" in prompt_res.result
            assert "AAPL" in prompt_res.result["messages"][0]["content"]["text"]

        asyncio.run(_run())
