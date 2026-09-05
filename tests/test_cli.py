"""Unit and integration tests for FinTech MCP CLI commands and formatting interface."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from high_performance_model.alerts.models import AlertEvent, AlertRule, AlertType
from high_performance_model.cli import (
    cli,
    format_alert_events_table,
    format_alert_rules_table,
    format_benchmark_table,
    format_indicators_table,
    format_mcp_prompts_table,
    format_mcp_resources_table,
    format_mcp_tools_table,
    format_order_book_table,
    format_orders_table,
    format_portfolio_panel,
    format_positions_table,
    format_quote_panel,
)
from high_performance_model.execution.models import Portfolio, SimulatedOrder
from high_performance_model.storage.persistence import save_ticks_to_jsonl
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import (
    BenchmarkRunResult,
    OrderType,
    Side,
)


class TestCliBasicsAndHelp:
    """Verify general CLI entry point, metadata, and versioning."""

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "High-performance Model Context Protocol" in result.output
        assert "serve" in result.output
        assert "quote" in result.output
        assert "indicators" in result.output
        assert "stream" in result.output
        assert "replay" in result.output
        assert "snapshot" in result.output
        assert "alerts" in result.output
        assert "exec" in result.output
        assert "mcp" in result.output
        assert "benchmark" in result.output

    def test_cli_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCliServe:
    """Verify MCP serve command with stdio and sse transports."""

    def test_serve_stdio(self, monkeypatch) -> None:
        ran = False

        def fake_run(coro):
            coro.close()
            nonlocal ran
            ran = True

        import asyncio

        monkeypatch.setattr(asyncio, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--transport", "stdio", "--capacity", "500"])
        assert result.exit_code == 0
        assert ran is True

    def test_serve_sse(self, monkeypatch) -> None:
        ran = False

        def fake_run(coro):
            coro.close()
            nonlocal ran
            ran = True

        import asyncio

        monkeypatch.setattr(asyncio, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--transport", "sse", "--port", "8888"])
        assert result.exit_code == 0
        assert "Starting FinTech MCP Server" in result.output
        assert ran is True

    def test_serve_keyboard_interrupt(self, monkeypatch) -> None:
        def fake_run(coro):
            coro.close()
            raise KeyboardInterrupt

        import asyncio

        monkeypatch.setattr(asyncio, "run", fake_run)

        runner = CliRunner()
        result_stdio = runner.invoke(cli, ["serve", "--transport", "stdio"])
        assert result_stdio.exit_code == 0

        result_sse = runner.invoke(cli, ["serve", "--transport", "sse"])
        assert result_sse.exit_code == 0
        assert "Shutting down SSE server" in result_sse.output


class TestCliQuoteAndMarket:
    """Verify market quote and order book snapshot CLI."""

    def test_quote_default(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["quote"])
        assert result.exit_code == 0
        assert "Market Quote: AAPL" in result.output
        assert "Order Book Depth" in result.output
        assert "Last Price" in result.output

    def test_quote_custom_symbol_and_depth(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["quote", "NVDA", "--depth", "3"])
        assert result.exit_code == 0
        assert "Market Quote: NVDA" in result.output
        assert "Top 3 Levels" in result.output

    def test_quote_json_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["quote", "MSFT", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "tick" in data and "order_book" in data
        assert data["tick"]["symbol"] == "MSFT"
        assert len(data["order_book"]["bids"]) == 5

    def test_quote_invalid_symbol(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["quote", "INVALID$$$"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCliIndicators:
    """Verify vectorized indicators calculation CLI."""

    def test_indicators_all_table(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["indicators", "AAPL"])
        assert result.exit_code == 0
        assert "Technical Indicators: AAPL" in result.output
        assert "SMA" in result.output
        assert "EMA" in result.output
        assert "RSI" in result.output
        assert "MACD" in result.output
        assert "Bollinger Bands" in result.output
        assert "ATR" in result.output
        assert "VWAP" in result.output

    def test_indicators_specific_and_json(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["indicators", "GOOGL", "-i", "rsi", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["symbol"] == "GOOGL"
        assert "RSI" in data["indicators"]
        assert "value" in data["indicators"]["RSI"]

    def test_indicators_individual_choices(self) -> None:
        runner = CliRunner()
        for ind in ["sma", "bollinger", "stochastic", "momentum", "roc"]:
            result = runner.invoke(cli, ["indicators", "AAPL", "-i", ind])
            assert result.exit_code == 0
            assert "Technical Indicators: AAPL" in result.output

    def test_indicators_invalid_symbol(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["indicators", "BAD#SYM"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCliStream:
    """Verify real-time tick/bar stream generator."""

    def test_stream_ticks_table(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stream", "AAPL", "-n", "3", "-r", "100.0"])
        assert result.exit_code == 0
        assert "Live Feed: AAPL (TICK)" in result.output

    def test_stream_bars_jsonl(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["stream", "AAPL", "-n", "2", "-m", "bar", "-r", "100.0", "-f", "jsonl"],
        )
        assert result.exit_code == 0
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["symbol"] == "AAPL"
        assert "close" in parsed

    def test_stream_ticks_json(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["stream", "NVDA", "-n", "1", "-r", "100.0", "-f", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["symbol"] == "NVDA"

    def test_stream_invalid_symbol(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["stream", "BAD@SYM"])
        assert result.exit_code == 1


class TestCliReplayAndSnapshot:
    """Verify historical replay and buffer snapshot persistence."""

    def test_replay_file(self, tmp_path: Path) -> None:
        feed = SyntheticMarketFeed(symbols=["AAPL"])
        ticks = feed.generate_history("AAPL", n_points=10)
        jsonl_file = tmp_path / "replay_test.jsonl"
        save_ticks_to_jsonl(ticks, jsonl_file)

        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(jsonl_file), "--speed", "0.0"])
        assert result.exit_code == 0
        assert "Historical Replay Summary" in result.output
        assert "10 of 10" in result.output

    def test_replay_with_limit_and_filter(self, tmp_path: Path) -> None:
        feed = SyntheticMarketFeed(symbols=["AAPL"])
        ticks = feed.generate_history("AAPL", n_points=10)
        jsonl_file = tmp_path / "replay_test.jsonl"
        save_ticks_to_jsonl(ticks, jsonl_file)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["replay", str(jsonl_file), "--limit", "4", "--symbol", "AAPL"],
        )
        assert result.exit_code == 0
        assert "4 of 4" in result.output

    def test_replay_empty_symbol_filter(self, tmp_path: Path) -> None:
        feed = SyntheticMarketFeed(symbols=["AAPL"])
        ticks = feed.generate_history("AAPL", n_points=5)
        jsonl_file = tmp_path / "replay_test.jsonl"
        save_ticks_to_jsonl(ticks, jsonl_file)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["replay", str(jsonl_file), "--symbol", "MSFT"],
        )
        assert result.exit_code == 0
        assert "No matching ticks" in result.output

    def test_snapshot_lifecycle(self, tmp_path: Path) -> None:
        snap_file = tmp_path / "buffer_snap.json"
        runner = CliRunner()

        # 1. Save
        res_save = runner.invoke(
            cli,
            ["snapshot", "save", "-f", str(snap_file), "-s", "AAPL,NVDA", "-n", "10"],
        )
        assert res_save.exit_code == 0
        assert "Snapshot Saved Successfully" in res_save.output
        assert snap_file.exists()

        # 2. Inspect
        res_inspect = runner.invoke(cli, ["snapshot", "inspect", str(snap_file)])
        assert res_inspect.exit_code == 0
        assert "Snapshot Metadata" in res_inspect.output
        assert "AAPL" in res_inspect.output

        # 3. Load
        res_load = runner.invoke(cli, ["snapshot", "load", str(snap_file)])
        assert res_load.exit_code == 0
        assert "Snapshot Loaded Successfully" in res_load.output


class TestCliAlerts:
    """Verify alert rules and monitoring CLI."""

    def test_alerts_list(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["alerts", "list"])
        assert result.exit_code == 0
        assert "Market Alert Rules" in result.output
        assert "AAPL" in result.output

    def test_alerts_add(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "alerts",
                "add",
                "-s",
                "AAPL",
                "-t",
                "price_above",
                "--threshold",
                "250.0",
                "-m",
                "Breakout alert",
            ],
        )
        assert result.exit_code == 0
        assert "Alert Rule Registered" in result.output
        assert "price_above" in result.output

    def test_alerts_add_invalid_symbol(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["alerts", "add", "-s", "INVALID$$$", "-t", "price_above", "--threshold", "100.0"],
        )
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_alerts_monitor(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["alerts", "monitor", "AAPL", "-n", "5"])
        assert result.exit_code == 0
        assert "Monitoring alerts on AAPL" in result.output


class TestCliExecution:
    """Verify simulated execution and portfolio management CLI."""

    def test_exec_submit_market_order(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["exec", "submit", "AAPL", "--side", "buy", "-q", "25.0"])
        assert result.exit_code == 0
        assert "Simulated Order Placed" in result.output
        assert "FILLED" in result.output
        assert "Portfolio Status" in result.output

    def test_exec_submit_limit_order(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "exec",
                "submit",
                "NVDA",
                "--side",
                "sell",
                "-q",
                "10.0",
                "-t",
                "limit",
                "-p",
                "300.0",
            ],
        )
        assert result.exit_code == 0
        assert "Simulated Order Placed" in result.output

    def test_exec_submit_invalid_symbol(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["exec", "submit", "INVALID$$$", "--side", "buy", "-q", "10.0"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_exec_portfolio(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["exec", "portfolio"])
        assert result.exit_code == 0
        assert "Portfolio Status" in result.output
        assert "Open Positions" in result.output

    def test_exec_orders(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["exec", "orders"])
        assert result.exit_code == 0
        assert "Simulated Orders" in result.output


class TestCliMCPInspector:
    """Verify MCP tools, resources, and prompts CLI inspection."""

    def test_mcp_tools(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "tools"])
        assert result.exit_code == 0
        assert "Registered MCP Tools" in result.output
        assert "get_market_quote" in result.output
        assert "calculate_technical" in result.output

    def test_mcp_resources(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "resources"])
        assert result.exit_code == 0
        assert "Registered MCP Resources" in result.output
        assert "indicators" in result.output
        assert "market://" in result.output

    def test_mcp_prompts(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "prompts"])
        assert result.exit_code == 0
        assert "Registered MCP Prompts" in result.output
        assert "analyze_market_structure" in result.output

    def test_mcp_call_tool(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["mcp", "call", "-t", "get_market_quote", "-a", '{"symbol": "AAPL"}'],
        )
        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "last_price" in result.output

    def test_mcp_call_tool_json_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["mcp", "call", "-t", "get_market_quote", "-a", "INVALID_JSON{"],
        )
        assert result.exit_code == 1
        assert "JSON Parse Error" in result.output

    def test_mcp_read_resource(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "read", "-u", "indicators://catalog"])
        assert result.exit_code == 0
        assert "indicators" in result.output

    def test_mcp_read_invalid_resource(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "read", "-u", "invalid://uri"])
        assert result.exit_code == 1
        assert "MCP Error" in result.output


class TestCliBenchmark:
    """Verify performance benchmark command."""

    def test_benchmark_table_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--ticks", "500", "--symbol", "NVDA"])
        assert result.exit_code == 0
        assert "Performance Benchmark Results" in result.output
        assert "Ingestion Throughput" in result.output

    def test_benchmark_json_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--ticks", "500", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_ticks"] == 500
        assert "ticks_per_second" in data


class TestFormatters:
    """Direct tests on formatting helpers in formatters.py."""

    def test_quote_and_order_book_formatters(self) -> None:
        feed = SyntheticMarketFeed(symbols=["AAPL"])
        tick = feed.generate_tick("AAPL")
        ob = feed.generate_order_book("AAPL", depth=5)

        panel = format_quote_panel(tick, ob)
        assert panel.title is not None
        assert "AAPL" in str(panel.title)

        table = format_order_book_table(ob, depth=5)
        assert "AAPL" in str(table.title)

    def test_indicators_formatter(self) -> None:
        ind_data = {
            "SMA": {"value_str": "$150.00", "params": "period=20", "signal": "Bullish"},
            "RSI": {"value_str": "75.00", "params": "period=14", "signal": "Overbought"},
        }
        table = format_indicators_table("AAPL", ind_data)
        assert "AAPL" in str(table.title)

    def test_portfolio_and_positions_formatter(self) -> None:
        port = Portfolio(cash_balance=95_000.0)
        pos = port.get_or_create_position("AAPL")
        pos.quantity = 100.0
        pos.average_entry_price = 150.0
        pos.current_price = 155.0

        panel = format_portfolio_panel(port)
        assert panel is not None

        pos_table = format_positions_table(port)
        assert "Open Positions" in str(pos_table.title)

    def test_orders_table_formatter(self) -> None:
        ord1 = SimulatedOrder(
            symbol="AAPL",
            side=Side.BUY,
            quantity=50.0,
            order_type=OrderType.LIMIT,
            price=150.0,
        )
        table = format_orders_table([ord1])
        assert "Simulated Orders" in str(table.title)

    def test_alerts_table_formatters(self) -> None:
        rule = AlertRule(
            symbol="AAPL",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=180.0,
            message="Target reached",
        )
        evt = AlertEvent(
            alert_id=rule.alert_id,
            symbol="AAPL",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=180.0,
            current_value=182.5,
            message="Target triggered",
        )

        r_table = format_alert_rules_table([rule])
        assert "Market Alert Rules" in str(r_table.title)

        e_table = format_alert_events_table([evt])
        assert "Triggered Alert Events" in str(e_table.title)

    def test_benchmark_formatter(self) -> None:
        res = BenchmarkRunResult(
            total_ticks=1000,
            duration_seconds=0.01,
            ticks_per_second=100_000.0,
            latency_p50_us=1.5,
            latency_p95_us=2.5,
            latency_p99_us=5.0,
            buffer_utilization_pct=10.0,
            metadata={"indicators": {"sma": 0.5}},
        )
        table = format_benchmark_table(res)
        assert "Performance Benchmark Results" in str(table.title)

    def test_mcp_formatters(self) -> None:
        tools = [{"name": "tool1", "description": "desc1", "inputSchema": {"required": ["arg1"]}}]
        resources = [
            {"uri": "res://1", "name": "res1", "mimeType": "text/plain", "description": "desc"}
        ]
        prompts = [{"name": "prompt1", "description": "desc", "arguments": [{"name": "arg1"}]}]

        t_table = format_mcp_tools_table(tools)
        r_table = format_mcp_resources_table(resources)
        p_table = format_mcp_prompts_table(prompts)

        assert "Registered MCP Tools" in str(t_table.title)
        assert "Registered MCP Resources" in str(r_table.title)
        assert "Registered MCP Prompts" in str(p_table.title)
