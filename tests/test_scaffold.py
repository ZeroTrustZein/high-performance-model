"""Scaffold and foundational component tests."""

import asyncio
import json

import numpy as np
from click.testing import CliRunner

import high_performance_model
from high_performance_model.cli.main import cli
from high_performance_model.indicators.series import (
    average_true_range,
    bollinger_bands,
    exponential_moving_average,
    macd,
    relative_strength_index,
    simple_moving_average,
    volume_weighted_average_price,
)
from high_performance_model.protocol.jsonrpc import (
    make_error_response,
    make_success_response,
    parse_message,
    serialize_response,
)
from high_performance_model.protocol.models import JsonRpcRequest
from high_performance_model.protocol.server import MCPServer
from high_performance_model.protocol.transports import MemoryTransport
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import MarketTick, OrderBook, OrderBookLevel, Side


def test_package_metadata():
    """Verify package version and top-level exports."""
    assert high_performance_model.__version__ == "0.1.0"
    assert hasattr(high_performance_model, "MCPServer")
    assert hasattr(high_performance_model, "MarketDataBuffer")
    assert hasattr(high_performance_model, "create_fintech_mcp_server")
    assert hasattr(high_performance_model, "MonteCarloStressTester")
    assert hasattr(high_performance_model, "StressTestingSuite")
    assert hasattr(high_performance_model, "compute_comprehensive_risk_ratios")


def test_domain_types():
    """Verify domain type validation and methods."""
    tick = MarketTick(symbol="NVDA", price=125.5, size=10.0, side=Side.BUY)
    assert tick.price == 125.5
    d = tick.to_dict()
    assert d["symbol"] == "NVDA"
    assert d["side"] == "buy"

    bids = [OrderBookLevel(price=100.0, quantity=5.0), OrderBookLevel(price=99.0, quantity=10.0)]
    asks = [OrderBookLevel(price=101.0, quantity=3.0), OrderBookLevel(price=102.0, quantity=8.0)]
    ob = OrderBook(symbol="NVDA", bids=bids, asks=asks)

    assert ob.best_bid == 100.0
    assert ob.best_ask == 101.0
    assert ob.spread == 1.0
    assert ob.mid_price == 100.5
    assert isinstance(ob.order_book_imbalance, float)


def test_telemetry_buffer(synthetic_feed: SyntheticMarketFeed):
    """Verify market data buffer operations and metrics."""
    buf = MarketDataBuffer(capacity=100)
    for _ in range(50):
        buf.push_tick(synthetic_feed.generate_tick("MSFT"))

    ticks = buf.get_ticks("MSFT", limit=10)
    assert len(ticks) == 10
    prices = buf.get_price_series("MSFT", limit=50)
    assert len(prices) == 50

    vwap = buf.compute_vwap("MSFT")
    assert vwap is not None and vwap > 0

    vol = buf.compute_realized_volatility("MSFT")
    assert vol is not None and vol >= 0


def test_synthetic_feed(synthetic_feed: SyntheticMarketFeed):
    """Verify synthetic feed produces valid data."""
    tick = synthetic_feed.generate_tick("AAPL")
    assert tick.symbol == "AAPL"
    assert tick.price > 0

    ob = synthetic_feed.generate_order_book("AAPL", depth=5)
    assert len(ob.bids) == 5
    assert len(ob.asks) == 5
    assert ob.best_ask is not None and ob.best_bid is not None
    assert ob.best_ask > ob.best_bid

    bars = synthetic_feed.generate_bars("AAPL", n_bars=10)
    assert len(bars) == 10
    assert bars[0].high >= bars[0].low


def test_technical_indicators(sample_prices: np.ndarray):
    """Verify vectorized technical indicator calculation."""
    sma = simple_moving_average(sample_prices, period=20)
    assert len(sma) == len(sample_prices)
    assert np.isnan(sma[0])
    assert not np.isnan(sma[25])

    ema = exponential_moving_average(sample_prices, period=20)
    assert len(ema) == len(sample_prices)
    assert not np.isnan(ema[-1])

    rsi = relative_strength_index(sample_prices, period=14)
    assert len(rsi) == len(sample_prices)
    assert 0 <= rsi[-1] <= 100

    m, s, h = macd(sample_prices)
    assert len(m) == len(sample_prices)
    assert len(s) == len(sample_prices)
    assert len(h) == len(sample_prices)

    bb = bollinger_bands(sample_prices, period=20)
    assert "upper" in bb and "lower" in bb and "middle" in bb
    assert bb["upper"][-1] >= bb["middle"][-1] >= bb["lower"][-1]

    highs = sample_prices * 1.02
    lows = sample_prices * 0.98
    atr = average_true_range(highs, lows, sample_prices, period=14)
    assert len(atr) == len(sample_prices)
    assert not np.isnan(atr[-1])

    volumes = np.ones_like(sample_prices) * 100.0
    vwap = volume_weighted_average_price(sample_prices, volumes)
    assert vwap > 0


def test_jsonrpc_encoding():
    """Verify JSON-RPC message serialization and parsing."""
    req_line = '{"jsonrpc": "2.0", "id": 1, "method": "ping"}'
    req, err = parse_message(req_line)
    assert err is None
    assert req is not None
    assert req.id == 1
    assert req.method == "ping"

    success_resp = make_success_response(1, {"status": "ok"})
    serialized = serialize_response(success_resp)
    assert '"result": {"status": "ok"}' in serialized

    err_resp = make_error_response(2, -32600, "Invalid Request")
    assert err_resp.error is not None
    assert err_resp.error.code == -32600


def test_mcp_server_protocol_lifecycle(mcp_server: MCPServer):
    """Verify MCP protocol initialize, tools/list, and tool execution."""

    async def _test():
        # 1. Initialize
        init_req = JsonRpcRequest(
            id=1,
            method="initialize",
            params={"clientInfo": {"name": "test-client", "version": "1.0.0"}},
        )
        init_resp = await mcp_server.handle_request(init_req)
        assert init_resp is not None
        assert init_resp.id == 1
        assert init_resp.result is not None
        assert "protocolVersion" in init_resp.result

        # 2. List tools
        tools_req = JsonRpcRequest(id=2, method="tools/list")
        tools_resp = await mcp_server.handle_request(tools_req)
        assert tools_resp is not None and tools_resp.result is not None
        tool_names = [t["name"] for t in tools_resp.result["tools"]]
        assert "get_market_quote" in tool_names
        assert "get_order_book_depth" in tool_names
        assert "calculate_technical_indicators" in tool_names

        # 3. Call tool: get_market_quote
        call_req = JsonRpcRequest(
            id=3,
            method="tools/call",
            params={"name": "get_market_quote", "arguments": {"symbol": "AAPL"}},
        )
        call_resp = await mcp_server.handle_request(call_req)
        assert call_resp is not None and call_resp.result is not None
        assert call_resp.id == 3
        result_data = json.loads(call_resp.result["content"][0]["text"])
        assert result_data["symbol"] == "AAPL"
        assert "last_price" in result_data

        # 4. Resources list and read
        res_list_req = JsonRpcRequest(id=4, method="resources/list")
        res_list_resp = await mcp_server.handle_request(res_list_req)
        assert res_list_resp is not None and res_list_resp.result is not None
        assert len(res_list_resp.result["resources"]) >= 2

        res_read_req = JsonRpcRequest(
            id=5,
            method="resources/read",
            params={"uri": "indicators://catalog"},
        )
        res_read_resp = await mcp_server.handle_request(res_read_req)
        assert res_read_resp is not None and res_read_resp.result is not None
        catalog = json.loads(res_read_resp.result["contents"][0]["text"])
        assert "indicators" in catalog

        # 5. Prompts list and get
        prompt_list_req = JsonRpcRequest(id=6, method="prompts/list")
        prompt_list_resp = await mcp_server.handle_request(prompt_list_req)
        assert prompt_list_resp is not None and prompt_list_resp.result is not None
        assert len(prompt_list_resp.result["prompts"]) >= 1

        prompt_get_req = JsonRpcRequest(
            id=7,
            method="prompts/get",
            params={"name": "analyze_market_structure", "arguments": {"symbol": "AAPL"}},
        )
        prompt_get_resp = await mcp_server.handle_request(prompt_get_req)
        assert prompt_get_resp is not None and prompt_get_resp.result is not None
        assert len(prompt_get_resp.result["messages"]) == 1

    asyncio.run(_test())


def test_memory_transport(mcp_server: MCPServer):
    """Verify bidirectional communication across MemoryTransport."""

    async def _test():
        transport = MemoryTransport()
        await transport.feed_input('{"jsonrpc": "2.0", "id": 100, "method": "ping"}')
        await transport.feed_eof()

        await mcp_server.run(transport)
        out = await transport.read_output(timeout=1.0)
        assert out is not None
        data = json.loads(out)
        assert data["id"] == 100
        assert data["result"] == {}

    asyncio.run(_test())


def test_cli_execution():
    """Verify CLI help, quote, and benchmark commands."""
    runner = CliRunner()

    result_help = runner.invoke(cli, ["--help"])
    assert result_help.exit_code == 0
    assert "High-performance Model Context Protocol" in result_help.output

    result_quote = runner.invoke(cli, ["quote", "AAPL"])
    assert result_quote.exit_code == 0
    assert "Market Quote: AAPL" in result_quote.output

    result_bench = runner.invoke(cli, ["benchmark", "--ticks", "1000"])
    assert result_bench.exit_code == 0
    assert "Performance Benchmark Results" in result_bench.output
