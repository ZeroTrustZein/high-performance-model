"""Application factory for FinTech Market Telemetry MCP server."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from high_performance_model.indicators.series import (
    average_true_range,
    bollinger_bands,
    expected_shortfall,
    exponential_moving_average,
    macd,
    max_drawdown,
    momentum,
    rate_of_change,
    relative_strength_index,
    simple_moving_average,
    sortino_ratio,
    stochastic_oscillator,
    value_at_risk,
    volume_weighted_average_price,
)
from high_performance_model.protocol.models import (
    CallToolResult,
    GetPromptResult,
    PromptMessage,
    ReadResourceResult,
    TextContent,
)
from high_performance_model.protocol.server import MCPServer
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import Side


def create_fintech_mcp_server(
    buffer_capacity: int = 50_000,
    warm_up_tickers: Optional[List[str]] = None,
) -> MCPServer:
    """Instantiate and configure full-featured FinTech Market MCP server."""
    server = MCPServer(
        name="fintech-market-telemetry",
        version="0.1.0",
    )

    symbols = warm_up_tickers or ["AAPL", "MSFT", "NVDA", "GOOGL", "BTC/USD"]
    buffer = MarketDataBuffer(capacity=buffer_capacity)
    feed = SyntheticMarketFeed(symbols=symbols)

    # Warm up buffer with historical ticks and books
    for sym in symbols:
        ticks = feed.generate_history(symbol=sym, n_points=150)
        for t in ticks:
            buffer.push_tick(t)
        ob = feed.generate_order_book(symbol=sym, depth=10)
        buffer.push_order_book(ob)

    # --- Tool 1: get_market_quote ---
    @server.tool(
        name="get_market_quote",
        description="Retrieve latest market quote, trade price, spread, and VWAP for a ticker.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock or crypto symbol (e.g. AAPL, NVDA)"}
            },
            "required": ["symbol"],
        },
    )
    async def get_market_quote(symbol: str) -> CallToolResult:
        sym = symbol.upper()
        # Generate fresh tick if buffer empty or on demand
        tick = feed.generate_tick(symbol=sym)
        buffer.push_tick(tick)
        vwap = buffer.compute_vwap(symbol=sym)
        ob = buffer.get_latest_order_book(sym)

        payload = {
            "symbol": sym,
            "last_price": tick.price,
            "last_size": tick.size,
            "last_side": tick.side.value,
            "timestamp": tick.timestamp.isoformat(),
            "vwap": vwap,
            "spread": ob.spread if ob else None,
            "mid_price": ob.mid_price if ob else None,
        }
        return CallToolResult(
            content=[TextContent(text=json.dumps(payload, indent=2))]
        )

    # --- Tool 2: get_order_book_depth ---
    @server.tool(
        name="get_order_book_depth",
        description="Fetch Level-2 order book depth and book imbalance for a symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Market ticker symbol"},
                "depth": {"type": "integer", "description": "Number of price levels (1-20)", "default": 5},
            },
            "required": ["symbol"],
        },
    )
    async def get_order_book_depth(symbol: str, depth: int = 5) -> CallToolResult:
        sym = symbol.upper()
        ob = feed.generate_order_book(sym, depth=max(1, min(depth, 20)))
        buffer.push_order_book(ob)

        payload = {
            "symbol": sym,
            "best_bid": ob.best_bid,
            "best_ask": ob.best_ask,
            "spread": ob.spread,
            "imbalance": ob.order_book_imbalance,
            "bids": [lvl.model_dump() for lvl in ob.bids[:depth]],
            "asks": [lvl.model_dump() for lvl in ob.asks[:depth]],
            "timestamp": ob.timestamp.isoformat(),
        }
        return CallToolResult(
            content=[TextContent(text=json.dumps(payload, indent=2))]
        )

    # --- Tool 3: calculate_technical_indicators ---
    @server.tool(
        name="calculate_technical_indicators",
        description="Compute vectorized technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Momentum, ROC, Stochastic) on price series.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Market ticker symbol"},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators: 'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'vwap', 'momentum', 'roc', 'stochastic'",
                },
            },
            "required": ["symbol"],
        },
    )
    async def calculate_technical_indicators(
        symbol: str, indicators: Optional[List[str]] = None
    ) -> CallToolResult:
        sym = symbol.upper()
        prices = buffer.get_price_series(sym, limit=200)
        volumes = buffer.get_volume_series(sym, limit=200)
        if len(prices) < 20:
            for t in feed.generate_history(sym, 50):
                buffer.push_tick(t)
            prices = buffer.get_price_series(sym, limit=200)
            volumes = buffer.get_volume_series(sym, limit=200)

        requested = set(i.lower() for i in (indicators or ["sma", "ema", "rsi", "macd", "bollinger"]))
        out: Dict[str, Any] = {"symbol": sym, "last_price": float(prices[-1]) if len(prices) else None}

        if "sma" in requested:
            arr = simple_moving_average(prices, period=20)
            valid = arr[~np.isnan(arr)]
            out["sma_20"] = float(round(valid[-1], 4)) if len(valid) else None

        if "ema" in requested:
            arr = exponential_moving_average(prices, period=20)
            out["ema_20"] = float(round(arr[-1], 4)) if len(arr) else None

        if "rsi" in requested:
            arr = relative_strength_index(prices, period=14)
            valid = arr[~np.isnan(arr)]
            out["rsi_14"] = float(round(valid[-1], 2)) if len(valid) else None

        if "macd" in requested:
            m_line, s_line, hist = macd(prices)
            out["macd"] = {
                "macd_line": float(round(m_line[-1], 4)),
                "signal_line": float(round(s_line[-1], 4)),
                "histogram": float(round(hist[-1], 4)),
            }

        if "bollinger" in requested:
            bb = bollinger_bands(prices, period=20)
            out["bollinger"] = {
                "upper": float(round(bb["upper"][-1], 4)),
                "middle": float(round(bb["middle"][-1], 4)),
                "lower": float(round(bb["lower"][-1], 4)),
                "bandwidth": float(round(bb["bandwidth"][-1], 4)),
            }

        if "atr" in requested:
            highs = prices * 1.01
            lows = prices * 0.99
            atr_arr = average_true_range(highs, lows, prices, period=14)
            valid = atr_arr[~np.isnan(atr_arr)]
            out["atr_14"] = float(round(valid[-1], 4)) if len(valid) else None

        if "vwap" in requested:
            out["vwap"] = volume_weighted_average_price(prices, volumes)

        if "momentum" in requested:
            mom = momentum(prices, period=10)
            valid = mom[~np.isnan(mom)]
            out["momentum_10"] = float(round(valid[-1], 4)) if len(valid) else None

        if "roc" in requested:
            roc_arr = rate_of_change(prices, period=10)
            valid = roc_arr[~np.isnan(roc_arr)]
            out["roc_10"] = float(round(valid[-1], 4)) if len(valid) else None

        if "stochastic" in requested:
            highs = prices * 1.01
            lows = prices * 0.99
            k, d = stochastic_oscillator(highs, lows, prices, 14, 3, 3)
            valid_k = k[~np.isnan(k)]
            valid_d = d[~np.isnan(d)]
            out["stochastic"] = {
                "percent_k": float(round(valid_k[-1], 2)) if len(valid_k) else None,
                "percent_d": float(round(valid_d[-1], 2)) if len(valid_d) else None,
            }

        return CallToolResult(
            content=[TextContent(text=json.dumps(out, indent=2))]
        )

    # --- Tool 4: compute_risk_metrics ---
    @server.tool(
        name="compute_risk_metrics",
        description="Compute realized volatility, Sharpe ratio, Sortino ratio, max drawdown, and VaR/CVaR for a symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Market ticker symbol"}
            },
            "required": ["symbol"],
        },
    )
    async def compute_risk_metrics(symbol: str) -> CallToolResult:
        sym = symbol.upper()
        prices = buffer.get_price_series(sym, limit=150)
        if len(prices) < 30:
            for t in feed.generate_history(sym, 50):
                buffer.push_tick(t)
            prices = buffer.get_price_series(sym, limit=150)

        vol = buffer.compute_realized_volatility(sym)
        returns = np.diff(np.log(prices))
        sharpe = sharpe_ratio(returns, annualize=True)
        sortino = sortino_ratio(returns, annualize=True)
        mdd, _ = max_drawdown(prices)
        var_95 = value_at_risk(returns, confidence_level=0.95, method="historical")
        cvar_95 = expected_shortfall(returns, confidence_level=0.95)

        payload = {
            "symbol": sym,
            "realized_volatility_annualized": vol,
            "sharpe_ratio_estimate": sharpe,
            "sortino_ratio_estimate": sortino,
            "max_drawdown": mdd,
            "value_at_risk_95": var_95,
            "expected_shortfall_95": cvar_95,
            "sample_size_ticks": len(prices),
        }
        return CallToolResult(
            content=[TextContent(text=json.dumps(payload, indent=2))]
        )

    # --- Tool 5: get_volume_profile ---
    @server.tool(
        name="get_volume_profile",
        description="Fetch binned volume profile and Point of Control (POC) price level for a symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Market ticker symbol"},
                "bins": {"type": "integer", "description": "Number of price bins", "default": 10},
            },
            "required": ["symbol"],
        },
    )
    async def get_volume_profile(symbol: str, bins: int = 10) -> CallToolResult:
        sym = symbol.upper()
        profile = buffer.compute_volume_profile(sym, bins=max(2, min(bins, 50)))
        return CallToolResult(
            content=[TextContent(text=json.dumps({"symbol": sym, **profile}, indent=2))]
        )

    # --- Tool 6: estimate_market_impact ---
    @server.tool(
        name="estimate_market_impact",
        description="Estimate execution slippage and price impact against the Level-2 order book.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Market ticker symbol"},
                "size": {"type": "number", "description": "Order size to execute"},
                "side": {"type": "string", "description": "Order side ('buy' or 'sell')", "default": "buy"},
            },
            "required": ["symbol", "size"],
        },
    )
    async def estimate_market_impact(
        symbol: str, size: float, side: str = "buy"
    ) -> CallToolResult:
        sym = symbol.upper()
        exec_side = Side.SELL if side.lower() == "sell" else Side.BUY
        impact = buffer.estimate_market_impact(sym, size=size, side=exec_side)
        return CallToolResult(
            content=[TextContent(text=json.dumps(impact, indent=2))]
        )

    # --- Resources ---
    @server.resource(
        uri="market://telemetry/snapshot",
        name="Market Telemetry Snapshot",
        description="Real-time multi-ticker price, volume, and VWAP snapshot",
    )
    async def resource_telemetry_snapshot(uri: str) -> ReadResourceResult:
        snapshot = {}
        for s in symbols:
            tick = buffer.get_latest_tick(s)
            snapshot[s] = {
                "price": tick.price if tick else None,
                "vwap": buffer.compute_vwap(s),
            }
        return ReadResourceResult(
            contents=[{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(snapshot, indent=2),
            }]
        )

    @server.resource(
        uri="indicators://catalog",
        name="Technical Indicators Catalog",
        description="Documentation and parameter schema of all available technical indicators",
    )
    async def resource_indicators_catalog(uri: str) -> ReadResourceResult:
        catalog = {
            "indicators": [
                {"name": "sma", "full_name": "Simple Moving Average", "default_period": 20},
                {"name": "ema", "full_name": "Exponential Moving Average", "default_period": 20},
                {"name": "rsi", "full_name": "Relative Strength Index", "default_period": 14, "range": [0, 100]},
                {"name": "macd", "full_name": "Moving Average Convergence Divergence", "fast": 12, "slow": 26, "signal": 9},
                {"name": "bollinger", "full_name": "Bollinger Bands", "period": 20, "std_dev": 2.0},
                {"name": "atr", "full_name": "Average True Range", "default_period": 14},
                {"name": "vwap", "full_name": "Volume-Weighted Average Price"},
                {"name": "momentum", "full_name": "Price Momentum", "default_period": 10},
                {"name": "roc", "full_name": "Rate of Change", "default_period": 10},
                {"name": "stochastic", "full_name": "Stochastic Oscillator", "k_period": 14, "d_period": 3},
            ]
        }
        return ReadResourceResult(
            contents=[{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(catalog, indent=2),
            }]
        )

    # --- Prompts ---
    @server.prompt(
        name="analyze_market_structure",
        description="Prompt template for LLM analysis of market micro-structure and momentum.",
    )
    async def prompt_market_structure(symbol: str = "AAPL") -> GetPromptResult:
        sym = symbol.upper()
        return GetPromptResult(
            description=f"Analyze market order flow and technical indicators for {sym}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        text=(
                            f"Please query `get_market_quote`, `get_order_book_depth`, and "
                            f"`calculate_technical_indicators` for {sym}. Evaluate the bid-ask imbalance, "
                            f"VWAP deviation, and momentum signals to summarize current market structure."
                        )
                    ),
                )
            ],
        )

    return server
