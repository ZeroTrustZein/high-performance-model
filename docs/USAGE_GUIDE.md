# Practical Usage Guide: High-Performance FinTech MCP Server

This guide provides real-world recipes, configuration templates, and programmatic walkthroughs for integrating, configuring, and operating **`high-performance-model`** in production environments, AI agent workflows, and quantitative research pipelines.

---

## Table of Contents
1. [Recipe 1: Connecting with MCP Clients (Claude Desktop & OpenCode)](#recipe-1-connecting-with-mcp-clients-claude-desktop--opencode)
2. [Recipe 2: Real-Time Synthetic Streaming & Bar Aggregation](#recipe-2-real-time-synthetic-streaming--bar-aggregation)
3. [Recipe 3: Vectorized Technical Analysis & Risk Analytics](#recipe-3-vectorized-technical-analysis--risk-analytics)
4. [Recipe 4: Simulated Order Execution Against L2 Depth Ladders](#recipe-4-simulated-order-execution-against-l2-depth-ladders)
5. [Recipe 5: Configuring Event-Driven Market Alerts](#recipe-5-configuring-event-driven-market-alerts)
6. [Recipe 6: Historical Telemetry Recording, Snapshots & Replay](#recipe-6-historical-telemetry-recording-snapshots--replay)
7. [Recipe 7: Terminal CLI in Shell Scripting & Automated Workflows](#recipe-7-terminal-cli-in-shell-scripting--automated-workflows)
8. [Recipe 8: Extending the Server with Custom Tools & Resources](#recipe-8-extending-the-server-with-custom-tools--resources)

---

## Recipe 1: Connecting with MCP Clients (Claude Desktop & OpenCode)

The server supports both standard I/O (`stdio`) and Server-Sent Events HTTP (`sse`).

### Claude Desktop Integration (`stdio`)

Add the server to your `claude_desktop_config.json`:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "fintech-market-telemetry": {
      "command": "high-performance-model",
      "args": [
        "serve",
        "--transport", "stdio",
        "--capacity", "50000",
        "--tickers", "AAPL,MSFT,NVDA,BTC/USD"
      ]
    }
  }
}
```

### OpenCode Integration (`stdio` & `sse`)

Configure in your global `~/.config/opencode/opencode.json` or workspace `.opencode/mcp.json`:

```json
{
  "mcp": {
    "fintech": {
      "command": "high-performance-model",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

Or connect over persistent network HTTP SSE:
1. Start the SSE server in a terminal or daemon:
   ```bash
   high-performance-model serve --transport sse --host 127.0.0.1 --port 8000
   ```
2. Configure OpenCode or your HTTP client to connect to `http://127.0.0.1:8000/sse`.

---

## Recipe 2: Real-Time Synthetic Streaming & Bar Aggregation

Ingest live market ticks, track Level-2 depth, and build multi-timeframe candlestick bars automatically.

```python
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import BarTimeframe

# 1. Initialize buffer and synthetic generator
feed = SyntheticMarketFeed(symbols=["AAPL", "NVDA", "BTC/USD"], seed=42)
buffer = MarketDataBuffer(capacity=10_000)

# 2. Push continuous ticks and build 1-minute bars
for _ in range(500):
    tick = feed.generate_tick("AAPL")
    completed_bar = buffer.push_tick(tick)
    if completed_bar:
        print(
            f"Finalized Bar: O={completed_bar.open} H={completed_bar.high} "
            f"L={completed_bar.low} C={completed_bar.close} V={completed_bar.volume}"
        )

# 3. Ingest Level-2 order book depth
order_book = feed.generate_order_book("AAPL", depth=10, spread_bps=4.5)
buffer.push_order_book(order_book)

# 4. Query real-time microstructural metrics
top_quote = buffer.compute_top_quote("AAPL")
print(f"Top Quote: Bid={top_quote.bid} Ask={top_quote.ask} Spread BPS={top_quote.spread_bps:.2f}")

depth = buffer.compute_market_depth_snapshot("AAPL", depth=5)
print(f"Order Book Imbalance: {depth.imbalance:+.4f} (Mid: ${depth.mid_price})")

ofi = buffer.compute_order_flow_imbalance("AAPL", limit=100)
print(f"Net Order Flow: {ofi['net_flow']} (Buy: {ofi['buy_volume']}, Sell: {ofi['sell_volume']})")
```

---

## Recipe 3: Vectorized Technical Analysis & Risk Analytics

Compute moving averages, oscillators, volatility channels, and institutional risk metrics directly over NumPy series.

```python
import numpy as np
from high_performance_model.indicators.series import (
    average_true_range,
    bollinger_bands,
    expected_shortfall,
    macd,
    max_drawdown,
    realized_volatility,
    relative_strength_index,
    sharpe_ratio,
    simple_moving_average,
    value_at_risk,
)

# Sample historical price series
np.random.seed(42)
base_price = 100.0
returns = np.random.normal(0.0005, 0.015, size=500)
prices = base_price * np.cumprod(1.0 + returns)
highs = prices * (1.0 + np.random.uniform(0.001, 0.01, size=500))
lows = prices * (1.0 - np.random.uniform(0.001, 0.01, size=500))

# 1. Moving averages and oscillators
sma_50 = simple_moving_average(prices, period=50)
rsi_14 = relative_strength_index(prices, period=14)
macd_line, signal_line, hist = macd(prices)
bands = bollinger_bands(prices, period=20, num_std=2.0)
atr_14 = average_true_range(highs, lows, prices, period=14)

print(f"Latest Price: ${prices[-1]:.2f}")
print(f"Latest SMA-50: ${sma_50[-1]:.2f}")
print(f"Latest RSI-14: {rsi_14[-1]:.2f}")
print(f"Latest Bollinger Bands: Upper=${bands['upper'][-1]:.2f}, Lower=${bands['lower'][-1]:.2f}")
print(f"Latest ATR-14: ${atr_14[-1]:.2f}")

# 2. Institutional risk metrics
log_returns = np.diff(np.log(prices))
ann_vol = realized_volatility(prices, annualize=True)
sharpe = sharpe_ratio(log_returns, risk_free_rate=0.04, annualize=True)
mdd, _ = max_drawdown(prices)
var_95 = value_at_risk(log_returns, confidence_level=0.95, method="historical")
cvar_95 = expected_shortfall(log_returns, confidence_level=0.95)

print("\n--- Risk Analytics ---")
print(f"Annualized Realized Volatility: {ann_vol * 100:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
print(f"Maximum Drawdown: {mdd * 100:.2f}%")
print(f"1-Day 95% Value at Risk (VaR): {var_95 * 100:.2f}%")
print(f"1-Day 95% Expected Shortfall (CVaR): {cvar_95 * 100:.2f}%")
```

---

## Recipe 4: Simulated Order Execution Against L2 Depth Ladders

Execute simulated market and limit orders with multi-tier book slippage, fee accounting, and real-time position tracking.

```python
from high_performance_model.execution.engine import ExecutionSimulator
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import OrderType, Side

# 1. Initialize execution simulator with custom maker/taker fee rates
feed = SyntheticMarketFeed()
engine = ExecutionSimulator(maker_fee_bps=1.0, taker_fee_bps=5.0)

# 2. Cache latest order book
ob = feed.generate_order_book("NVDA", depth=10)
engine.update_order_book(ob)
print(f"NVDA Book: Best Bid=${ob.best_bid}, Best Ask=${ob.best_ask}")

# 3. Submit market buy order (crosses spread and walks ask ladder)
order = engine.submit_order(
    symbol="NVDA",
    side=Side.BUY,
    quantity=25.0,
    order_type=OrderType.MARKET,
)
print(f"Order Status: {order.status.value}")
print(f"Average Fill Price: ${order.average_fill_price:.2f}")
print(f"Fee Paid: ${order.fee_paid:.4f}")

# 4. Check updated portfolio state
portfolio = engine.get_portfolio_summary()
print(f"Cash Balance: ${portfolio['cash_balance']:,.2f}")
print(f"Total Equity: ${portfolio['total_portfolio_value']:,.2f}")
print("Positions:", portfolio["positions"])

# 5. Process subsequent market ticks to update mark-to-market and limit orders
new_tick = feed.generate_tick("NVDA")
engine.on_tick(new_tick)
pos = engine.portfolio.positions["NVDA"]
print(f"Unrealized PnL: ${pos.unrealized_pnl:+.2f} (Current Price: ${pos.current_price:.2f})")
```

---

## Recipe 5: Configuring Event-Driven Market Alerts

Monitor real-time prices, order book spreads, and volume spikes with observer subscriptions.

```python
from high_performance_model.alerts.engine import AlertEngine
from high_performance_model.alerts.models import AlertEvent, AlertType
from high_performance_model.telemetry.generator import SyntheticMarketFeed

# 1. Create alert engine and register rules
alerts = AlertEngine()

# Trigger when AAPL rises above $225.0
alerts.create_rule(
    symbol="AAPL",
    alert_type=AlertType.PRICE_ABOVE,
    threshold=225.0,
    one_shot=True,
    message="AAPL target breakout reached",
)

# Trigger on abnormal bid-ask spread wider than $0.25
alerts.create_rule(
    symbol="AAPL",
    alert_type=AlertType.SPREAD_WIDER_THAN,
    threshold=0.25,
    one_shot=False,
    message="Liquidity warning: spread blowout",
)


# 2. Register event listener
def on_alert_fired(event: AlertEvent):
    print(
        f"🚨 [ALERT FIRED] {event.message} | Value: {event.current_value} (Threshold: {event.threshold})"
    )


alerts.subscribe(on_alert_fired)

# 3. Stream ticks and books through alert engine
feed = SyntheticMarketFeed(symbols=["AAPL"])
for _ in range(50):
    tick = feed.generate_tick("AAPL")
    alerts.on_tick(tick)
    ob = feed.generate_order_book("AAPL")
    alerts.on_order_book(ob)
```

---

## Recipe 6: Historical Telemetry Recording, Snapshots & Replay

Record high-frequency market streams to disk and replay them at variable speeds for backtesting.

```python
import asyncio
from high_performance_model.storage.persistence import (
    MarketDataPersistence,
    load_buffer_snapshot,
    save_buffer_snapshot,
)
from high_performance_model.storage.replay import HistoricalReplayEngine
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed

feed = SyntheticMarketFeed(symbols=["BTC/USD"])
buffer = MarketDataBuffer(capacity=5_000)

# 1. Ingest telemetry into buffer
for _ in range(200):
    buffer.push_tick(feed.generate_tick("BTC/USD"))
buffer.push_order_book(feed.generate_order_book("BTC/USD"))

# 2. Persist snapshot to disk
snapshot_info = save_buffer_snapshot(buffer, "btc_snapshot.json")
print("Snapshot saved:", snapshot_info)

# 3. Restore snapshot into a clean buffer
restored_buffer = load_buffer_snapshot("btc_snapshot.json")
print(f"Restored symbols: {restored_buffer.symbols}")
print(f"Tick count: {len(restored_buffer.get_ticks('BTC/USD'))}")


# 4. Asynchronous variable-speed replay
async def run_replay():
    ticks = restored_buffer.get_ticks("BTC/USD")
    replay = HistoricalReplayEngine(ticks=ticks)

    # Stream ticks at 2x real-time speed
    async for tick in replay.stream_ticks(speed_multiplier=2.0, max_ticks=10):
        print(f"Replayed: {tick.symbol} @ ${tick.price} (seq={tick.sequence})")


asyncio.run(run_replay())
```

---

## Recipe 7: Terminal CLI in Shell Scripting & Automated Workflows

Use the CLI for shell scripting, diagnostics, and test harness integration.

### Launch Server Over SSE
```bash
high-performance-model serve --transport sse --host 127.0.0.1 --port 8000 &
```

### Inspect Real-Time Quotes as JSON
```bash
high-performance-model quote NVDA --json
```
Output:
```json
{
  "symbol": "NVDA",
  "bid": 128.75,
  "ask": 128.81,
  "bid_size": 140.0,
  "ask_size": 115.0,
  "last_price": 128.8,
  "spread": 0.06,
  "spread_bps": 4.659
}
```

### Compute Specific Technical Indicators
```bash
high-performance-model indicators AAPL --indicator rsi --period 14 --json
```

### Stream Live OHLCV Candlesticks as JSONL
```bash
high-performance-model stream BTC/USD --count 5 --mode bar --format jsonl
```

### Run Throughput Latency Benchmark
```bash
high-performance-model benchmark --ticks 50000 --symbol NVDA
```

---

## Recipe 8: Extending the Server with Custom Tools & Resources

Add custom algorithmic tools and resources to `MCPServer`.

```python
import asyncio
from high_performance_model.protocol.models import CallToolResult, TextContent
from high_performance_model.protocol.transports import StdioTransport
from high_performance_model.server.app import create_fintech_mcp_server

# 1. Create standard FinTech MCP server instance
server = create_fintech_mcp_server(warm_up_tickers=["AAPL", "NVDA"])


# 2. Register custom MCP tool
@server.tool(
    name="calculate_custom_spread_arbitrage",
    description="Calculate potential arbitrage spread between two synthetic assets.",
    input_schema={
        "type": "object",
        "properties": {
            "symbol_a": {"type": "string"},
            "symbol_b": {"type": "string"},
        },
        "required": ["symbol_a", "symbol_b"],
    },
)
async def calculate_custom_spread_arbitrage(symbol_a: str, symbol_b: str):
    buffer = getattr(server, "buffer")
    q_a = buffer.compute_top_quote(symbol_a)
    q_b = buffer.compute_top_quote(symbol_b)

    if not q_a or not q_b:
        return CallToolResult(
            content=[TextContent(text="Error: insufficient quote telemetry for symbols")],
            isError=True,
        )

    spread = q_a.mid_price - q_b.mid_price
    return CallToolResult(
        content=[TextContent(text=f"Pair spread ({symbol_a} - {symbol_b}): ${spread:.4f}")]
    )


# 3. Run server
if __name__ == "__main__":
    asyncio.run(server.run(StdioTransport()))
```
