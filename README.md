# High-Performance Model Context Protocol (MCP) Server for FinTech Telemetry

[![CI](https://github.com/zein/high-performance-model/actions/workflows/ci.yml/badge.svg)](https://github.com/zein/high-performance-model/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Protocol: MCP 2024-11-05](https://img.shields.io/badge/MCP-2024--11--05-success.svg)](https://modelcontextprotocol.io/)

A high-throughput, low-latency Model Context Protocol (MCP) server engineered for real-time FinTech market telemetry, Level-2 (L2) order book depth tracking, vectorized technical indicators, simulated order execution, and event-driven market alerting implemented in Python.

It provides both standard I/O (`stdio`) and Server-Sent Events HTTP (`sse`) transports, exposing 13 specialized financial tools, 5 real-time telemetry resources, and 2 prompt workflows to AI coding assistants (Claude Desktop, OpenCode, Cursor) and automated trading agents.

---

## Key Features

- **Standard Model Context Protocol (MCP) Core**:
  - Full JSON-RPC 2.0 implementation complying with MCP specification `2024-11-05`.
  - Non-blocking `stdio` transport for direct local client integration (Claude Desktop, OpenCode).
  - Production-ready `sse` (Server-Sent Events) HTTP transport for remote network connectivity.
  - Complete error handling with standard JSON-RPC codes and detailed diagnostics.

- **High-Throughput Ring Buffer & Market Telemetry**:
  - $O(1)$ fixed-capacity circular memory buffer (`MarketDataBuffer`) eliminating heap reallocation and GC pauses.
  - Real-time candlestick bar aggregation (`BarAggregator`) streaming 1s, 1m, 5m, 1h OHLCV bars.
  - Microstructural metrics: Bid-Ask Spread in basis points (bps), Volume-Weighted Microprice, Order Book Imbalance (OBI), Effective Spread, and Order Flow Imbalance (OFI).
  - Volume-at-Price Profile with Point of Control (POC) and multi-level depth market impact estimation.

- **Vectorized Technical Indicators & Quantitative Risk**:
  - Zero-loop NumPy vectorization (`sliding_window_view`, 1D discrete convolutions).
  - Technical analysis suite: SMA, EMA, RSI (Wilder's), MACD (fast/slow/signal/hist), Bollinger Bands, ATR, VWAP, Momentum, Rate of Change (ROC), Stochastic Oscillator, and Rolling Z-Score.
  - Institutional risk analytics: Annualized Realized Volatility, Sharpe Ratio, Sortino Ratio (downside semi-variance), Maximum Drawdown (MDD), Value at Risk (VaR 95%), and Expected Shortfall (CVaR 95%).

- **Simulated Execution & Portfolio Engine**:
  - Continuous order matching against Level-2 order book depth ladders with price-time priority.
  - Supports Market, Limit, and Stop Loss order types with partial fill tracking.
  - Configurable maker (1.0 bps) and taker (5.0 bps) fee structures.
  - Real-time portfolio state machine: position accumulation, short covering, long/short flipping, average entry price recomputation, cash accounting, and mark-to-market unrealized/realized PnL.

- **Real-Time Market Alerting Engine**:
  - Declarative condition rules: `PRICE_ABOVE`, `PRICE_BELOW`, `SPREAD_WIDER_THAN`, `IMBALANCE_SPIKE`, and `VOLUME_SPIKE`.
  - Observer subscription model with non-blocking error-isolated callback dispatch.
  - Support for both one-shot threshold triggers and recurring event counters.

- **Telemetry Persistence, Snapshots & Historical Replay**:
  - Multi-format serialization: streaming JSONL (ticks & bars), standardized CSV, and atomic JSON buffer snapshots.
  - Seekable asynchronous historical replay engine (`HistoricalReplayEngine`) with speed controls ($0.0\times$ instant to $N\times$ real-time), pause, resume, and buffer re-injection.

- **Rich Terminal CLI & MCP Inspector**:
  - Multi-subcommand CLI powered by Click and Rich (`high-performance-model` and `fintech-mcp`).
  - Interactive dashboards, live streaming tables, order submission, portfolio valuation, and direct terminal MCP invocation.

---

## Architecture Overview

```
                                [ External MCP Clients ]
                         (Claude Desktop, OpenCode, Terminal CLI)
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │              MCP Protocol & Transport Layer             │
               │  ┌───────────────────┬───────────────────┬───────────┐  │
               │  │  StdioTransport   │   SSETransport    │  Memory   │  │
               │  │  (Async Stdin)    │   (HTTP Starlette)│ Transport │  │
               │  └─────────┬─────────┴─────────┬─────────┴─────┬─────┘  │
               │            └───────────────────┼───────────────┘        │
               │                                ▼                        │
               │                    MCPServer (JSON-RPC 2.0)             │
               │        - Tools (13)   - Resources (5)   - Prompts (2)   │
               └────────────────────────────────┼────────────────────────┘
                                                │
                                                ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │                              Core FinTech Engine Subsystems                          │
 │                                                                                      │
 │   ┌───────────────────────────┐                 ┌────────────────────────────────┐   │
 │   │  Synthetic Telemetry Feed │                 │   Historical Replay Engine     │   │
 │   │  - Jump-Diffusion SDE     │                 │   - Variable Speed (0.0x - Nx) │   │
 │   │  - Multi-Asset Volatility │                 │   - Seek / Pause / Resume      │   │
 │   └─────────────┬─────────────┘                 └────────────────┬───────────────┘   │
 │                 │ Ticks & L2 Books                               │ Replayed Ticks    │
 │                 └───────────────────────┬────────────────────────┘                   │
 │                                         ▼                                            │
 │                         ┌───────────────────────────────┐                            │
 │                         │    MarketDataBuffer (Ring)    │                            │
 │                         │  - O(1) Fixed-Capacity Deque  │                            │
 │                         │  - Microstructural Analytics  │                            │
 │                         │  - Streaming Bar Aggregator   │                            │
 │                         └───────┬───────────────┬───────┘                            │
 │                                 │               │                                    │
 │                 ┌───────────────┘               └───────────────┐                    │
 │                 ▼                                               ▼                    │
 │  ┌─────────────────────────────┐                 ┌─────────────────────────────┐     │
 │  │ Vectorized Indicator Engine │                 │  Simulated Execution Engine │     │
 │  │ - NumPy Vectorization       │                 │  - L2 Depth Ladder Matching │     │
 │  │ - SMA, EMA, RSI, MACD       │                 │  - Multi-Level Slippage     │     │
 │  │ - Bollinger, ATR, VWAP      │                 │  - Position & PnL Tracking  │     │
 │  │ - VaR, CVaR, Sharpe, Sortino│                 │  - Maker/Taker Fee Modeling │     │
 │  └─────────────────────────────┘                 └─────────────────────────────┘     │
 │                 │                                               │                    │
 │                 └───────────────┬───────────────┬───────────────┘                    │
 │                                 ▼               ▼                                    │
 │                 ┌───────────────────────────────┐                                    │
 │                 │      Real-Time Alert Engine   │                                    │
 │                 │  - Price Breakouts / Bounds   │                                    │
 │                 │  - Volume Spikes / OFI Shifts │                                    │
 │                 │  - Observer Event Dispatcher  │                                    │
 │                 └───────────────┬───────────────┘                                    │
 │                                 ▼                                                    │
 │                 ┌───────────────────────────────┐                                    │
 │                 │ Storage & Persistence Layer   │                                    │
 │                 │  - Streaming JSONL Ticks/Bars │                                    │
 │                 │  - Standard CSV Time Series   │                                    │
 │                 │  - Atomic Buffer Snapshots    │                                    │
 │                 └───────────────────────────────┘                                    │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Standard Installation
```bash
git clone https://github.com/zein/high-performance-model.git
cd high-performance-model
pip install -e .
```

### Development Suite (Tests, Coverage, Linters, Type Checking)
```bash
pip install -e ".[dev]"
```

Both `high-performance-model` and `fintech-mcp` CLI entrypoints will be installed.

---

## MCP Client Configuration

### 1. Claude Desktop
Add the server configuration to your `claude_desktop_config.json`:

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

### 2. OpenCode
Add to your project's `.opencode/mcp.json` or global `~/.config/opencode/opencode.json`:

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

Or run over persistent HTTP SSE:
```bash
high-performance-model serve --transport sse --host 127.0.0.1 --port 8000
```
Then connect your client to `http://127.0.0.1:8000/sse`.

---

## MCP Primitives Catalog

### Tools (13)
| Tool Name | Parameters | Description |
|---|---|---|
| `get_market_quote` | `symbol: str` | Current top-of-book bid/ask quote, spread in bps, and microprice. |
| `get_order_book_depth` | `symbol: str`, `depth: int = 10` | L2 order book depth ladder with price, size, and notional values. |
| `calculate_technical_indicators` | `symbol: str`, `indicator: str`, `period: int` | Vectorized technical indicators (SMA, EMA, RSI, MACD, Bollinger, etc.). |
| `compute_risk_metrics` | `symbol: str`, `limit: int = 150` | Realized Volatility, Sharpe, Sortino, Max Drawdown, VaR 95%, CVaR 95%. |
| `get_volume_profile` | `symbol: str`, `bins: int = 10` | Volume distribution across price levels with Point of Control (POC). |
| `estimate_market_impact` | `symbol: str`, `size: float`, `side: str` | Simulates depth fill to estimate average execution price and slippage. |
| `submit_simulated_order` | `symbol: str`, `side: str`, `quantity: float`, `order_type: str`, `price: float` | Submits simulated market, limit, or stop order to matching engine. |
| `cancel_simulated_order` | `order_id: str` | Cancels an open simulated order. |
| `get_portfolio_state` | *None* | Returns total equity, cash balance, realized/unrealized PnL, positions. |
| `create_market_alert` | `symbol: str`, `alert_type: str`, `threshold: float`, `one_shot: bool` | Creates a real-time price, spread, or volume alert rule. |
| `list_market_alerts` | `symbol: Optional[str]` | Lists active monitoring rules and trigger event counts. |
| `export_buffer_snapshot` | `file_path: str` | Serializes current in-memory telemetry buffer to JSON snapshot file. |
| `import_buffer_snapshot` | `file_path: str` | Restores in-memory telemetry buffer state from JSON snapshot. |

### Resources (5)
| Resource URI | MIME Type | Description |
|---|---|---|
| `market://telemetry/snapshot` | `application/json` | Global telemetry state for all active symbols in memory. |
| `indicators://catalog` | `application/json` | Catalog of all supported technical indicators, math formulas, and defaults. |
| `portfolio://state` | `application/json` | Live portfolio equity, cash, unrealized/realized PnL, and positions. |
| `alerts://active` | `application/json` | Currently registered alert rules and historical trigger events. |
| `execution://history` | `application/json` | Audit log of simulated trade executions with fees and timestamps. |

### Prompts (2)
| Prompt Name | Arguments | Description |
|---|---|---|
| `analyze_market_structure` | `symbol: str` | Guided market structure prompt analyzing liquidity, microprice, and depth. |
| `evaluate_trading_opportunity` | `symbol: str`, `timeframe: str` | Multi-indicator synthesis prompt for trading opportunities with risk bounds. |

---

## Python Programmatic Quickstart

### 1. Ingest Telemetry & Compute Microstructural Analytics
```python
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed

feed = SyntheticMarketFeed(symbols=["AAPL", "NVDA", "BTC/USD"])
buffer = MarketDataBuffer(capacity=10_000)

# Push live ticks and Level-2 order book
for _ in range(100):
    buffer.push_tick(feed.generate_tick("AAPL"))
buffer.push_order_book(feed.generate_order_book("AAPL", depth=10))

# Microprice and spread basis points
quote = buffer.compute_top_quote("AAPL")
print(f"AAPL Bid: ${quote.bid} | Ask: ${quote.ask} | Spread: {quote.spread_bps:.2f} bps")

# Order book depth snapshot & imbalance
depth = buffer.compute_market_depth_snapshot("AAPL", depth=5)
print(f"Depth Imbalance: {depth.imbalance:+.4f} (Mid: ${depth.mid_price})")
```

### 2. Vectorized Technical Indicators & Risk
```python
from high_performance_model.indicators.series import (
    bollinger_bands,
    macd,
    relative_strength_index,
    sharpe_ratio,
    simple_moving_average,
)

prices = buffer.get_price_series("AAPL", limit=100)

sma = simple_moving_average(prices, period=20)
rsi = relative_strength_index(prices, period=14)
macd_line, signal_line, hist = macd(prices)
bands = bollinger_bands(prices, period=20, num_std=2.0)

print(f"Latest SMA-20: ${sma[-1]:.2f} | RSI-14: {rsi[-1]:.2f}")
print(f"Bollinger Upper: ${bands['upper'][-1]:.2f} | Lower: ${bands['lower'][-1]:.2f}")
```

### 3. Simulated Execution & Position Tracking
```python
from high_performance_model.execution.engine import ExecutionSimulator
from high_performance_model.types import OrderType, Side

engine = ExecutionSimulator()
engine.update_order_book(buffer.get_latest_order_book("AAPL"))

# Execute simulated market buy order
order = engine.submit_order(
    symbol="AAPL",
    side=Side.BUY,
    quantity=50.0,
    order_type=OrderType.MARKET,
)
print(
    f"Order {order.order_id} filled @ ${order.average_fill_price:.2f} (Fee: ${order.fee_paid:.4f})"
)

# Mark position to market
engine.on_tick(buffer.get_latest_tick("AAPL"))
pos = engine.portfolio.positions["AAPL"]
print(f"Position: {pos.quantity} shares | Unrealized PnL: ${pos.unrealized_pnl:+.2f}")
```

---

## CLI Command Reference

The command-line interface provides 10 core command trees:

| Command | Synopsis | Example |
|---|---|---|
| `serve` | Launch MCP server over `stdio` or `sse` | `high-performance-model serve --transport stdio` |
| `quote` | Real-time quote, spread, microprice, and depth | `high-performance-model quote NVDA --depth 5` |
| `indicators` | Calculate vectorized technical indicators | `high-performance-model indicators AAPL --indicator rsi` |
| `stream` | Live terminal stream of ticks or candlestick bars | `high-performance-model stream BTC/USD --count 10 --rate 2.0` |
| `replay` | Replay historical JSONL/CSV recordings | `high-performance-model replay ticks.jsonl --speed 2.0` |
| `snapshot` | Save, inspect, and restore buffer state snapshots | `high-performance-model snapshot save --file snap.json` |
| `alerts` | Manage threshold rules and monitor events live | `high-performance-model alerts add --symbol AAPL --threshold 220` |
| `exec` | Submit simulated orders and view portfolio PnL | `high-performance-model exec submit NVDA --side buy --quantity 20` |
| `mcp` | Directly inspect and invoke MCP tools & resources | `high-performance-model mcp call --tool get_market_quote --args '{"symbol":"AAPL"}'` |
| `benchmark` | Run high-throughput latency & ingestion stress tests | `high-performance-model benchmark --ticks 50000 --symbol NVDA` |

---

## Performance Benchmarks

Run the built-in benchmark to verify your environment:
```bash
high-performance-model benchmark --ticks 100000 --symbol NVDA
```

Representative results on standard modern hardware:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Benchmark Metric               ┃ Result             ┃
╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇
│ Total Ingested Ticks           │ 100,000            │
│ Ingestion Throughput           │ 315,240 ticks/sec  │
│ Ingestion Latency (P50)        │ 2.30 μs            │
│ Ingestion Latency (P95)        │ 7.80 μs            │
│ Ingestion Latency (P99)        │ 14.20 μs           │
│ SMA (10k points)               │ 0.12 ms            │
│ EMA (10k points)               │ 0.18 ms            │
│ RSI (10k points)               │ 0.25 ms            │
│ MACD (10k points)              │ 0.42 ms            │
│ Bollinger Bands (10k points)   │ 0.58 ms            │
│ Full Indicator Suite Total     │ 1.85 ms            │
└────────────────────────────────┴────────────────────┘
```

---

## Project Structure

```
high-performance-model/
├── docs/
│   ├── ARCHITECTURE.md          # System architecture, ASCII flow, math & algorithms
│   ├── API_REFERENCE.md         # Full programmatic reference for all modules & classes
│   └── USAGE_GUIDE.md           # Real-world recipes, configuration & agent integration
├── src/
│   └── high_performance_model/
│       ├── __init__.py          # Package version and top-level exports
│       ├── __main__.py          # Direct python -m execution entry point
│       ├── types.py             # Domain models, Pydantic contracts, and validators
│       ├── alerts/              # Real-time event and threshold monitoring
│       │   ├── engine.py        # AlertEngine observer dispatcher
│       │   └── models.py        # AlertRule, AlertEvent, and AlertType enums
│       ├── cli/                 # Command-line interface and formatters
│       │   ├── main.py          # Click CLI subcommands
│       │   └── formatters.py    # Rich panels and colorized tables
│       ├── execution/           # Simulated execution and portfolio engine
│       │   ├── engine.py        # ExecutionSimulator depth matching engine
│       │   └── models.py        # SimulatedOrder, Position, and Portfolio
│       ├── indicators/          # Vectorized technical analysis & risk analytics
│       │   └── series.py        # NumPy indicators (SMA, EMA, RSI, MACD, VaR, Sharpe)
│       ├── protocol/            # Model Context Protocol implementation
│       │   ├── jsonrpc.py       # JSON-RPC 2.0 framing and error utilities
│       │   ├── models.py        # MCP schema contracts (Tools, Resources, Prompts)
│       │   ├── server.py        # Core MCPServer dispatcher
│       │   ├── sse.py           # HTTP Server-Sent Events transport server
│       │   └── transports.py    # StdioTransport, MemoryTransport, and Base Transport
│       ├── server/              # FinTech MCP application assembly
│       │   └── app.py           # create_fintech_mcp_server application factory
│       ├── storage/             # Telemetry persistence and replay
│       │   ├── persistence.py   # JSONL, CSV, and snapshot disk handlers
│       │   └── replay.py        # HistoricalReplayEngine with speed controls
│       ├── telemetry/           # Market data ring buffers and synthetic feeds
│       │   ├── benchmark.py     # Telemetry benchmark runner
│       │   ├── buffer.py        # MarketDataBuffer & BarAggregator
│       │   └── generator.py     # SyntheticMarketFeed jump-diffusion generator
│       └── py.typed             # PEP 561 typing marker
├── tests/                       # Comprehensive pytest suite (208 tests, 95% coverage)
├── .github/workflows/ci.yml     # GitHub Actions cross-platform CI matrix
├── pyproject.toml               # Build system, dependencies, and entrypoints
├── requirements.txt             # Core runtime dependencies
├── requirements-dev.txt         # Testing and development tools
└── README.md                    # Project overview and documentation
```

---

## In-Depth Documentation

For advanced technical details, consult the dedicated documentation guides:
- [System Architecture & Technical Deep-Dive](docs/ARCHITECTURE.md): Mathematical formulations for microstructural metrics, stochastic jump-diffusion dynamics, vectorized indicator algorithms, and L2 matching engine mechanics.
- [Complete API Reference](docs/API_REFERENCE.md): Comprehensive signature, parameter, return type, and model schema reference for all classes and functions.
- [Practical Usage Guide & Recipes](docs/USAGE_GUIDE.md): Real-world cookbook covering MCP client setup, streaming aggregation, risk evaluation, simulated execution, alert monitoring, and historical replay.

---

## Verification & Testing

The test suite contains 208 comprehensive tests verifying all subsystems, edge cases, protocol compliance, and CLI commands:

```bash
# Run test suite with pytest
python -m pytest --cov=src/high_performance_model --cov-report=term-missing

# Run Ruff linter and formatter check
python -m ruff check src tests
python -m ruff format --check .

# Run static type checker
python -m mypy src
```

---

## License

This project is licensed under the [MIT License](LICENSE).
