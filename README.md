# High-Performance Model Context Protocol (MCP) Server

A high-performance Model Context Protocol (MCP) server for real-time FinTech market telemetry and vectorized technical indicators implemented in Python.

## Features

- **Standard MCP Protocol**: Full JSON-RPC 2.0 implementation over `stdio` and `sse` transports conforming to Model Context Protocol standards.
- **FinTech Telemetry Engine**: Low-latency ring-buffered tick ingestion, order book depth tracking, and microstructural metric calculation.
- **Vectorized Technical Indicators**: High-throughput indicator computation (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, Stochastic, Momentum, ROC) powered by NumPy.
- **Simulated Execution & Portfolio Subsystem**: Simulated order matching with L2 depth ladder execution, fee modeling, position management, and PnL calculation.
- **Real-Time Alert Engine**: Event-driven alert evaluation on ticks and order books (price levels, volume spikes, wide spreads, book imbalances).
- **Historical Telemetry Persistence & Replay**: JSONL and CSV serialization, buffer snapshot restore, and variable-speed replay engine.
- **Developer CLI & MCP Inspector**: Rich terminal CLI for running servers, streaming live feeds, querying technical indicators, submitting orders, managing alerts, and directly inspecting MCP tools/resources.

## Quickstart

```bash
pip install -e .
high-performance-model --help
```

Both `high-performance-model` and `fintech-mcp` CLI entrypoints are available.

### Run Test Suite

```bash
pytest
```

---

## CLI Reference & Usage

### 1. Launch MCP Server (`serve`)

Run the MCP server over standard I/O (stdio) or Server-Sent Events (SSE HTTP):

```bash
# Stdio transport (for Claude Desktop, OpenCode, and local clients)
high-performance-model serve --transport stdio

# SSE HTTP transport on custom host and port
high-performance-model serve --transport sse --host 127.0.0.1 --port 8000 --capacity 100000 --tickers "AAPL,MSFT,NVDA,BTC/USD"
```

### 2. Market Quotes & Depth Snapshot (`quote`)

Inspect real-time quotes, microprices, bid-ask spreads, and L2 order book depth:

```bash
# Formatted terminal display
high-performance-model quote AAPL --depth 5

# JSON output for automated scripting
high-performance-model quote NVDA --json
```

### 3. Vectorized Technical Indicators (`indicators`)

Compute indicators directly on market series:

```bash
# Display full indicator suite (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, etc.)
high-performance-model indicators AAPL

# Specific indicator with customized period
high-performance-model indicators AAPL --indicator rsi --period 14

# Raw JSON output
high-performance-model indicators NVDA --indicator bollinger --json
```

### 4. Synthetic Market Stream (`stream`)

Stream simulated ticks or aggregated candlestick bars live to console or stdout:

```bash
# Stream 10 live ticks at 5 ticks/sec
high-performance-model stream AAPL --count 10 --rate 5.0 --mode tick

# Stream 1-second OHLCV bars in JSONL format
high-performance-model stream MSFT --count 5 --mode bar --format jsonl
```

### 5. Historical Telemetry Replay (`replay`)

Replay ticks from JSONL or CSV recordings into the engine with speed controls:

```bash
# Instant replay of historical file
high-performance-model replay market_data.jsonl --speed 0.0

# 2x real-time replay with symbol filter and limit
high-performance-model replay recordings.csv --speed 2.0 --symbol AAPL --limit 1000
```

### 6. Buffer State Snapshots (`snapshot`)

Save, inspect, and reload ring buffer states:

```bash
# Export active buffer state to JSON snapshot
high-performance-model snapshot save --file snapshot.json --tickers "AAPL,NVDA,BTC/USD" --ticks-count 200

# Inspect snapshot metadata and counts
high-performance-model snapshot inspect snapshot.json

# Restore buffer state from snapshot
high-performance-model snapshot load snapshot.json
```

### 7. Real-Time Market Alerts (`alerts`)

Define threshold conditions and monitor real-time trigger events:

```bash
# List active alert rules
high-performance-model alerts list

# Add new price breakout rule
high-performance-model alerts add --symbol AAPL --type price_above --threshold 220.0 --message "AAPL target breakout"

# Monitor ticks in real-time against active alert triggers
high-performance-model alerts monitor AAPL --count 20
```

### 8. Simulated Execution & Portfolio (`exec`)

Submit simulated market/limit orders and track real-time portfolio PnL:

```bash
# Submit a simulated market buy order
high-performance-model exec submit AAPL --side buy --quantity 50

# Submit a simulated limit sell order
high-performance-model exec submit NVDA --side sell --quantity 20 --type limit --price 250.0

# View portfolio equity, cash balance, and open positions
high-performance-model exec portfolio

# View active simulated orders
high-performance-model exec orders
```

### 9. MCP Tool & Resource Inspector (`mcp`)

Inspect and execute MCP primitives directly from the terminal without external clients:

```bash
# List all registered MCP tools
high-performance-model mcp tools

# List registered resources
high-performance-model mcp resources

# List prompt templates
high-performance-model mcp prompts

# Directly invoke an MCP tool
high-performance-model mcp call --tool get_market_quote --args '{"symbol": "AAPL"}'

# Read an MCP resource
high-performance-model mcp read --uri indicators://catalog
```

### 10. Performance Benchmark (`benchmark`)

Run high-throughput latency and ingestion benchmarks:

```bash
# Run benchmark on 100,000 ticks
high-performance-model benchmark --ticks 100000 --symbol NVDA

# Export benchmark metrics as JSON
high-performance-model benchmark --ticks 50000 --json
```
