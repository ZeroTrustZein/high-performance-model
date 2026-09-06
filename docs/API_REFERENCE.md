# API Reference: High-Performance FinTech MCP Server

Complete programmatic reference for all modules, classes, methods, models, schemas, and utility functions in **`high_performance_model`**.

---

## Table of Contents
1. [Domain Types & Models (`high_performance_model.types`)](#1-domain-types--models)
2. [Protocol & Transports (`high_performance_model.protocol`)](#2-protocol--transports)
3. [Market Telemetry Buffer & Generation (`high_performance_model.telemetry`)](#3-market-telemetry-buffer--generation)
4. [Vectorized Technical Indicators & Risk Suite (`high_performance_model.indicators`)](#4-vectorized-technical-indicators--risk-suite)
5. [Simulated Execution & Portfolio Engine (`high_performance_model.execution`)](#5-simulated-execution--portfolio-engine)
6. [Real-Time Alerts Engine (`high_performance_model.alerts`)](#6-real-time-alerts-engine)
7. [Storage, Persistence & Replay (`high_performance_model.storage`)](#7-storage-persistence--replay)
8. [MCP Server Factory & Primitives (`high_performance_model.server.app`)](#8-mcp-server-factory--primitives)
9. [Command-Line Interface (`high_performance_model.cli`)](#9-command-line-interface)

---

## 1. Domain Types & Models

Located in `high_performance_model.types`.

### Enumerations

#### `Side`
Order and trade execution side.
- `Side.BUY = "buy"`
- `Side.SELL = "sell"`

#### `OrderType`
Financial order execution types.
- `OrderType.MARKET = "market"`
- `OrderType.LIMIT = "limit"`
- `OrderType.STOP = "stop"`
- `OrderType.STOP_LIMIT = "stop_limit"`
- `OrderType.TRAILING_STOP = "trailing_stop"`

#### `TimeInForce`
Order duration and cancellation instructions.
- `TimeInForce.GTC = "GTC"` (Good 'Til Canceled)
- `TimeInForce.IOC = "IOC"` (Immediate or Cancel)
- `TimeInForce.FOK = "FOK"` (Fill or Kill)
- `TimeInForce.DAY = "DAY"` (Good for Day)
- `TimeInForce.GTD = "GTD"` (Good 'Til Date)

#### `AssetClass`
Financial market asset classifications.
- `AssetClass.EQUITY = "equity"`
- `AssetClass.CRYPTO = "crypto"`
- `AssetClass.FX = "fx"`
- `AssetClass.COMMODITY = "commodity"`
- `AssetClass.INDEX = "index"`
- `AssetClass.OPTION = "option"`
- `AssetClass.FUTURE = "future"`

#### `IndicatorType`
Supported technical analysis indicators.
- `IndicatorType.SMA = "sma"`
- `IndicatorType.EMA = "ema"`
- `IndicatorType.RSI = "rsi"`
- `IndicatorType.MACD = "macd"`
- `IndicatorType.BOLLINGER = "bollinger"`
- `IndicatorType.ATR = "atr"`
- `IndicatorType.VWAP = "vwap"`
- `IndicatorType.MOMENTUM = "momentum"`
- `IndicatorType.ROC = "roc"`
- `IndicatorType.STOCHASTIC = "stochastic"`

#### `BarTimeframe`
Candlestick aggregation intervals.
- `SEC_1 = "1s"`, `SEC_5 = "5s"`, `SEC_15 = "15s"`, `SEC_30 = "30s"`
- `MIN_1 = "1m"`, `MIN_5 = "5m"`, `MIN_15 = "15m"`, `MIN_30 = "30m"`
- `HOUR_1 = "1h"`, `HOUR_4 = "4h"`, `DAY_1 = "1d"`, `WEEK_1 = "1w"`

#### `MarketState`
- `PRE_MARKET = "pre_market"`, `OPEN = "open"`, `POST_MARKET = "post_market"`, `CLOSED = "closed"`, `HALTED = "halted"`

#### `LiquidityTier`
- `TIER_1 = "tier_1"`, `TIER_2 = "tier_2"`, `TIER_3 = "tier_3"`, `TIER_4 = "tier_4"`, `ILLIQUID = "illiquid"`

---

### Helper Functions

#### `validate_symbol(symbol: str) -> str`
Validates and normalizes ticker symbols to uppercase strings matching `^[A-Z0-9_\-\.\/]{1,20}$`.
- **Raises**: `TypeError` if not a string, `ValueError` if empty or invalid format.

#### `timeframe_to_seconds(timeframe: Union[BarTimeframe, str]) -> int`
Converts timeframe identifier into duration in seconds (e.g. `"1m"` $\to 60$, `"1h"` $\to 3600$).

#### `compute_spread_bps(bid: float, ask: float) -> float`
Computes the bid-ask spread in basis points relative to the midpoint price:
$$\text{bps} = \frac{\text{ask} - \text{bid}}{(\text{bid} + \text{ask}) / 2.0} \times 10,000$$

#### `compute_microprice(bid: float, ask: float, bid_qty: float, ask_qty: float) -> float`
Computes order book microprice weighted by opposite side depth:
$$P_{\text{micro}} = \frac{\text{bid} \cdot \text{ask\_qty} + \text{ask} \cdot \text{bid\_qty}}{\text{bid\_qty} + \text{ask\_qty}}$$

---

### Data Models

#### `MarketTick`
High-frequency market tick telemetry record.

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | `Symbol` | *Required* | Normalized ticker symbol. |
| `price` | `float` | *Required* | Executed trade price ($> 0.0$). |
| `size` | `float` | *Required* | Executed trade quantity ($\ge 0.0$). |
| `side` | `Side` | *Required* | Trade aggressor side (`buy` or `sell`). |
| `timestamp` | `datetime` | `now(UTC)` | ISO-8601 UTC timestamp. |
| `sequence` | `int` | `0` | Monotonic exchange sequence ID. |
| `exchange` | `Optional[str]` | `None` | Originating market venue. |
| `conditions` | `List[str]` | `[]` | Trade condition flags. |

- **`notional -> float`**: Property returning `price * size`.
- **`is_uptick(previous: Optional[MarketTick]) -> Optional[bool]`**: True if price increased, False if decreased, None if equal or no predecessor.
- **`to_dict() -> Dict[str, Any]`**: Serializes tick to dictionary.

#### `OrderBookLevel`
Single price-quantity depth level.
- Attributes: `price: float`, `quantity: float`, `orders_count: int = 1`.
- Properties: `notional -> float`.
- Methods: `to_tuple() -> Tuple[float, float, int]`, `to_dict() -> Dict[str, Any]`.

#### `OrderBook`
Level-2 order book depth snapshot.
- Attributes: `symbol: Symbol`, `bids: List[OrderBookLevel]`, `asks: List[OrderBookLevel]`, `timestamp: datetime`, `sequence: int`.
- Properties:
  - `best_bid -> Optional[float]`
  - `best_ask -> Optional[float]`
  - `spread -> Optional[float]`
  - `spread_bps -> Optional[float]`
  - `mid_price -> Optional[float]`
  - `microprice -> Optional[float]`
  - `order_book_imbalance -> float` (Ratio from -1.0 to +1.0)
  - `total_bid_volume -> float`, `total_ask_volume -> float`
  - `total_bid_notional -> float`, `total_ask_notional -> float`
  - `is_crossed -> bool` (True if best_bid $\ge$ best_ask)
- Methods:
  - `depth_summary(levels: int = 5) -> Dict[str, Any]`
  - `to_dict() -> Dict[str, Any]`

#### `Bar`
Aggregated OHLCV candlestick bar.
- Attributes: `symbol: Symbol`, `open: float`, `high: float`, `low: float`, `close: float`, `volume: float`, `timestamp: datetime`, `timeframe: BarTimeframe`, `vwap: Optional[float]`, `trades_count: int`, `turnover: Optional[float]`.
- Properties: `range -> float`, `body -> float`, `is_bullish -> bool`, `is_bearish -> bool`, `typical_price -> float`.
- Methods: `to_dict() -> Dict[str, Any]`.

#### `Quote`
Top-of-book market quote representation.
- Attributes: `symbol: Symbol`, `bid: float`, `ask: float`, `bid_size: float`, `ask_size: float`, `timestamp: datetime`, `last_price: Optional[float]`, `last_size: Optional[float]`.
- Properties: `spread -> float`, `mid_price -> float`, `spread_bps -> float`.
- Methods: `to_dict() -> Dict[str, Any]`.

#### `MarketDepthSnapshot`
Export contract for L2 order book depth.
- Attributes: `symbol: Symbol`, `bids: List[OrderBookLevel]`, `asks: List[OrderBookLevel]`, `spread: Optional[float]`, `mid_price: Optional[float]`, `imbalance: float`, `sequence: Optional[int]`, `timestamp: datetime`.
- Methods: `to_dict() -> Dict[str, Any]`.

#### `IndicatorConfig`
Configuration parameters for technical indicator computation.
- Attributes: `indicator: IndicatorType`, `period: int = 20`, `fast_period: int = 12`, `slow_period: int = 26`, `signal_period: int = 9`, `num_std: float = 2.0`, `extra_params: Dict[str, Any]`.

#### `TechnicalIndicatorResult`
Calculated technical indicator metrics.
- Attributes: `symbol: Symbol`, `indicator: str`, `timestamp: datetime`, `values: Dict[str, float]`, `metadata: Dict[str, Any]`.
- Methods: `get_value(key: str, default: Optional[float] = None) -> Optional[float]`, `to_dict() -> Dict[str, Any]`.

#### `RiskMetrics`
Quantitative portfolio and telemetry risk metrics.
- Attributes: `symbol: Symbol`, `realized_volatility: float`, `sharpe_ratio: Optional[float]`, `max_drawdown: float`, `value_at_risk_95: Optional[float]`, `expected_shortfall_95: Optional[float]`, `sample_size: int`, `timestamp: datetime`, `metadata: Dict[str, Any]`.
- Methods: `to_dict() -> Dict[str, Any]`.

#### `MarketTelemetrySummary`
Comprehensive single-asset market state snapshot.
- Attributes: `symbol: Symbol`, `last_price: float`, `last_size: float`, `last_side: Side`, `volume_24h: Optional[float]`, `vwap: Optional[float]`, `high_24h: Optional[float]`, `low_24h: Optional[float]`, `spread: Optional[float]`, `change_24h_pct: Optional[float]`, `market_state: MarketState`, `timestamp: datetime`.
- Methods: `to_dict() -> Dict[str, Any]`.

#### `TradeExecution`
Simulated trade fill record.
- Attributes: `trade_id: str`, `symbol: Symbol`, `price: float`, `size: float`, `side: Side`, `timestamp: datetime`, `order_type: OrderType`, `maker_order_id: Optional[str]`, `taker_order_id: Optional[str]`, `fee: float`.
- Properties: `notional -> float`.
- Methods: `to_dict() -> Dict[str, Any]`.

#### `BenchmarkRunResult`
Performance benchmark execution metrics.
- Attributes: `total_ticks: int`, `duration_seconds: float`, `ticks_per_second: float`, `latency_p50_us: float`, `latency_p95_us: float`, `latency_p99_us: float`, `buffer_utilization_pct: float`, `metadata: Dict[str, Any]`.
- Methods: `to_dict() -> Dict[str, Any]`.

---

## 2. Protocol & Transports

Located in `high_performance_model.protocol`.

### `MCPServer`
```python
from high_performance_model.protocol.server import MCPServer
```
Core Model Context Protocol JSON-RPC 2.0 server.

#### `__init__(name: str, version: str = "0.1.0")`
Creates a server instance with empty registries for tools, resources, and prompts.

#### `tool(name: Optional[str] = None, description: Optional[str] = None, input_schema: Optional[Dict[str, Any]] = None) -> Callable`
Decorator registering an async function as an MCP tool.

#### `resource(uri: str, name: Optional[str] = None, description: Optional[str] = None, mime_type: str = "application/json") -> Callable`
Decorator registering an async function as an MCP resource reader.

#### `prompt(name: Optional[str] = None, description: Optional[str] = None) -> Callable`
Decorator registering an async function as an MCP prompt generator.

#### `async handle_request(request: JsonRpcRequest) -> Optional[JsonRpcResponse]`
Dispatches JSON-RPC requests to registered handlers.

#### `async run(transport: Transport) -> None`
Executes server message loop on the provided transport until stream closes.

---

### Transports (`high_performance_model.protocol.transports`)

#### `Transport` (Abstract Base Class)
- `async read_message() -> Optional[str]`
- `async write_message(message: str) -> None`
- `async close() -> None`
- `async __aiter__() -> AsyncIterator[str]`

#### `StdioTransport(Transport)`
Asynchronous line-delimited JSON-RPC over `sys.stdin` and `sys.stdout`.

#### `MemoryTransport(Transport)`
Bidirectional in-memory queue transport for testing.
- `async feed_input(message: str) -> None`
- `async feed_eof() -> None`
- `async read_output(timeout: float = 2.0) -> Optional[str]`

#### `SSETransport` & `SSEServer` (`high_performance_model.protocol.sse`)
HTTP Server-Sent Events transport server.
- `async start() -> None`: Starts HTTP server listening on configured host and port.
- `async stop() -> None`: Shuts down HTTP server cleanly.

---

## 3. Market Telemetry Buffer & Generation

Located in `high_performance_model.telemetry`.

### `MarketDataBuffer`
```python
from high_performance_model.telemetry.buffer import MarketDataBuffer
```
Fixed-capacity ring buffer and real-time analytical calculation engine.

#### `__init__(capacity: int = 50_000)`
Initializes empty symbol partitions with circular `deque` storage.

#### Core Mutation Methods
- `push_tick(tick: MarketTick) -> Optional[Bar]`: Ingests tick in $O(1)$ and feeds active 1-minute `BarAggregator`. Emits completed `Bar` if interval closed.
- `push_order_book(order_book: OrderBook) -> None`: Updates active L2 book snapshot.
- `push_bar(bar: Bar) -> None`: Ingests aggregated OHLCV bar into rolling history.
- `clear() -> None`: Flushes all queues and state.

#### Retrieval & Inspection
- `symbols -> List[str]`: List of all tracked ticker symbols.
- `get_latest_tick(symbol: str) -> Optional[MarketTick]`
- `get_ticks(symbol: str, limit: int = 100) -> List[MarketTick]`
- `get_latest_order_book(symbol: str) -> Optional[OrderBook]`
- `get_bars(symbol: str, limit: int = 100) -> List[Bar]`
- `get_price_series(symbol: str, limit: int = 200) -> np.ndarray`: 1D NumPy array of prices.
- `get_volume_series(symbol: str, limit: int = 200) -> np.ndarray`: 1D NumPy array of sizes.

#### Microstructural Analytics
- `compute_vwap(symbol: str, limit: int = 200) -> Optional[float]`
- `compute_realized_volatility(symbol: str, limit: int = 100) -> Optional[float]`
- `compute_top_quote(symbol: str) -> Optional[Quote]`
- `compute_market_depth_snapshot(symbol: str, depth: int = 10) -> Optional[MarketDepthSnapshot]`
- `compute_effective_spread(symbol: str) -> Optional[float]`
- `compute_order_flow_imbalance(symbol: str, limit: int = 100) -> Dict[str, float]`
- `compute_volume_profile(symbol: str, bins: int = 10, limit: int = 500) -> Dict[str, Any]`
- `estimate_market_impact(symbol: str, size: float, side: Side) -> Dict[str, Any]`
- `compute_risk_metrics(symbol: str, limit: int = 150) -> Optional[RiskMetrics]`
- `compute_telemetry_summary(symbol: str, limit: int = 200) -> Optional[MarketTelemetrySummary]`
- `aggregate_and_store_bars(symbol: str, timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1) -> List[Bar]`

---

### `BarAggregator`
```python
from high_performance_model.telemetry.buffer import BarAggregator
```
Incremental streaming candlestick bar builder.
- `__init__(symbol: str, timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1)`
- `update(tick: MarketTick) -> Optional[Bar]`
- `flush() -> Optional[Bar]`

#### `aggregate_bars_from_ticks(ticks: List[MarketTick], timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1) -> List[Bar]`
Batch helper converting a raw sequence of ticks into sequential OHLCV bars.

---

### `SyntheticMarketFeed`
```python
from high_performance_model.telemetry.generator import SyntheticMarketFeed
```
Simulated low-latency market stream using jump-diffusion stochastic differential equations.
- `__init__(symbols: Optional[List[str]] = None, base_prices: Optional[Dict[str, float]] = None, seed: Optional[int] = 42)`
- `get_asset_class(symbol: str) -> AssetClass`
- `generate_tick(symbol: Optional[str] = None, timestamp: Optional[datetime] = None) -> MarketTick`
- `generate_order_book(symbol: Optional[str] = None, depth: int = 10, spread_bps: float = 5.0) -> OrderBook`
- `generate_bar(symbol: Optional[str] = None, timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1, timestamp: Optional[datetime] = None) -> Bar`
- `generate_history(symbol: str, n_points: int = 100, timeframe: Union[BarTimeframe, str] = BarTimeframe.MIN_1) -> List[MarketTick]`

---

### `run_telemetry_benchmark`
```python
from high_performance_model.telemetry.benchmark import run_telemetry_benchmark
```
- `run_telemetry_benchmark(ticks_count: int = 10_000, symbol: str = "NVDA", capacity: int = 100_000, run_indicators: bool = True, sample_latency_interval: int = 10) -> BenchmarkRunResult`

---

## 4. Vectorized Technical Indicators & Risk Suite

Located in `high_performance_model.indicators.series`.

All functions operate on 1D `numpy.ndarray` or `list[float]` inputs and return NumPy arrays or scalar floats.

### Trend & Moving Averages
- **`simple_moving_average(data, period: int = 20) -> np.ndarray`**
- **`exponential_moving_average(data, period: int = 20, alpha: Optional[float] = None) -> np.ndarray`**
- **`macd(data, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`**: Returns `(macd_line, signal_line, histogram)`.

### Momentum & Oscillators
- **`relative_strength_index(data, period: int = 14) -> np.ndarray`**
- **`momentum(data, period: int = 10) -> np.ndarray`**
- **`rate_of_change(data, period: int = 10) -> np.ndarray`**
- **`stochastic_oscillator(high, low, close, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Tuple[np.ndarray, np.ndarray]`**: Returns `(k_line, d_line)`.
- **`rolling_z_score(data, period: int = 20) -> np.ndarray`**

### Volatility & Volume
- **`bollinger_bands(data, period: int = 20, num_std: float = 2.0) -> Dict[str, np.ndarray]`**: Returns dict with keys `"upper"`, `"middle"`, `"lower"`, `"bandwidth"`.
- **`average_true_range(high, low, close, period: int = 14) -> np.ndarray`**
- **`volume_weighted_average_price(prices, volumes) -> float`**

### Quantitative Risk Analytics
- **`realized_volatility(data, period: Optional[int] = None, annualize: bool = True, trading_periods: float = 362880.0) -> float`**
- **`sharpe_ratio(returns, risk_free_rate: float = 0.0, annualize: bool = True, periods_per_year: float = 362880.0) -> float`**
- **`sortino_ratio(returns, risk_free_rate: float = 0.0, target_return: float = 0.0, annualize: bool = True, periods_per_year: float = 362880.0) -> float`**
- **`max_drawdown(data) -> Tuple[float, np.ndarray]`**: Returns `(max_drawdown_float, drawdown_series)`.
- **`value_at_risk(returns, confidence_level: float = 0.95, method: str = "historical") -> float`**
- **`expected_shortfall(returns, confidence_level: float = 0.95) -> float`**

---

## 5. Simulated Execution & Portfolio Engine

Located in `high_performance_model.execution`.

### `OrderStatus`
`PENDING`, `FILLED`, `PARTIALLY_FILLED`, `CANCELLED`, `REJECTED`.

### `SimulatedOrder`
Simulated order contract with `remaining_quantity`, `is_active`, and `to_dict()`.

### `Position`
Per-asset holding with `unrealized_pnl`, `market_value`, `update_market_price()`, and `apply_fill()`.

### `Portfolio`
Aggregated portfolio state.
- Attributes: `initial_cash: float = 100_000.0`, `cash_balance: float = 100_000.0`, `positions: Dict[str, Position]`, `total_realized_pnl: float`, `total_fees_paid: float`.
- Properties: `total_unrealized_pnl -> float`, `total_portfolio_value -> float`, `total_return_pct -> float`.
- Methods: `get_or_create_position(symbol: str) -> Position`, `to_dict() -> Dict[str, Any]`.

### `ExecutionSimulator`
Matching engine evaluating market orders against L2 depth and limit orders against incoming ticks.
- `__init__(portfolio: Optional[Portfolio] = None, maker_fee_bps: float = 1.0, taker_fee_bps: float = 5.0)`
- `update_order_book(order_book: OrderBook) -> None`
- `submit_order(symbol: str, side: Side, quantity: float, order_type: OrderType = OrderType.MARKET, price: Optional[float] = None, stop_price: Optional[float] = None, time_in_force: TimeInForce = TimeInForce.GTC, auto_match: bool = True) -> SimulatedOrder`
- `cancel_order(order_id: str) -> Optional[SimulatedOrder]`
- `match_market_order(order: SimulatedOrder, order_book: OrderBook) -> List[TradeExecution]`
- `on_tick(tick: MarketTick) -> List[TradeExecution]`
- `get_open_orders(symbol: Optional[str] = None) -> List[SimulatedOrder]`
- `get_order_history(symbol: Optional[str] = None) -> List[SimulatedOrder]`
- `get_trade_history(symbol: Optional[str] = None) -> List[TradeExecution]`
- `get_portfolio_summary() -> Dict[str, Any]`

---

## 6. Real-Time Alerts Engine

Located in `high_performance_model.alerts`.

### `AlertType`
`PRICE_ABOVE`, `PRICE_BELOW`, `SPREAD_WIDER_THAN`, `IMBALANCE_SPIKE`, `VOLUME_SPIKE`.

### `AlertRule`
Threshold condition model with `alert_id`, `symbol`, `alert_type`, `threshold`, `one_shot`, `triggered`, `trigger_count`, `message`.

### `AlertEvent`
Triggered event record with `event_id`, `alert_id`, `symbol`, `threshold`, `current_value`, `message`, `timestamp`.

### `AlertEngine`
- `create_rule(symbol: str, alert_type: AlertType, threshold: float, one_shot: bool = True, message: Optional[str] = None) -> AlertRule`
- `delete_rule(alert_id: str) -> bool`
- `get_active_rules(symbol: Optional[str] = None) -> List[AlertRule]`
- `get_event_history(symbol: Optional[str] = None) -> List[AlertEvent]`
- `subscribe(callback: Callable[[AlertEvent], None]) -> None`
- `unsubscribe(callback: Callable[[AlertEvent], None]) -> None`
- `on_tick(tick: MarketTick) -> List[AlertEvent]`
- `on_order_book(order_book: OrderBook) -> List[AlertEvent]`
- `clear() -> None`
- `to_dict() -> Dict[str, Any]`

---

## 7. Storage, Persistence & Replay

Located in `high_performance_model.storage`.

### Standalone Storage Functions
- `save_ticks_to_jsonl(ticks: List[MarketTick], file_path: Union[str, Path]) -> int`
- `load_ticks_from_jsonl(file_path: Union[str, Path], limit: Optional[int] = None) -> List[MarketTick]`
- `save_ticks_to_csv(ticks: List[MarketTick], file_path: Union[str, Path]) -> int`
- `load_ticks_from_csv(file_path: Union[str, Path], limit: Optional[int] = None) -> List[MarketTick]`
- `save_bars_to_jsonl(bars: List[Bar], file_path: Union[str, Path]) -> int`
- `load_bars_from_jsonl(file_path: Union[str, Path], limit: Optional[int] = None) -> List[Bar]`
- `save_buffer_snapshot(buffer: MarketDataBuffer, file_path: Union[str, Path]) -> Dict[str, Any]`
- `load_buffer_snapshot(file_path: Union[str, Path], buffer: Optional[MarketDataBuffer] = None) -> MarketDataBuffer`

### `MarketDataPersistence`
Repository managing disk files:
- `__init__(base_directory: Union[str, Path] = "data")`
- `get_symbol_path(symbol: str, ext: str = "jsonl") -> Path`
- `export_buffer(buffer: MarketDataBuffer, filename: str = "snapshot.json") -> Path`
- `import_buffer(filename: str = "snapshot.json") -> MarketDataBuffer`
- `append_ticks(symbol: str, ticks: List[MarketTick]) -> int`
- `read_ticks(symbol: str, limit: Optional[int] = None) -> List[MarketTick]`

### `HistoricalReplayEngine`
Asynchronous historical replay with inter-arrival delay simulation:
- `__init__(ticks: Optional[List[MarketTick]] = None, source_file: Optional[Union[str, Path]] = None)`
- `total_ticks -> int`, `cursor -> int`, `state -> ReplayState`
- `pause() -> None`
- `resume() -> None`
- `stop() -> None`
- `seek(index: int) -> int`
- `reset() -> None`
- `async stream_ticks(speed_multiplier: float = 1.0, max_ticks: Optional[int] = None, on_tick: Optional[Callable[[MarketTick], None]] = None) -> AsyncIterator[MarketTick]`
- `async replay_into_buffer(buffer: MarketDataBuffer, speed_multiplier: float = 0.0, max_ticks: Optional[int] = None) -> int`

---

## 8. MCP Server Factory & Primitives

Located in `high_performance_model.server.app`.

### `create_fintech_mcp_server`
```python
def create_fintech_mcp_server(
    buffer_capacity: int = 50_000,
    warm_up_tickers: Optional[List[str]] = None,
    execution_simulator: Optional[ExecutionSimulator] = None,
    alert_engine: Optional[AlertEngine] = None,
    persistence: Optional[MarketDataPersistence] = None,
) -> MCPServer: ...
```

Configures and returns a fully initialized `MCPServer` instance with pre-warmed telemetry buffers, simulated execution engine, alert engine, and persistence repository.

---

## 9. Command-Line Interface

Entry points: `high-performance-model` and `fintech-mcp`.

- `serve`: Options `--transport [stdio|sse]`, `--host`, `--port`, `--capacity`, `--tickers`, `--warmup`.
- `quote <symbol>`: Options `--depth`, `--json`.
- `indicators <symbol>`: Options `--indicator`, `--period`, `--json`.
- `stream <symbol>`: Options `--count`, `--rate`, `--mode [tick|bar]`, `--format [table|json|jsonl]`.
- `replay <file>`: Options `--speed`, `--limit`, `--symbol`.
- `snapshot [save|inspect|load]`: Snapshot management.
- `alerts [list|add|monitor]`: Alert rule creation and real-time monitoring.
- `exec [submit|portfolio|orders]`: Simulated trading and portfolio status.
- `mcp [tools|resources|prompts|call|read]`: MCP protocol inspector.
- `benchmark`: Options `--ticks`, `--symbol`, `--capacity`, `--json`.
