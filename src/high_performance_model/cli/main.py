"""Command Line Interface for high-performance FinTech MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from high_performance_model.alerts.engine import AlertEngine
from high_performance_model.alerts.models import AlertType
from high_performance_model.cli.formatters import (
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
from high_performance_model.execution.engine import ExecutionSimulator
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
from high_performance_model.protocol.jsonrpc import JsonRpcRequest
from high_performance_model.protocol.sse import SSEServer
from high_performance_model.protocol.transports import StdioTransport
from high_performance_model.server.app import create_fintech_mcp_server
from high_performance_model.storage.persistence import (
    load_buffer_snapshot,
    save_buffer_snapshot,
)
from high_performance_model.storage.replay import HistoricalReplayEngine
from high_performance_model.telemetry.benchmark import run_telemetry_benchmark
from high_performance_model.telemetry.generator import SyntheticMarketFeed
from high_performance_model.types import (
    OrderType,
    Side,
    validate_symbol,
)

console = Console()
logger = logging.getLogger("high_performance_model.cli")


@click.group()
@click.version_option(version="0.1.0", prog_name="high-performance-model")
def cli() -> None:
    """High-performance Model Context Protocol server for FinTech telemetry."""
    pass


# ---------------------------------------------------------------------------
# Command: serve
# ---------------------------------------------------------------------------
@cli.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="MCP transport protocol (stdio or sse)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="HTTP host to bind for SSE transport",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="HTTP port to bind for SSE transport",
)
@click.option(
    "--capacity",
    default=50_000,
    type=int,
    show_default=True,
    help="Telemetry ring buffer capacity",
)
@click.option(
    "--tickers",
    "-s",
    default="AAPL,MSFT,NVDA,GOOGL,BTC/USD",
    show_default=True,
    help="Comma-separated ticker symbols to initialize",
)
@click.option(
    "--warmup",
    default=150,
    type=int,
    show_default=True,
    help="Warm-up ticks per symbol",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Application log verbosity",
)
def serve(
    transport: str,
    host: str,
    port: int,
    capacity: int,
    tickers: str,
    warmup: int,
    log_level: str,
) -> None:
    """Launch Model Context Protocol server."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sym_list = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    server = create_fintech_mcp_server(
        buffer_capacity=capacity,
        warm_up_tickers=sym_list,
    )

    if transport.lower() == "stdio":
        stdio_transport = StdioTransport()
        try:
            asyncio.run(server.run(stdio_transport))
        except KeyboardInterrupt:
            sys.exit(0)
    elif transport.lower() == "sse":
        console.print(
            Panel(
                f"[bold green]Starting FinTech MCP Server (SSE)[/bold green]\n"
                f"Listening on [bold cyan]http://{host}:{port}/sse[/bold cyan]\n"
                f"POST messages to [bold cyan]http://{host}:{port}/messages?sessionId=<id>[/bold cyan]\n"
                f"Active Symbols: [yellow]{', '.join(sym_list)}[/yellow]\n"
                f"Ring Buffer Capacity: [yellow]{capacity:,}[/yellow]",
                title="FinTech MCP SSE Runtime",
                border_style="green",
            )
        )
        sse_server = SSEServer(mcp_server=server, host=host, port=port)

        async def _run_sse() -> None:
            await sse_server.start()
            try:
                # Keep running until interrupted
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await sse_server.stop()

        try:
            asyncio.run(_run_sse())
        except KeyboardInterrupt:
            console.print("[yellow]Shutting down SSE server...[/yellow]")
            sys.exit(0)


# ---------------------------------------------------------------------------
# Command: quote
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("symbol", default="AAPL")
@click.option(
    "--depth",
    "-d",
    default=5,
    type=int,
    show_default=True,
    help="Order book depth levels to display",
)
@click.option(
    "--json-output",
    "-j",
    "--json",
    "json_mode",
    is_flag=True,
    help="Output quote and depth as raw JSON",
)
def quote(symbol: str, depth: int, json_mode: bool) -> None:
    """Fetch current market quote and order book snapshot."""
    try:
        sym = validate_symbol(symbol)
    except (ValueError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    feed = SyntheticMarketFeed(symbols=[sym])
    tick = feed.generate_tick(sym)
    ob = feed.generate_order_book(sym, depth=max(depth, 5))

    if json_mode:
        payload = {
            "tick": tick.to_dict(),
            "order_book": {
                "symbol": ob.symbol,
                "best_bid": ob.best_bid,
                "best_ask": ob.best_ask,
                "spread": ob.spread,
                "spread_bps": ob.spread_bps,
                "microprice": ob.microprice,
                "imbalance": ob.order_book_imbalance,
                "bids": [b.to_dict() for b in ob.bids[:depth]],
                "asks": [a.to_dict() for a in ob.asks[:depth]],
            },
        }
        click.echo(json.dumps(payload, indent=2))
        return

    console.print(format_quote_panel(tick, ob))

    # Preserves "Market Quote: {symbol}" title for compatibility
    quote_table = Table(title=f"Market Quote: {sym}")
    quote_table.add_column("Field", style="cyan", no_wrap=True)
    quote_table.add_column("Value", style="green")

    quote_table.add_row("Last Price", f"${tick.price:.2f}")
    quote_table.add_row("Last Size", f"{tick.size:.2f}")
    quote_table.add_row("Side", tick.side.value.upper())
    quote_table.add_row("Best Bid", f"${ob.best_bid:.2f}" if ob.best_bid else "N/A")
    quote_table.add_row("Best Ask", f"${ob.best_ask:.2f}" if ob.best_ask else "N/A")
    quote_table.add_row("Spread", f"${ob.spread:.4f}" if ob.spread else "N/A")
    quote_table.add_row("Imbalance", f"{ob.order_book_imbalance:.4f}")

    console.print(quote_table)
    console.print(format_order_book_table(ob, depth=depth))


# ---------------------------------------------------------------------------
# Command: indicators
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("symbol", default="AAPL")
@click.option(
    "--indicator",
    "-i",
    type=click.Choice(
        [
            "all",
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
        ],
        case_sensitive=False,
    ),
    default="all",
    show_default=True,
    help="Specific indicator to compute or 'all'",
)
@click.option(
    "--period",
    "-p",
    default=20,
    type=int,
    show_default=True,
    help="Indicator lookback period",
)
@click.option(
    "--bars",
    "-n",
    default=100,
    type=int,
    show_default=True,
    help="Number of historical bars to generate",
)
@click.option(
    "--json-output",
    "-j",
    "--json",
    "json_mode",
    is_flag=True,
    help="Output indicators as JSON",
)
def indicators(
    symbol: str,
    indicator: str,
    period: int,
    bars: int,
    json_mode: bool,
) -> None:
    """Compute vectorized technical indicators on market data."""
    try:
        sym = validate_symbol(symbol)
    except (ValueError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    feed = SyntheticMarketFeed(symbols=[sym])
    hist_bars = feed.generate_bars(sym, n_bars=max(bars, period + 35))

    closes = np.array([b.close for b in hist_bars], dtype=np.float64)
    highs = np.array([b.high for b in hist_bars], dtype=np.float64)
    lows = np.array([b.low for b in hist_bars], dtype=np.float64)
    volumes = np.array([b.volume for b in hist_bars], dtype=np.float64)
    last_price = closes[-1]

    results: Dict[str, Dict[str, Any]] = {}
    chosen = indicator.lower()

    if chosen in {"all", "sma"}:
        vals = simple_moving_average(closes, period=period)
        val = vals[-1]
        sig = "Bullish (Above)" if last_price > val else "Bearish (Below)"
        results["SMA"] = {
            "value": round(float(val), 2),
            "value_str": f"${val:.2f}",
            "params": f"period={period}",
            "signal": sig,
        }

    if chosen in {"all", "ema"}:
        vals = exponential_moving_average(closes, period=period)
        val = vals[-1]
        sig = "Bullish (Above)" if last_price > val else "Bearish (Below)"
        results["EMA"] = {
            "value": round(float(val), 2),
            "value_str": f"${val:.2f}",
            "params": f"period={period}",
            "signal": sig,
        }

    if chosen in {"all", "rsi"}:
        vals = relative_strength_index(closes, period=14)
        val = vals[-1]
        if val >= 70:
            sig = "Overbought (>=70)"
        elif val <= 30:
            sig = "Oversold (<=30)"
        else:
            sig = "Neutral (30-70)"
        results["RSI"] = {
            "value": round(float(val), 2),
            "value_str": f"{val:.2f}",
            "params": "period=14",
            "signal": sig,
        }

    if chosen in {"all", "macd"}:
        macd_line, signal_line, hist = macd(closes)
        m_val, s_val, h_val = macd_line[-1], signal_line[-1], hist[-1]
        sig = "Bullish Crossover" if h_val > 0 else "Bearish Divergence"
        results["MACD"] = {
            "value": {
                "macd": round(float(m_val), 3),
                "signal": round(float(s_val), 3),
                "hist": round(float(h_val), 3),
            },
            "value_str": f"M:{m_val:+.2f} S:{s_val:+.2f} H:{h_val:+.2f}",
            "params": "fast=12, slow=26, sig=9",
            "signal": sig,
        }

    if chosen in {"all", "bollinger"}:
        bb = bollinger_bands(closes, period=period, num_std=2.0)
        up, mid, low = bb["upper"][-1], bb["middle"][-1], bb["lower"][-1]
        if last_price >= up:
            sig = "Overbought (Above Upper)"
        elif last_price <= low:
            sig = "Oversold (Below Lower)"
        else:
            sig = "In Range (Mid Band)"
        results["Bollinger Bands"] = {
            "value": {
                "upper": round(float(up), 2),
                "middle": round(float(mid), 2),
                "lower": round(float(low), 2),
            },
            "value_str": f"U:${up:.2f} M:${mid:.2f} L:${low:.2f}",
            "params": f"period={period}, std=2.0",
            "signal": sig,
        }

    if chosen in {"all", "atr"}:
        vals = average_true_range(highs, lows, closes, period=14)
        val = vals[-1]
        results["ATR"] = {
            "value": round(float(val), 2),
            "value_str": f"${val:.2f}",
            "params": "period=14",
            "signal": "Volatility Range",
        }

    if chosen in {"all", "vwap"}:
        val = volume_weighted_average_price(closes, volumes)
        sig = "Above VWAP (Bullish)" if last_price >= val else "Below VWAP (Bearish)"
        results["VWAP"] = {
            "value": round(float(val), 2),
            "value_str": f"${val:.2f}",
            "params": "cumulative",
            "signal": sig,
        }

    if chosen in {"all", "momentum"}:
        vals = momentum(closes, period=10)
        val = vals[-1]
        sig = "Positive Momentum" if val > 0 else "Negative Momentum"
        results["Momentum"] = {
            "value": round(float(val), 2),
            "value_str": f"{val:+.2f}",
            "params": "period=10",
            "signal": sig,
        }

    if chosen in {"all", "roc"}:
        vals = rate_of_change(closes, period=10)
        val = vals[-1]
        sig = "Accelerating Up" if val > 0 else "Decelerating Down"
        results["ROC"] = {
            "value": round(float(val), 2),
            "value_str": f"{val:+.2f}%",
            "params": "period=10",
            "signal": sig,
        }

    if chosen in {"all", "stochastic"}:
        k_line, d_line = stochastic_oscillator(highs, lows, closes, k_period=14, d_period=3)
        k_val, d_val = k_line[-1], d_line[-1]
        sig = "Overbought" if k_val >= 80 else ("Oversold" if k_val <= 20 else "Neutral")
        results["Stochastic"] = {
            "value": {"k": round(float(k_val), 2), "d": round(float(d_val), 2)},
            "value_str": f"%K:{k_val:.1f} %D:{d_val:.1f}",
            "params": "k=14, d=3",
            "signal": sig,
        }

    if json_mode:
        click.echo(
            json.dumps(
                {
                    "symbol": sym,
                    "last_price": last_price,
                    "indicators": results,
                },
                indent=2,
            )
        )
        return

    console.print(format_indicators_table(sym, results))


# ---------------------------------------------------------------------------
# Command: stream
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("symbol", default="AAPL")
@click.option(
    "--count",
    "-n",
    default=10,
    type=int,
    show_default=True,
    help="Number of events to stream (0 for infinite)",
)
@click.option(
    "--rate",
    "-r",
    default=5.0,
    type=float,
    show_default=True,
    help="Stream generation rate (events per second)",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["tick", "bar"], case_sensitive=False),
    default="tick",
    show_default=True,
    help="Stream individual ticks or 1s bars",
)
@click.option(
    "--format",
    "-f",
    "out_format",
    type=click.Choice(["table", "json", "jsonl"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output streaming format",
)
def stream(
    symbol: str,
    count: int,
    rate: float,
    mode: str,
    out_format: str,
) -> None:
    """Stream live synthetic market data ticks or bars."""
    try:
        sym = validate_symbol(symbol)
    except (ValueError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    feed = SyntheticMarketFeed(symbols=[sym])
    delay = 1.0 / max(rate, 0.1)
    emitted = 0

    if out_format == "table":
        table = Table(title=f"Live Feed: {sym} ({mode.upper()})", box=box.SIMPLE)
        if mode == "tick":
            table.add_column("Seq", justify="right", style="dim")
            table.add_column("Timestamp", style="dim")
            table.add_column("Symbol", style="cyan bold")
            table.add_column("Side", justify="center")
            table.add_column("Price", justify="right", style="green bold")
            table.add_column("Size", justify="right")
        else:
            table.add_column("Timestamp", style="dim")
            table.add_column("Symbol", style="cyan bold")
            table.add_column("Open", justify="right")
            table.add_column("High", justify="right", style="green")
            table.add_column("Low", justify="right", style="red")
            table.add_column("Close", justify="right", style="green bold")
            table.add_column("Volume", justify="right")

    try:
        while count == 0 or emitted < count:
            if mode == "tick":
                event = feed.generate_tick(sym)
                event.sequence = emitted + 1
                if out_format == "json":
                    click.echo(json.dumps(event.to_dict(), indent=2))
                elif out_format == "jsonl":
                    click.echo(json.dumps(event.to_dict()))
                else:
                    side_style = "green" if event.side == Side.BUY else "red"
                    table.add_row(
                        str(event.sequence),
                        event.timestamp.strftime("%H:%M:%S.%f")[:-3],
                        event.symbol,
                        f"[{side_style}]{event.side.value.upper()}[/{side_style}]",
                        f"${event.price:.2f}",
                        f"{event.size:.2f}",
                    )
            else:
                bars = feed.generate_bars(sym, n_bars=1)
                bar = bars[0]
                if out_format == "json":
                    click.echo(json.dumps(bar.to_dict(), indent=2))
                elif out_format == "jsonl":
                    click.echo(json.dumps(bar.to_dict()))
                else:
                    table.add_row(
                        bar.timestamp.strftime("%H:%M:%S"),
                        bar.symbol,
                        f"${bar.open:.2f}",
                        f"${bar.high:.2f}",
                        f"${bar.low:.2f}",
                        f"${bar.close:.2f}",
                        f"{bar.volume:.0f}",
                    )

            emitted += 1
            if count == 0 or emitted < count:
                time.sleep(delay)

        if out_format == "table":
            console.print(table)

    except KeyboardInterrupt:
        if out_format == "table":
            console.print(table)
        console.print(f"\n[yellow]Stream stopped after {emitted} events.[/yellow]")


# ---------------------------------------------------------------------------
# Command: replay
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--speed",
    "-s",
    default=0.0,
    type=float,
    show_default=True,
    help="Replay speed multiplier (0.0 for instant execution)",
)
@click.option(
    "--limit",
    "-l",
    default=0,
    type=int,
    show_default=True,
    help="Maximum ticks to replay (0 for all)",
)
@click.option(
    "--symbol",
    default=None,
    help="Filter replayed ticks by symbol",
)
def replay(
    file: str,
    speed: float,
    limit: int,
    symbol: Optional[str],
) -> None:
    """Replay historical market telemetry from JSONL or CSV file."""
    p = Path(file)
    console.print(f"[bold cyan]Loading replay source:[/bold cyan] {p.name}")

    engine = HistoricalReplayEngine(source_file=p)
    if symbol:
        sym_upper = symbol.strip().upper()
        engine._ticks = [t for t in engine._ticks if t.symbol == sym_upper]

    if limit > 0:
        engine._ticks = engine._ticks[:limit]

    total = len(engine._ticks)
    if total == 0:
        console.print("[yellow]No matching ticks found in replay file.[/yellow]")
        return

    console.print(
        f"[green]Loaded {total:,} ticks. Starting replay (speed multiplier: {speed:.1f}x)...[/green]"
    )

    t0 = time.perf_counter()
    replayed_count = 0

    async def _run_replay() -> None:
        nonlocal replayed_count
        async for _ in engine.stream_ticks(speed_multiplier=speed):
            replayed_count += 1

    try:
        asyncio.run(_run_replay())
    except KeyboardInterrupt:
        console.print("[yellow]Replay interrupted by user.[/yellow]")

    elapsed = time.perf_counter() - t0
    rate = replayed_count / elapsed if elapsed > 0 else 0.0

    table = Table(title="Historical Replay Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")
    table.add_row("Source File", p.name)
    table.add_row("Ticks Replayed", f"{replayed_count:,} of {total:,}")
    table.add_row("Elapsed Time", f"{elapsed:.4f} sec")
    table.add_row("Replay Rate", f"{rate:,.0f} ticks/sec")
    console.print(table)


# ---------------------------------------------------------------------------
# Command Group: snapshot
# ---------------------------------------------------------------------------
@cli.group()
def snapshot() -> None:
    """Manage market data buffer snapshots and state persistence."""
    pass


@snapshot.command("save")
@click.option(
    "--file",
    "-f",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Destination snapshot file path",
)
@click.option(
    "--tickers",
    "-s",
    default="AAPL,MSFT,NVDA",
    show_default=True,
    help="Comma-separated tickers to generate into snapshot",
)
@click.option(
    "--ticks-count",
    "-n",
    default=100,
    type=int,
    show_default=True,
    help="Number of ticks per symbol",
)
def snapshot_save(output_file: str, tickers: str, ticks_count: int) -> None:
    """Save synthesized market data state into snapshot file."""
    p = Path(output_file)
    syms = [s.strip().upper() for s in tickers.split(",") if s.strip()]

    feed = SyntheticMarketFeed(symbols=syms)
    server = create_fintech_mcp_server(warm_up_tickers=syms)
    buffer = getattr(server, "buffer")

    for s in syms:
        for t in feed.generate_history(s, n_points=ticks_count):
            buffer.push_tick(t)

    meta = save_buffer_snapshot(buffer, p)
    console.print(
        Panel(
            f"[bold green]Snapshot Saved Successfully[/bold green]\n"
            f"File: [cyan]{p}[/cyan]\n"
            f"Symbols: [yellow]{meta['symbols_count']}[/yellow]\n"
            f"Total Ticks: [yellow]{meta['total_ticks']:,}[/yellow]\n"
            f"Total Bars: [yellow]{meta['total_bars']:,}[/yellow]",
            title="Snapshot Persistence",
            border_style="green",
        )
    )


@snapshot.command("inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def snapshot_inspect(file: str) -> None:
    """Inspect metadata and contents of a buffer snapshot file."""
    p = Path(file)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    table = Table(title=f"Snapshot Metadata: {p.name}", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green bold")

    table.add_row("Version", str(data.get("version", "unknown")))
    table.add_row("Saved At", str(data.get("timestamp", "unknown")))
    table.add_row("Capacity", f"{data.get('capacity', 0):,}")

    ticks_map = data.get("ticks", {})
    bars_map = data.get("bars", {})
    obs_map = data.get("order_books", {})

    total_ticks = sum(len(v) for v in ticks_map.values())
    total_bars = sum(len(v) for v in bars_map.values())

    table.add_row("Symbols", ", ".join(ticks_map.keys()) or "None")
    table.add_row("Total Ticks", f"{total_ticks:,}")
    table.add_row("Total Bars", f"{total_bars:,}")
    table.add_row("Active Order Books", str(len(obs_map)))

    console.print(table)


@snapshot.command("load")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def snapshot_load(file: str) -> None:
    """Load and restore market data buffer from a snapshot file."""
    p = Path(file)
    buf = load_buffer_snapshot(p)
    console.print(
        Panel(
            f"[bold green]Snapshot Loaded Successfully[/bold green]\n"
            f"File: [cyan]{p.name}[/cyan]\n"
            f"Buffer Capacity: [yellow]{buf.capacity:,}[/yellow]\n"
            f"Symbols: [yellow]{', '.join(buf.symbols) or 'None'}[/yellow]",
            title="Snapshot Loader",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Command Group: alerts
# ---------------------------------------------------------------------------
@cli.group()
def alerts() -> None:
    """Configure and monitor real-time market telemetry alerts."""
    pass


@alerts.command("list")
def alerts_list() -> None:
    """List registered alert rules."""
    engine = AlertEngine()
    engine.create_rule("AAPL", AlertType.PRICE_ABOVE, 200.0, message="AAPL breakout target")
    engine.create_rule("BTC/USD", AlertType.PRICE_BELOW, 60_000.0, message="BTC key support")
    engine.create_rule("NVDA", AlertType.VOLUME_SPIKE, 50_000.0, message="NVDA institutional flow")

    console.print(format_alert_rules_table(list(engine.rules.values())))


@alerts.command("add")
@click.option("--symbol", "-s", required=True, help="Ticker symbol")
@click.option(
    "--type",
    "-t",
    "alert_type_str",
    required=True,
    type=click.Choice(
        [
            "price_above",
            "price_below",
            "volume_spike",
            "spread_wider_than",
            "imbalance_spike",
        ],
        case_sensitive=False,
    ),
    help="Trigger condition type",
)
@click.option("--threshold", required=True, type=float, help="Trigger threshold value")
@click.option("--message", "-m", default=None, help="Optional notification message")
def alerts_add(
    symbol: str,
    alert_type_str: str,
    threshold: float,
    message: Optional[str],
) -> None:
    """Create a new market condition alert rule."""
    try:
        sym = validate_symbol(symbol)
    except (ValueError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    atype = AlertType(alert_type_str.lower())
    engine = AlertEngine()
    rule = engine.create_rule(
        symbol=sym,
        alert_type=atype,
        threshold=threshold,
        message=message,
    )

    console.print(
        Panel(
            f"[bold green]Alert Rule Registered[/bold green]\n"
            f"Alert ID: [cyan]{rule.alert_id}[/cyan]\n"
            f"Symbol: [yellow]{rule.symbol}[/yellow]\n"
            f"Condition: [white]{rule.alert_type.value}[/white]\n"
            f"Threshold: [green]{rule.threshold:,.4f}[/green]\n"
            f"Message: {rule.message or 'N/A'}",
            title="Alert Configuration",
            border_style="green",
        )
    )


@alerts.command("monitor")
@click.argument("symbol", default="AAPL")
@click.option(
    "--count",
    "-n",
    default=15,
    type=int,
    show_default=True,
    help="Number of evaluation ticks",
)
def alerts_monitor(symbol: str, count: int) -> None:
    """Monitor real-time ticks against active threshold alerts."""
    try:
        sym = validate_symbol(symbol)
    except (ValueError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    engine = AlertEngine()
    feed = SyntheticMarketFeed(symbols=[sym])
    current_tick = feed.generate_tick(sym)

    # Add realistic threshold rules around current price
    engine.create_rule(
        sym,
        AlertType.PRICE_ABOVE,
        current_tick.price * 1.001,
        one_shot=False,
        message=f"{sym} Up-tick breakout",
    )
    engine.create_rule(
        sym,
        AlertType.PRICE_BELOW,
        current_tick.price * 0.999,
        one_shot=False,
        message=f"{sym} Down-tick dip",
    )

    console.print(f"[bold cyan]Monitoring alerts on {sym} ({count} ticks)...[/bold cyan]")
    for _ in range(count):
        tick = feed.generate_tick(sym)
        engine.on_tick(tick)

    evts = engine.get_event_history(sym)
    if evts:
        console.print(format_alert_events_table(evts))
    else:
        console.print("[green]No alerts triggered during monitoring session.[/green]")


# ---------------------------------------------------------------------------
# Command Group: exec (Simulated Execution)
# ---------------------------------------------------------------------------
@cli.group(name="exec")
def exec_group() -> None:
    """Simulated trade execution and portfolio position tracking."""
    pass


@exec_group.command("submit")
@click.argument("symbol", default="AAPL")
@click.option(
    "--side",
    required=True,
    type=click.Choice(["buy", "sell"], case_sensitive=False),
    help="Order side",
)
@click.option("--quantity", "-q", required=True, type=float, help="Order quantity")
@click.option(
    "--type",
    "-t",
    "order_type_str",
    type=click.Choice(["market", "limit", "stop"], case_sensitive=False),
    default="market",
    show_default=True,
    help="Order execution type",
)
@click.option("--price", "-p", type=float, default=None, help="Limit price")
@click.option("--stop-price", type=float, default=None, help="Stop trigger price")
def exec_submit(
    symbol: str,
    side: str,
    quantity: float,
    order_type_str: str,
    price: Optional[float],
    stop_price: Optional[float],
) -> None:
    """Submit a simulated order against real-time synthetic L2 book."""
    try:
        sym = validate_symbol(symbol)
    except (ValueError, TypeError) as err:
        console.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    order_side = Side(side.lower())
    order_type = OrderType(order_type_str.lower())

    feed = SyntheticMarketFeed(symbols=[sym])
    ob = feed.generate_order_book(sym, depth=10)

    sim = ExecutionSimulator()
    sim.update_order_book(ob)

    order = sim.submit_order(
        symbol=sym,
        side=order_side,
        quantity=quantity,
        order_type=order_type,
        price=price,
        stop_price=stop_price,
    )

    console.print(
        Panel(
            f"[bold green]Simulated Order Placed[/bold green]\n"
            f"Order ID: [cyan]{order.order_id}[/cyan]\n"
            f"Symbol: [yellow]{order.symbol}[/yellow] | Side: [bold]{order.side.value.upper()}[/bold]\n"
            f"Type: {order.order_type.value.upper()} | Quantity: {order.quantity:,.2f}\n"
            f"Status: [bold green]{order.status.value.upper()}[/bold green]\n"
            f"Filled Qty: {order.filled_quantity:,.2f} @ Avg Price: ${order.average_fill_price or 0.0:,.2f}\n"
            f"Fee Paid: ${order.fee_paid:.4f}",
            title="Execution Report",
            border_style="green",
        )
    )
    console.print(format_portfolio_panel(sim.portfolio))


@exec_group.command("portfolio")
def exec_portfolio() -> None:
    """Display current simulated portfolio status and open positions."""
    server = create_fintech_mcp_server()
    sim: ExecutionSimulator = getattr(server, "execution_simulator")
    feed: SyntheticMarketFeed = getattr(server, "feed")

    # Simulate a couple of initial trades to display realistic portfolio
    sim.update_order_book(feed.generate_order_book("AAPL"))
    sim.submit_order("AAPL", Side.BUY, 50.0)
    sim.update_order_book(feed.generate_order_book("NVDA"))
    sim.submit_order("NVDA", Side.BUY, 25.0)

    console.print(format_portfolio_panel(sim.portfolio))
    console.print(format_positions_table(sim.portfolio))


@exec_group.command("orders")
def exec_orders() -> None:
    """List active and historical simulated orders."""
    server = create_fintech_mcp_server()
    sim: ExecutionSimulator = getattr(server, "execution_simulator")
    feed: SyntheticMarketFeed = getattr(server, "feed")

    sim.update_order_book(feed.generate_order_book("AAPL"))
    sim.submit_order("AAPL", Side.BUY, 50.0)
    sim.submit_order(
        "AAPL",
        Side.SELL,
        25.0,
        order_type=OrderType.LIMIT,
        price=250.0,
    )

    console.print(format_orders_table(list(sim.orders.values())))


# ---------------------------------------------------------------------------
# Command Group: mcp (Direct Inspector / CLI Invocation)
# ---------------------------------------------------------------------------
@cli.group()
def mcp() -> None:
    """Direct Model Context Protocol server inspector and tool caller."""
    pass


@mcp.command("tools")
def mcp_tools() -> None:
    """List registered MCP tools and their parameter schemas."""
    server = create_fintech_mcp_server()

    async def _list() -> List[Dict[str, Any]]:
        req = JsonRpcRequest(id=1, method="tools/list")
        resp = await server.handle_request(req)
        if resp and resp.result:
            return resp.result.get("tools", [])
        return []

    tools_list = asyncio.run(_list())
    console.print(format_mcp_tools_table(tools_list))


@mcp.command("resources")
def mcp_resources() -> None:
    """List registered MCP resources and URIs."""
    server = create_fintech_mcp_server()

    async def _list() -> List[Dict[str, Any]]:
        req = JsonRpcRequest(id=1, method="resources/list")
        resp = await server.handle_request(req)
        if resp and resp.result:
            return resp.result.get("resources", [])
        return []

    res_list = asyncio.run(_list())
    console.print(format_mcp_resources_table(res_list))


@mcp.command("prompts")
def mcp_prompts() -> None:
    """List registered MCP prompt templates."""
    server = create_fintech_mcp_server()

    async def _list() -> List[Dict[str, Any]]:
        req = JsonRpcRequest(id=1, method="prompts/list")
        resp = await server.handle_request(req)
        if resp and resp.result:
            return resp.result.get("prompts", [])
        return []

    p_list = asyncio.run(_list())
    console.print(format_mcp_prompts_table(p_list))


@mcp.command("call")
@click.option("--tool", "-t", required=True, help="MCP tool name to invoke")
@click.option(
    "--args",
    "-a",
    default="{}",
    help='Tool arguments as JSON string (e.g. \'{"symbol": "AAPL"}\')',
)
def mcp_call(tool: str, args: str) -> None:
    """Invoke an MCP tool directly through the server engine."""
    server = create_fintech_mcp_server()
    try:
        parsed_args = json.loads(args)
    except json.JSONDecodeError as err:
        console.print(f"[bold red]JSON Parse Error in arguments:[/bold red] {err}")
        sys.exit(1)

    async def _call() -> Any:
        req = JsonRpcRequest(
            id=1,
            method="tools/call",
            params={"name": tool, "arguments": parsed_args},
        )
        return await server.handle_request(req)

    resp = asyncio.run(_call())
    if resp and resp.result:
        content = resp.result.get("content", [])
        for item in content:
            txt = item.get("text", "")
            try:
                formatted_json = json.dumps(json.loads(txt), indent=2)
                console.print(formatted_json)
            except Exception:
                console.print(txt)
    elif resp and resp.error:
        console.print(f"[bold red]MCP Error ({resp.error.code}):[/bold red] {resp.error.message}")
        sys.exit(1)


@mcp.command("read")
@click.option("--uri", "-u", required=True, help="MCP resource URI to read")
def mcp_read(uri: str) -> None:
    """Read contents of an MCP resource."""
    server = create_fintech_mcp_server()

    async def _read() -> Any:
        req = JsonRpcRequest(
            id=1,
            method="resources/read",
            params={"uri": uri},
        )
        return await server.handle_request(req)

    resp = asyncio.run(_read())
    if resp and resp.result:
        contents = resp.result.get("contents", [])
        for item in contents:
            txt = item.get("text", "")
            try:
                formatted_json = json.dumps(json.loads(txt), indent=2)
                console.print(formatted_json)
            except Exception:
                console.print(txt)
    elif resp and resp.error:
        console.print(f"[bold red]MCP Error ({resp.error.code}):[/bold red] {resp.error.message}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command: benchmark
# ---------------------------------------------------------------------------
@cli.command()
@click.option(
    "--ticks",
    default=100_000,
    show_default=True,
    help="Number of ticks for benchmark ingestion",
)
@click.option(
    "--symbol",
    default="NVDA",
    show_default=True,
    help="Benchmark ticker symbol",
)
@click.option(
    "--json-output",
    "-j",
    "--json",
    "json_mode",
    is_flag=True,
    help="Output results as raw JSON",
)
def benchmark(ticks: int, symbol: str, json_mode: bool) -> None:
    """Run performance benchmark on telemetry ingestion and indicator calculation."""
    bench_res = run_telemetry_benchmark(ticks_count=ticks, symbol=symbol)

    if json_mode:
        click.echo(json.dumps(bench_res.to_dict(), indent=2))
        return

    # Preserves exact title "Performance Benchmark Results" for compatibility
    table = format_benchmark_table(bench_res)
    console.print(table)


if __name__ == "__main__":
    cli()
