"""Command Line Interface for high-performance FinTech MCP server."""

from __future__ import annotations

import asyncio
import sys
import time

import click
from rich.console import Console
from rich.table import Table

from high_performance_model.indicators.series import (
    bollinger_bands,
    exponential_moving_average,
    macd,
    relative_strength_index,
    simple_moving_average,
)
from high_performance_model.protocol.transports import StdioTransport
from high_performance_model.server.app import create_fintech_mcp_server
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.telemetry.generator import SyntheticMarketFeed

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="high-performance-model")
def cli() -> None:
    """High-performance Model Context Protocol server for FinTech telemetry."""
    pass


@cli.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio"]),
    default="stdio",
    help="MCP transport protocol (default: stdio)",
)
def serve(transport: str) -> None:
    """Launch Model Context Protocol server."""
    if transport == "stdio":
        server = create_fintech_mcp_server()
        stdio_transport = StdioTransport()
        try:
            asyncio.run(server.run(stdio_transport))
        except KeyboardInterrupt:
            sys.exit(0)


@cli.command()
@click.argument("symbol", default="AAPL")
def quote(symbol: str) -> None:
    """Fetch current market quote and order book snapshot."""
    sym = symbol.upper()
    feed = SyntheticMarketFeed()
    tick = feed.generate_tick(sym)
    ob = feed.generate_order_book(sym, depth=5)

    table = Table(title=f"Market Quote: {sym}")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Last Price", f"${tick.price:.2f}")
    table.add_row("Last Size", f"{tick.size:.2f}")
    table.add_row("Side", tick.side.value.upper())
    table.add_row("Best Bid", f"${ob.best_bid:.2f}" if ob.best_bid else "N/A")
    table.add_row("Best Ask", f"${ob.best_ask:.2f}" if ob.best_ask else "N/A")
    table.add_row("Spread", f"${ob.spread:.4f}" if ob.spread else "N/A")
    table.add_row("Imbalance", f"{ob.order_book_imbalance:.4f}")

    console.print(table)


@cli.command()
@click.option("--ticks", default=100_000, help="Number of ticks for benchmark ingestion")
def benchmark(ticks: int) -> None:
    """Run performance benchmark on telemetry ingestion and indicator calculation."""
    console.print(f"[bold cyan]Running FinTech MCP Benchmark ({ticks:,} ticks)...[/bold cyan]")

    buffer = MarketDataBuffer(capacity=ticks + 1000)
    feed = SyntheticMarketFeed()

    # 1. Ingestion Benchmark
    t0 = time.perf_counter()
    for _ in range(ticks):
        tick = feed.generate_tick("NVDA")
        buffer.push_tick(tick)
    duration_ingest = time.perf_counter() - t0
    rate_ingest = ticks / duration_ingest if duration_ingest > 0 else 0.0

    # 2. Vectorized Indicators Benchmark
    prices = buffer.get_price_series("NVDA", limit=ticks)
    t1 = time.perf_counter()
    _ = simple_moving_average(prices, period=20)
    _ = exponential_moving_average(prices, period=20)
    _ = relative_strength_index(prices, period=14)
    _ = macd(prices)
    _ = bollinger_bands(prices, period=20)
    duration_indicators = time.perf_counter() - t1

    table = Table(title="Performance Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Ingestion Count", f"{ticks:,} ticks")
    table.add_row("Ingestion Time", f"{duration_ingest:.4f} sec")
    table.add_row("Ingestion Throughput", f"{rate_ingest:,.0f} ticks/sec")
    table.add_row("Multi-Indicator Suite Time", f"{duration_indicators * 1000:.2f} ms")

    console.print(table)


if __name__ == "__main__":
    cli()
