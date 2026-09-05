"""Command Line Interface for high-performance FinTech MCP server."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from high_performance_model.protocol.transports import StdioTransport
from high_performance_model.server.app import create_fintech_mcp_server
from high_performance_model.telemetry.benchmark import run_telemetry_benchmark
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

    bench_res = run_telemetry_benchmark(ticks_count=ticks, symbol="NVDA")

    table = Table(title="Performance Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Ingestion Count", f"{bench_res.total_ticks:,} ticks")
    table.add_row("Ingestion Time", f"{bench_res.duration_seconds:.4f} sec")
    table.add_row("Ingestion Throughput", f"{bench_res.ticks_per_second:,.0f} ticks/sec")
    table.add_row("Latency p50", f"{bench_res.latency_p50_us:.2f} µs")
    table.add_row("Latency p95", f"{bench_res.latency_p95_us:.2f} µs")
    table.add_row("Latency p99", f"{bench_res.latency_p99_us:.2f} µs")
    table.add_row("Buffer Utilization", f"{bench_res.buffer_utilization_pct:.1f}%")

    indicators_meta = bench_res.metadata.get("indicators", {})
    if indicators_meta:
        total_ind_ms = sum(indicators_meta.values())
        table.add_row("Multi-Indicator Suite Time", f"{total_ind_ms:.2f} ms")

    console.print(table)


if __name__ == "__main__":
    cli()
