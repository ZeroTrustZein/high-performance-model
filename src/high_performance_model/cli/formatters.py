"""Rich formatting utilities for the FinTech MCP CLI interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from high_performance_model.alerts.models import AlertEvent, AlertRule
from high_performance_model.execution.models import Portfolio, SimulatedOrder
from high_performance_model.types import (
    BenchmarkRunResult,
    MarketTick,
    OrderBook,
    Side,
)


def format_quote_panel(tick: MarketTick, order_book: Optional[OrderBook] = None) -> Panel:
    """Format market tick and top-of-book metrics into a rich panel."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style="cyan bold", justify="left")
    grid.add_column(style="green", justify="right")
    grid.add_column(style="cyan bold", justify="left")
    grid.add_column(style="green", justify="right")

    side_style = "green bold" if tick.side == Side.BUY else "red bold"
    grid.add_row("Symbol:", tick.symbol, "Exchange:", tick.exchange or "SIM")
    grid.add_row("Last Price:", f"${tick.price:,.2f}", "Size:", f"{tick.size:,.2f}")
    grid.add_row(
        "Side:",
        Text(tick.side.value.upper(), style=side_style),
        "Timestamp:",
        tick.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    if order_book:
        microprice = order_book.microprice
        spread_bps = order_book.spread_bps
        grid.add_row(
            "Best Bid:",
            f"${order_book.best_bid:,.2f}" if order_book.best_bid else "N/A",
            "Best Ask:",
            f"${order_book.best_ask:,.2f}" if order_book.best_ask else "N/A",
        )
        grid.add_row(
            "Spread:",
            f"${order_book.spread:,.4f}" if order_book.spread else "N/A",
            "Spread (bps):",
            f"{spread_bps:.2f} bps" if spread_bps is not None else "N/A",
        )
        grid.add_row(
            "Microprice:",
            f"${microprice:,.4f}" if microprice is not None else "N/A",
            "Imbalance:",
            f"{order_book.order_book_imbalance:+.4f}",
        )

    return Panel(
        grid,
        title=f"[bold white]Market Quote: {tick.symbol}[/bold white]",
        border_style="cyan",
        box=box.ROUNDED,
    )


def format_order_book_table(order_book: OrderBook, depth: int = 5) -> Table:
    """Format order book depth ladder into a side-by-side bids/asks table."""
    table = Table(
        title=f"Order Book Depth: {order_book.symbol} (Top {depth} Levels)",
        box=box.ROUNDED,
    )
    table.add_column("Bid Count", justify="right", style="dim")
    table.add_column("Bid Qty", justify="right", style="green")
    table.add_column("Bid Price", justify="right", style="green bold")
    table.add_column("Ask Price", justify="right", style="red bold")
    table.add_column("Ask Qty", justify="right", style="red")
    table.add_column("Ask Count", justify="right", style="dim")

    bids = order_book.bids[:depth]
    asks = order_book.asks[:depth]
    max_len = max(len(bids), len(asks))

    for i in range(max_len):
        b = bids[i] if i < len(bids) else None
        a = asks[i] if i < len(asks) else None

        b_orders = str(b.orders_count) if b and b.orders_count is not None else ("-" if b else "")
        b_qty = f"{b.quantity:,.2f}" if b else ""
        b_px = f"${b.price:,.2f}" if b else ""

        a_px = f"${a.price:,.2f}" if a else ""
        a_qty = f"{a.quantity:,.2f}" if a else ""
        a_orders = str(a.orders_count) if a and a.orders_count is not None else ("-" if a else "")

        table.add_row(b_orders, b_qty, b_px, a_px, a_qty, a_orders)

    return table


def format_indicators_table(symbol: str, indicators: Dict[str, Any]) -> Table:
    """Format technical indicators with signal interpretations."""
    table = Table(title=f"Technical Indicators: {symbol}", box=box.ROUNDED)
    table.add_column("Indicator", style="cyan bold")
    table.add_column("Parameters", style="dim")
    table.add_column("Latest Value", style="green bold")
    table.add_column("Signal / Interpretation", style="yellow")

    for name, data in indicators.items():
        val_str = data.get("value_str", "N/A")
        params = data.get("params", "-")
        signal = data.get("signal", "Neutral")
        style = (
            "green"
            if "Bullish" in signal or "Oversold" in signal
            else ("red" if "Bearish" in signal or "Overbought" in signal else "yellow")
        )
        table.add_row(name, params, val_str, Text(signal, style=style))

    return table


def format_benchmark_table(result: BenchmarkRunResult) -> Table:
    """Format performance benchmark metrics table."""
    table = Table(title="Performance Benchmark Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Ingestion Count", f"{result.total_ticks:,} ticks")
    table.add_row("Ingestion Time", f"{result.duration_seconds:.4f} sec")
    table.add_row("Ingestion Throughput", f"{result.ticks_per_second:,.0f} ticks/sec")
    table.add_row("Latency p50", f"{result.latency_p50_us:.2f} µs")
    table.add_row("Latency p95", f"{result.latency_p95_us:.2f} µs")
    table.add_row("Latency p99", f"{result.latency_p99_us:.2f} µs")
    table.add_row("Buffer Utilization", f"{result.buffer_utilization_pct:.1f}%")

    indicators_meta = result.metadata.get("indicators", {})
    if indicators_meta:
        total_ind_ms = sum(indicators_meta.values())
        table.add_row("Multi-Indicator Suite Time", f"{total_ind_ms:.2f} ms")
        for ind_name, ms in indicators_meta.items():
            table.add_row(f"  └─ {ind_name.upper()}", f"{ms:.3f} ms")

    return table


def format_portfolio_panel(portfolio: Portfolio) -> Panel:
    """Format portfolio overview panel."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style="cyan bold")
    grid.add_column(style="green")
    grid.add_column(style="cyan bold")
    grid.add_column(style="green")

    pnl_style = "green bold" if portfolio.total_unrealized_pnl >= 0 else "red bold"
    ret_style = "green bold" if portfolio.total_return_pct >= 0 else "red bold"

    grid.add_row(
        "Cash Balance:",
        f"${portfolio.cash_balance:,.2f}",
        "Initial Cash:",
        f"${portfolio.initial_cash:,.2f}",
    )
    grid.add_row(
        "Portfolio Value:",
        f"${portfolio.total_portfolio_value:,.2f}",
        "Total Return:",
        Text(f"{portfolio.total_return_pct:+.2f}%", style=ret_style),
    )
    grid.add_row(
        "Realized PnL:",
        f"${portfolio.total_realized_pnl:,.2f}",
        "Unrealized PnL:",
        Text(f"${portfolio.total_unrealized_pnl:+,.2f}", style=pnl_style),
    )
    grid.add_row(
        "Fees Paid:",
        f"${portfolio.total_fees_paid:,.2f}",
        "Active Positions:",
        str(len([p for p in portfolio.positions.values() if abs(p.quantity) > 0])),
    )

    return Panel(
        grid,
        title="[bold white]Portfolio Status[/bold white]",
        border_style="green",
        box=box.ROUNDED,
    )


def format_positions_table(portfolio: Portfolio) -> Table:
    """Format active positions table."""
    table = Table(title="Open Positions", box=box.ROUNDED)
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Quantity", justify="right")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Mark Price", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("Unrealized PnL", justify="right")

    for pos in portfolio.positions.values():
        if abs(pos.quantity) < 1e-9:
            continue
        pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
        qty_style = "green" if pos.quantity > 0 else "red"
        table.add_row(
            pos.symbol,
            Text(f"{pos.quantity:+,.2f}", style=qty_style),
            f"${pos.average_entry_price:,.2f}",
            f"${pos.current_price:,.2f}",
            f"${pos.market_value:,.2f}",
            Text(f"${pos.unrealized_pnl:+,.2f}", style=pnl_style),
        )

    return table


def format_orders_table(orders: List[SimulatedOrder]) -> Table:
    """Format simulated orders table."""
    table = Table(title="Simulated Orders", box=box.ROUNDED)
    table.add_column("Order ID", style="dim")
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Side", justify="center")
    table.add_column("Type", justify="center")
    table.add_column("Quantity", justify="right")
    table.add_column("Limit / Stop", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Filled", justify="right")
    table.add_column("Avg Fill", justify="right")

    for ord in orders:
        side_style = "green" if ord.side == Side.BUY else "red"
        status_style = (
            "green bold"
            if ord.status.value == "filled"
            else (
                "yellow"
                if ord.status.value in {"pending", "partially_filled"}
                else "dim"
            )
        )
        price_str = (
            f"${ord.price:,.2f}"
            if ord.price
            else (f"Stop: ${ord.stop_price:,.2f}" if ord.stop_price else "MKT")
        )
        avg_str = f"${ord.average_fill_price:,.2f}" if ord.average_fill_price else "-"
        table.add_row(
            ord.order_id,
            ord.symbol,
            Text(ord.side.value.upper(), style=side_style),
            ord.order_type.value.upper(),
            f"{ord.quantity:,.2f}",
            price_str,
            Text(ord.status.value.upper(), style=status_style),
            f"{ord.filled_quantity:,.2f}",
            avg_str,
        )

    return table


def format_alert_rules_table(rules: List[AlertRule]) -> Table:
    """Format active alert rules table."""
    table = Table(title="Market Alert Rules", box=box.ROUNDED)
    table.add_column("Rule ID", style="dim")
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Type", style="yellow")
    table.add_column("Threshold", justify="right", style="green")
    table.add_column("Triggered", justify="center")
    table.add_column("Count", justify="right")
    table.add_column("Message", style="white")

    for r in rules:
        trig_style = "red bold" if r.triggered else "green"
        trig_text = "YES" if r.triggered else "NO"
        table.add_row(
            r.alert_id,
            r.symbol,
            r.alert_type.value,
            f"{r.threshold:,.4f}",
            Text(trig_text, style=trig_style),
            str(r.trigger_count),
            r.message or "-",
        )

    return table


def format_alert_events_table(events: List[AlertEvent]) -> Table:
    """Format fired alert events table."""
    table = Table(title="Triggered Alert Events", box=box.ROUNDED)
    table.add_column("Event ID", style="dim")
    table.add_column("Rule ID", style="dim")
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Alert Type", style="yellow")
    table.add_column("Threshold", justify="right")
    table.add_column("Current Val", justify="right", style="bold red")
    table.add_column("Timestamp", style="dim")
    table.add_column("Message", style="white")

    for e in events:
        table.add_row(
            e.event_id,
            e.alert_id,
            e.symbol,
            e.alert_type.value,
            f"{e.threshold:,.4f}",
            f"{e.current_value:,.4f}",
            e.timestamp.strftime("%H:%M:%S.%f")[:-3],
            e.message,
        )

    return table


def format_mcp_tools_table(tools: List[Dict[str, Any]]) -> Table:
    """Format MCP server tools list table."""
    table = Table(title="Registered MCP Tools", box=box.ROUNDED)
    table.add_column("Tool Name", style="cyan bold", overflow="fold")
    table.add_column("Description", style="white")
    table.add_column("Required Args", style="yellow", overflow="fold")

    for tool in tools:
        schema = tool.get("inputSchema", {})
        req = schema.get("required", [])
        props = list(schema.get("properties", {}).keys())
        req_str = ", ".join(req) if req else (", ".join(props) if props else "None")
        table.add_row(tool.get("name", ""), tool.get("description", ""), req_str)

    return table


def format_mcp_resources_table(resources: List[Dict[str, Any]]) -> Table:
    """Format MCP server resources list table."""
    table = Table(title="Registered MCP Resources", box=box.ROUNDED)
    table.add_column("URI", style="cyan bold", overflow="fold")
    table.add_column("Name", style="green")
    table.add_column("MIME Type", style="dim")
    table.add_column("Description", style="white")

    for res in resources:
        table.add_row(
            res.get("uri", ""),
            res.get("name", ""),
            res.get("mimeType", "text/plain"),
            res.get("description", "-"),
        )

    return table


def format_mcp_prompts_table(prompts: List[Dict[str, Any]]) -> Table:
    """Format MCP server prompts list table."""
    table = Table(title="Registered MCP Prompts", box=box.ROUNDED)
    table.add_column("Prompt Name", style="cyan bold")
    table.add_column("Description", style="white")
    table.add_column("Arguments", style="yellow")

    for p in prompts:
        args = [a.get("name", "") for a in p.get("arguments", [])]
        args_str = ", ".join(args) if args else "None"
        table.add_row(p.get("name", ""), p.get("description", ""), args_str)

    return table
