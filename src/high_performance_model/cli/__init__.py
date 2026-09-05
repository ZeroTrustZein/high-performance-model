"""CLI command line interface for FinTech MCP Server."""

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
from high_performance_model.cli.main import cli

__all__ = [
    "cli",
    "format_alert_events_table",
    "format_alert_rules_table",
    "format_benchmark_table",
    "format_indicators_table",
    "format_mcp_prompts_table",
    "format_mcp_resources_table",
    "format_mcp_tools_table",
    "format_order_book_table",
    "format_orders_table",
    "format_portfolio_panel",
    "format_positions_table",
    "format_quote_panel",
]
