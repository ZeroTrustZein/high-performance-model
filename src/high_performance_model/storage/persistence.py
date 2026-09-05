"""Market telemetry storage and persistence handlers for JSONL, CSV, and snapshot formats."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.types import (
    Bar,
    BarTimeframe,
    MarketTick,
    OrderBook,
    OrderBookLevel,
    Side,
)


def save_ticks_to_jsonl(ticks: List[MarketTick], file_path: Union[str, Path]) -> int:
    """Serialize a list of MarketTicks to a newline-delimited JSON (JSONL) file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for tick in ticks:
            line = json.dumps(tick.to_dict())
            f.write(line + "\n")
            count += 1
    return count


def load_ticks_from_jsonl(
    file_path: Union[str, Path], limit: Optional[int] = None
) -> List[MarketTick]:
    """Deserialize MarketTicks from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tick file not found: {path}")

    ticks: List[MarketTick] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Parse ISO timestamp if present
            if "timestamp" in data and isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            ticks.append(MarketTick(**data))
            if limit and len(ticks) >= limit:
                break
    return ticks


def save_ticks_to_csv(ticks: List[MarketTick], file_path: Union[str, Path]) -> int:
    """Serialize MarketTicks to a standardized CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "price", "size", "side", "sequence", "timestamp"])
        for tick in ticks:
            writer.writerow([
                tick.symbol,
                tick.price,
                tick.size,
                tick.side.value,
                tick.sequence or "",
                tick.timestamp.isoformat(),
            ])
            count += 1
    return count


def load_ticks_from_csv(
    file_path: Union[str, Path], limit: Optional[int] = None
) -> List[MarketTick]:
    """Deserialize MarketTicks from a CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tick CSV file not found: {path}")

    ticks: List[MarketTick] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else datetime.now(timezone.utc)
            seq = int(row["sequence"]) if row.get("sequence") else 0
            tick = MarketTick(
                symbol=row["symbol"],
                price=float(row["price"]),
                size=float(row["size"]),
                side=Side(row["side"]),
                sequence=seq,
                timestamp=ts,
            )
            ticks.append(tick)
            if limit and len(ticks) >= limit:
                break
    return ticks


def save_bars_to_jsonl(bars: List[Bar], file_path: Union[str, Path]) -> int:
    """Serialize candle Bars to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for bar in bars:
            f.write(json.dumps(bar.to_dict()) + "\n")
            count += 1
    return count


def load_bars_from_jsonl(
    file_path: Union[str, Path], limit: Optional[int] = None
) -> List[Bar]:
    """Deserialize candle Bars from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Bar file not found: {path}")

    bars: List[Bar] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "timestamp" in data and isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            if "timeframe" in data and isinstance(data["timeframe"], str):
                data["timeframe"] = BarTimeframe(data["timeframe"])
            bars.append(Bar(**data))
            if limit and len(bars) >= limit:
                break
    return bars


def save_buffer_snapshot(
    buffer: MarketDataBuffer, file_path: Union[str, Path]
) -> Dict[str, Any]:
    """Persist the full in-memory state of a MarketDataBuffer to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "version": "1.0",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "capacity": buffer.capacity,
        "ticks": {},
        "bars": {},
        "order_books": {},
    }

    # Extract all symbols from internal dicts
    symbols = set(buffer._ticks.keys()) | set(buffer._bars.keys()) | set(buffer._order_books.keys())

    for sym in symbols:
        ticks = buffer.get_ticks(sym, limit=buffer.capacity)
        state["ticks"][sym] = [t.to_dict() for t in ticks]

        bars = buffer.get_bars(sym, limit=buffer.capacity)
        state["bars"][sym] = [b.to_dict() for b in bars]

        ob = buffer.get_latest_order_book(sym)
        if ob:
            state["order_books"][sym] = ob.to_dict()

    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return {
        "file_path": str(path),
        "symbols_count": len(symbols),
        "total_ticks": sum(len(v) for v in state["ticks"].values()),
        "total_bars": sum(len(v) for v in state["bars"].values()),
        "total_order_books": len(state["order_books"]),
    }


def load_buffer_snapshot(
    file_path: Union[str, Path],
    buffer: Optional[MarketDataBuffer] = None,
) -> MarketDataBuffer:
    """Load and restore MarketDataBuffer state from a saved snapshot file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    target_buffer = buffer or MarketDataBuffer(capacity=state.get("capacity", 50_000))

    # Restore ticks
    for sym, tick_list in state.get("ticks", {}).items():
        for t_dict in tick_list:
            if "timestamp" in t_dict and isinstance(t_dict["timestamp"], str):
                t_dict["timestamp"] = datetime.fromisoformat(t_dict["timestamp"])
            target_buffer.push_tick(MarketTick(**t_dict))

    # Restore bars
    for sym, bar_list in state.get("bars", {}).items():
        for b_dict in bar_list:
            if "timestamp" in b_dict and isinstance(b_dict["timestamp"], str):
                b_dict["timestamp"] = datetime.fromisoformat(b_dict["timestamp"])
            if "timeframe" in b_dict and isinstance(b_dict["timeframe"], str):
                b_dict["timeframe"] = BarTimeframe(b_dict["timeframe"])
            target_buffer.push_bar(Bar(**b_dict))

    # Restore order books
    for sym, ob_dict in state.get("order_books", {}).items():
        bids = [OrderBookLevel(**b) for b in ob_dict.get("bids", [])]
        asks = [OrderBookLevel(**a) for a in ob_dict.get("asks", [])]
        ts = (
            datetime.fromisoformat(ob_dict["timestamp"])
            if ob_dict.get("timestamp")
            else datetime.now(timezone.utc)
        )
        ob = OrderBook(
            symbol=ob_dict["symbol"],
            bids=bids,
            asks=asks,
            timestamp=ts,
            sequence=ob_dict.get("sequence"),
        )
        target_buffer.push_order_book(ob)

    return target_buffer


class MarketDataPersistence:
    """High-level repository managing disk storage for market telemetry."""

    def __init__(self, base_directory: Union[str, Path] = "data") -> None:
        self.base_dir = Path(base_directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_symbol_path(self, symbol: str, ext: str = "jsonl") -> Path:
        """Construct path for symbol data file."""
        clean_sym = symbol.replace("/", "_").lower()
        return self.base_dir / f"{clean_sym}.{ext}"

    def export_buffer(self, buffer: MarketDataBuffer, filename: str = "snapshot.json") -> Path:
        """Export full buffer state to a snapshot file."""
        target = self.base_dir / filename
        save_buffer_snapshot(buffer, target)
        return target

    def import_buffer(self, filename: str = "snapshot.json") -> MarketDataBuffer:
        """Import snapshot file into fresh MarketDataBuffer."""
        target = self.base_dir / filename
        return load_buffer_snapshot(target)

    def append_ticks(self, symbol: str, ticks: List[MarketTick]) -> int:
        """Append ticks to symbol JSONL file."""
        target = self.get_symbol_path(symbol, "jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            for t in ticks:
                f.write(json.dumps(t.to_dict()) + "\n")
        return len(ticks)

    def read_ticks(self, symbol: str, limit: Optional[int] = None) -> List[MarketTick]:
        """Read ticks from symbol JSONL file."""
        target = self.get_symbol_path(symbol, "jsonl")
        if not target.exists():
            return []
        return load_ticks_from_jsonl(target, limit=limit)
