"""Historical market tick replay engine for backtesting and streaming simulation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional, Union

from high_performance_model.storage.persistence import (
    load_ticks_from_csv,
    load_ticks_from_jsonl,
)
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.types import MarketTick


@dataclass
class ReplayState:
    """Current status snapshot of the historical replay engine."""

    total_ticks: int
    played_ticks: int
    is_running: bool
    is_paused: bool
    current_symbol: Optional[str] = None
    current_timestamp: Optional[datetime] = None
    speed_multiplier: float = 1.0

    @property
    def progress_pct(self) -> float:
        """Percentage of replay completed."""
        if self.total_ticks == 0:
            return 100.0
        return round((self.played_ticks / self.total_ticks) * 100.0, 2)


class HistoricalReplayEngine:
    """Asynchronous market tick replay engine with speed controls and buffer injection."""

    def __init__(
        self,
        ticks: Optional[List[MarketTick]] = None,
        source_file: Optional[Union[str, Path]] = None,
    ) -> None:
        if ticks is not None:
            self._ticks = list(ticks)
        elif source_file is not None:
            p = Path(source_file)
            ext = p.suffix.lower()
            if ext == ".csv":
                self._ticks = load_ticks_from_csv(p)
            elif ext in {".jsonl", ".json"}:
                self._ticks = load_ticks_from_jsonl(p)
            else:
                raise ValueError(f"Unsupported file extension: {p.suffix}")
        else:
            self._ticks = []

        # Sort chronologically by timestamp
        self._ticks.sort(key=lambda t: t.timestamp)
        self._cursor = 0
        self._is_running = False
        self._is_paused = False
        self._stop_requested = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()

    @property
    def total_ticks(self) -> int:
        return len(self._ticks)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def state(self) -> ReplayState:
        curr_tick = self._ticks[self._cursor - 1] if 0 < self._cursor <= len(self._ticks) else None
        return ReplayState(
            total_ticks=len(self._ticks),
            played_ticks=self._cursor,
            is_running=self._is_running,
            is_paused=self._is_paused,
            current_symbol=curr_tick.symbol if curr_tick else None,
            current_timestamp=curr_tick.timestamp if curr_tick else None,
        )

    def pause(self) -> None:
        """Pause playback stream."""
        if self._is_running and not self._is_paused:
            self._is_paused = True
            self._pause_event.clear()

    def resume(self) -> None:
        """Resume playback stream."""
        if self._is_running and self._is_paused:
            self._is_paused = False
            self._pause_event.set()

    def stop(self) -> None:
        """Stop replay immediately."""
        self._stop_requested = True
        self._is_running = False
        self._is_paused = False
        self._pause_event.set()

    def seek(self, index: int) -> int:
        """Move playback cursor to specified tick index."""
        clamped = max(0, min(index, len(self._ticks)))
        self._cursor = clamped
        return self._cursor

    def reset(self) -> None:
        """Reset replay position to beginning."""
        self._cursor = 0
        self._stop_requested = False
        self._is_paused = False
        self._pause_event.set()

    async def stream_ticks(
        self,
        speed_multiplier: float = 1.0,
        max_ticks: Optional[int] = None,
        on_tick: Optional[Callable[[MarketTick], None]] = None,
    ) -> AsyncIterator[MarketTick]:
        """Asynchronously stream ticks with realistic inter-arrival delays."""
        self._is_running = True
        self._stop_requested = False
        self._pause_event.set()

        ticks_streamed = 0
        prev_tick: Optional[MarketTick] = None

        try:
            while self._cursor < len(self._ticks) and not self._stop_requested:
                if max_ticks and ticks_streamed >= max_ticks:
                    break

                # Handle pause
                await self._pause_event.wait()
                if self._stop_requested:
                    break

                tick = self._ticks[self._cursor]
                self._cursor += 1
                ticks_streamed += 1

                # Calculate delay between ticks if speed_multiplier > 0
                if speed_multiplier > 0 and prev_tick is not None:
                    delta_sec = (tick.timestamp - prev_tick.timestamp).total_seconds()
                    if delta_sec > 0:
                        # Cap max sleep to 1.0 sec to avoid hanging on long gaps
                        sleep_time = min(1.0, delta_sec / speed_multiplier)
                        if sleep_time > 0.001:
                            await asyncio.sleep(sleep_time)

                prev_tick = tick
                if on_tick:
                    on_tick(tick)

                yield tick

        finally:
            self._is_running = False

    async def replay_into_buffer(
        self,
        buffer: MarketDataBuffer,
        speed_multiplier: float = 0.0,
        max_ticks: Optional[int] = None,
    ) -> int:
        """Stream replayed ticks directly into a target MarketDataBuffer."""
        count = 0
        async for tick in self.stream_ticks(speed_multiplier=speed_multiplier, max_ticks=max_ticks):
            buffer.push_tick(tick)
            count += 1
        return count
