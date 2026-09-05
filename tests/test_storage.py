"""Comprehensive test suite for market telemetry persistence and historical replay subsystems."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from high_performance_model.storage.persistence import (
    MarketDataPersistence,
    load_bars_from_jsonl,
    load_buffer_snapshot,
    load_ticks_from_csv,
    load_ticks_from_jsonl,
    save_bars_to_jsonl,
    save_buffer_snapshot,
    save_ticks_to_csv,
    save_ticks_to_jsonl,
)
from high_performance_model.storage.replay import HistoricalReplayEngine
from high_performance_model.telemetry.buffer import MarketDataBuffer
from high_performance_model.types import (
    Bar,
    BarTimeframe,
    MarketTick,
    OrderBook,
    OrderBookLevel,
    Side,
)


@pytest.fixture
def sample_ticks() -> list[MarketTick]:
    base_ts = datetime(2026, 9, 6, 10, 0, 0, tzinfo=timezone.utc)
    return [
        MarketTick(symbol="AAPL", price=150.25, size=10.0, side=Side.BUY, sequence=1, timestamp=base_ts),
        MarketTick(symbol="AAPL", price=150.30, size=25.0, side=Side.BUY, sequence=2, timestamp=base_ts + timedelta(milliseconds=200)),
        MarketTick(symbol="AAPL", price=150.20, size=15.0, side=Side.SELL, sequence=3, timestamp=base_ts + timedelta(milliseconds=500)),
    ]


@pytest.fixture
def sample_bars() -> list[Bar]:
    base_ts = datetime(2026, 9, 6, 10, 0, 0, tzinfo=timezone.utc)
    return [
        Bar(
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.5,
            close=151.2,
            volume=5000.0,
            timeframe=BarTimeframe.MIN_1,
            timestamp=base_ts,
            vwap=150.8,
            trades_count=45,
        ),
        Bar(
            symbol="AAPL",
            open=151.2,
            high=153.0,
            low=150.8,
            close=152.5,
            volume=6200.0,
            timeframe=BarTimeframe.MIN_1,
            timestamp=base_ts + timedelta(minutes=1),
            vwap=152.1,
            trades_count=52,
        ),
    ]


class TestTickAndBarPersistence:
    """Test JSONL and CSV serialization/deserialization."""

    def test_jsonl_ticks_roundtrip(self, tmp_path: Path, sample_ticks: list[MarketTick]) -> None:
        file_path = tmp_path / "ticks.jsonl"
        saved_count = save_ticks_to_jsonl(sample_ticks, file_path)
        assert saved_count == len(sample_ticks)

        loaded = load_ticks_from_jsonl(file_path)
        assert len(loaded) == len(sample_ticks)
        assert loaded[0].symbol == "AAPL"
        assert loaded[0].price == 150.25
        assert loaded[0].side == Side.BUY

        # Test limit
        loaded_lim = load_ticks_from_jsonl(file_path, limit=2)
        assert len(loaded_lim) == 2

    def test_csv_ticks_roundtrip(self, tmp_path: Path, sample_ticks: list[MarketTick]) -> None:
        file_path = tmp_path / "ticks.csv"
        saved_count = save_ticks_to_csv(sample_ticks, file_path)
        assert saved_count == len(sample_ticks)

        loaded = load_ticks_from_csv(file_path)
        assert len(loaded) == len(sample_ticks)
        assert loaded[1].price == 150.30
        assert loaded[1].sequence == 2

        # Test limit
        loaded_lim = load_ticks_from_csv(file_path, limit=1)
        assert len(loaded_lim) == 1

    def test_jsonl_bars_roundtrip(self, tmp_path: Path, sample_bars: list[Bar]) -> None:
        file_path = tmp_path / "bars.jsonl"
        saved_count = save_bars_to_jsonl(sample_bars, file_path)
        assert saved_count == len(sample_bars)

        loaded = load_bars_from_jsonl(file_path)
        assert len(loaded) == len(sample_bars)
        assert loaded[0].open == 150.0
        assert loaded[0].close == 151.2
        assert loaded[0].timeframe == BarTimeframe.MIN_1

    def test_file_not_found_errors(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError):
            load_ticks_from_jsonl(missing)
        with pytest.raises(FileNotFoundError):
            load_ticks_from_csv(missing)
        with pytest.raises(FileNotFoundError):
            load_bars_from_jsonl(missing)


class TestBufferSnapshotPersistence:
    """Test full buffer state saving and restoring."""

    def test_buffer_snapshot_lifecycle(
        self, tmp_path: Path, sample_ticks: list[MarketTick], sample_bars: list[Bar]
    ) -> None:
        buf = MarketDataBuffer(capacity=1000)
        for t in sample_ticks:
            buf.push_tick(t)
        for b in sample_bars:
            buf.push_bar(b)

        ob = OrderBook(
            symbol="AAPL",
            bids=[OrderBookLevel(price=150.0, quantity=10.0)],
            asks=[OrderBookLevel(price=150.5, quantity=12.0)],
        )
        buf.push_order_book(ob)

        snapshot_file = tmp_path / "buffer_snapshot.json"
        meta = save_buffer_snapshot(buf, snapshot_file)
        assert meta["total_ticks"] == 3
        assert meta["total_bars"] == 2
        assert meta["total_order_books"] == 1

        # Restore into fresh buffer
        restored_buf = load_buffer_snapshot(snapshot_file)
        assert len(restored_buf.get_ticks("AAPL")) == 3
        assert len(restored_buf.get_bars("AAPL")) == 2
        restored_ob = restored_buf.get_latest_order_book("AAPL")
        assert restored_ob is not None
        assert restored_ob.best_bid == 150.0
        assert restored_ob.best_ask == 150.5

    def test_persistence_repository(self, tmp_path: Path, sample_ticks: list[MarketTick]) -> None:
        repo = MarketDataPersistence(base_directory=tmp_path)
        appended = repo.append_ticks("BTC/USD", sample_ticks)
        assert appended == len(sample_ticks)

        read_back = repo.read_ticks("BTC/USD")
        assert len(read_back) == len(sample_ticks)


class TestHistoricalReplayEngine:
    """Test streaming replay engine with speed controls and buffer injection."""

    def test_replay_state_and_seeking(self, sample_ticks: list[MarketTick]) -> None:
        engine = HistoricalReplayEngine(ticks=sample_ticks)
        assert engine.total_ticks == 3
        assert engine.cursor == 0
        assert engine.state.progress_pct == 0.0

        engine.seek(2)
        assert engine.cursor == 2
        engine.reset()
        assert engine.cursor == 0

    def test_instant_stream_replay(self, sample_ticks: list[MarketTick]) -> None:
        async def _run():
            engine = HistoricalReplayEngine(ticks=sample_ticks)
            collected: list[MarketTick] = []
            async for tick in engine.stream_ticks(speed_multiplier=0.0):
                collected.append(tick)

            assert len(collected) == 3
            assert engine.state.progress_pct == 100.0

        asyncio.run(_run())

    def test_replay_into_buffer(self, sample_ticks: list[MarketTick]) -> None:
        async def _run():
            buf = MarketDataBuffer(capacity=500)
            engine = HistoricalReplayEngine(ticks=sample_ticks)
            count = await engine.replay_into_buffer(buf, speed_multiplier=0.0)
            assert count == 3
            assert len(buf.get_ticks("AAPL")) == 3

        asyncio.run(_run())
