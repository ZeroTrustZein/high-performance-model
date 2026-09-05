"""Comprehensive unit tests for FinTech domain types, enums, data contracts, and validators."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from high_performance_model.types import (
    AssetClass,
    Bar,
    BarTimeframe,
    BenchmarkRunResult,
    IndicatorConfig,
    IndicatorType,
    LiquidityTier,
    MarketDepthSnapshot,
    MarketState,
    MarketTelemetrySummary,
    MarketTick,
    OrderBook,
    OrderBookLevel,
    OrderType,
    Quote,
    RiskMetrics,
    Side,
    TechnicalIndicatorResult,
    TimeInForce,
    TradeExecution,
    compute_microprice,
    compute_spread_bps,
    timeframe_to_seconds,
    validate_symbol,
)


class TestEnums:
    """Test domain enumerations values, string subtyping, and membership."""

    def test_side_enum(self) -> None:
        assert Side.BUY == "buy"
        assert Side.SELL == "sell"
        assert issubclass(Side, str)
        assert set(Side) == {Side.BUY, Side.SELL}

    def test_order_type_enum(self) -> None:
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP == "stop"
        assert OrderType.STOP_LIMIT == "stop_limit"
        assert OrderType.TRAILING_STOP == "trailing_stop"
        assert issubclass(OrderType, str)

    def test_time_in_force_enum(self) -> None:
        assert TimeInForce.GTC == "GTC"
        assert TimeInForce.IOC == "IOC"
        assert TimeInForce.FOK == "FOK"
        assert TimeInForce.DAY == "DAY"
        assert TimeInForce.GTD == "GTD"
        assert issubclass(TimeInForce, str)

    def test_asset_class_enum(self) -> None:
        assert AssetClass.EQUITY == "equity"
        assert AssetClass.CRYPTO == "crypto"
        assert AssetClass.FX == "fx"
        assert AssetClass.COMMODITY == "commodity"
        assert AssetClass.INDEX == "index"
        assert AssetClass.OPTION == "option"
        assert AssetClass.FUTURE == "future"
        assert issubclass(AssetClass, str)

    def test_indicator_type_enum(self) -> None:
        assert IndicatorType.SMA == "sma"
        assert IndicatorType.EMA == "ema"
        assert IndicatorType.RSI == "rsi"
        assert IndicatorType.MACD == "macd"
        assert IndicatorType.BOLLINGER == "bollinger"
        assert IndicatorType.ATR == "atr"
        assert IndicatorType.VWAP == "vwap"
        assert IndicatorType.MOMENTUM == "momentum"
        assert IndicatorType.ROC == "roc"
        assert IndicatorType.STOCHASTIC == "stochastic"
        assert issubclass(IndicatorType, str)

    def test_bar_timeframe_enum(self) -> None:
        assert BarTimeframe.SEC_1 == "1s"
        assert BarTimeframe.MIN_1 == "1m"
        assert BarTimeframe.MIN_5 == "5m"
        assert BarTimeframe.MIN_15 == "15m"
        assert BarTimeframe.MIN_30 == "30m"
        assert BarTimeframe.HOUR_1 == "1h"
        assert BarTimeframe.HOUR_4 == "4h"
        assert BarTimeframe.DAY_1 == "1d"
        assert BarTimeframe.WEEK_1 == "1w"
        assert issubclass(BarTimeframe, str)

    def test_market_state_enum(self) -> None:
        assert MarketState.PRE_MARKET == "pre_market"
        assert MarketState.OPEN == "open"
        assert MarketState.POST_MARKET == "post_market"
        assert MarketState.CLOSED == "closed"
        assert MarketState.HALTED == "halted"
        assert issubclass(MarketState, str)

    def test_liquidity_tier_enum(self) -> None:
        assert LiquidityTier.TIER_1 == "tier_1"
        assert LiquidityTier.TIER_2 == "tier_2"
        assert LiquidityTier.TIER_3 == "tier_3"
        assert LiquidityTier.TIER_4 == "tier_4"
        assert LiquidityTier.ILLIQUID == "illiquid"
        assert issubclass(LiquidityTier, str)


class TestHelperFunctions:
    """Test validation and mathematical domain helper functions."""

    def test_validate_symbol_valid(self) -> None:
        assert validate_symbol("aapl") == "AAPL"
        assert validate_symbol("  msft  ") == "MSFT"
        assert validate_symbol("BTC/USD") == "BTC/USD"
        assert validate_symbol("EUR-USD") == "EUR-USD"
        assert validate_symbol("BRK.B") == "BRK.B"
        assert validate_symbol("ETH_USDT") == "ETH_USDT"

    def test_validate_symbol_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Symbol must be a string"):
            validate_symbol(123)  # type: ignore[arg-type]

    def test_validate_symbol_empty(self) -> None:
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            validate_symbol("")
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            validate_symbol("   ")

    def test_validate_symbol_invalid_characters(self) -> None:
        with pytest.raises(ValueError, match="Invalid ticker symbol format"):
            validate_symbol("AAPL$")
        with pytest.raises(ValueError, match="Invalid ticker symbol format"):
            validate_symbol("AAPL MSFT")
        with pytest.raises(ValueError, match="Invalid ticker symbol format"):
            validate_symbol("A" * 21)

    def test_timeframe_to_seconds(self) -> None:
        assert timeframe_to_seconds(BarTimeframe.SEC_1) == 1
        assert timeframe_to_seconds(BarTimeframe.SEC_5) == 5
        assert timeframe_to_seconds(BarTimeframe.SEC_15) == 15
        assert timeframe_to_seconds(BarTimeframe.SEC_30) == 30
        assert timeframe_to_seconds(BarTimeframe.MIN_1) == 60
        assert timeframe_to_seconds(BarTimeframe.MIN_5) == 300
        assert timeframe_to_seconds(BarTimeframe.MIN_15) == 900
        assert timeframe_to_seconds(BarTimeframe.MIN_30) == 1800
        assert timeframe_to_seconds(BarTimeframe.HOUR_1) == 3600
        assert timeframe_to_seconds(BarTimeframe.HOUR_4) == 14400
        assert timeframe_to_seconds(BarTimeframe.DAY_1) == 86400
        assert timeframe_to_seconds(BarTimeframe.WEEK_1) == 604800

        # String format acceptance
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds(" 1H ") == 3600

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            timeframe_to_seconds("2m")

    def test_compute_spread_bps(self) -> None:
        # mid = 100.5, spread = 1.0 -> 1.0 / 100.5 * 10,000 = 99.5025
        bps = compute_spread_bps(100.0, 101.0)
        assert pytest.approx(bps, 0.001) == 99.5025

        # Zero spread
        assert compute_spread_bps(150.0, 150.0) == 0.0

        # Non-positive prices
        with pytest.raises(ValueError, match="strictly positive"):
            compute_spread_bps(0.0, 100.0)
        with pytest.raises(ValueError, match="strictly positive"):
            compute_spread_bps(-5.0, 100.0)

        # Inverted bid/ask
        with pytest.raises(ValueError, match="cannot be strictly lower"):
            compute_spread_bps(105.0, 100.0)

    def test_compute_microprice(self) -> None:
        # Equal depth returns exact mid-price
        assert compute_microprice(100.0, 102.0, 50.0, 50.0) == 101.0

        # Asymmetric depth: more ask volume pushes microprice closer to bid
        # (100 * 100 + 102 * 10) / 110 = (10000 + 1020) / 110 = 100.181818
        p = compute_microprice(100.0, 102.0, bid_qty=10.0, ask_qty=100.0)
        assert pytest.approx(p, 0.0001) == 100.181818

        # Zero quantities fallback to mid
        assert compute_microprice(100.0, 102.0, 0.0, 0.0) == 101.0

        # Invalid prices or quantities
        with pytest.raises(ValueError, match="strictly positive"):
            compute_microprice(0.0, 100.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="non-negative"):
            compute_microprice(100.0, 102.0, -1.0, 1.0)


class TestMarketTick:
    """Test MarketTick domain model validation, methods, and edge cases."""

    def test_valid_tick_and_normalization(self) -> None:
        tick = MarketTick(symbol="nvda", price=125.50, size=50.0, side=Side.BUY)
        assert tick.symbol == "NVDA"
        assert tick.price == 125.50
        assert tick.size == 50.0
        assert tick.side == Side.BUY
        assert tick.sequence == 0
        assert tick.notional == 6275.0
        assert tick.timestamp.tzinfo is not None

    def test_tick_uptick_logic(self) -> None:
        t1 = MarketTick(symbol="AAPL", price=150.0, size=10.0, side=Side.BUY)
        assert t1.is_uptick(None) is None

        t2 = MarketTick(symbol="AAPL", price=150.25, size=10.0, side=Side.BUY)
        assert t2.is_uptick(t1) is True

        t3 = MarketTick(symbol="AAPL", price=149.75, size=10.0, side=Side.SELL)
        assert t3.is_uptick(t2) is False

        t4 = MarketTick(symbol="AAPL", price=149.75, size=5.0, side=Side.SELL)
        assert t4.is_uptick(t3) is None

    def test_tick_to_dict(self) -> None:
        now = datetime.now(timezone.utc)
        tick = MarketTick(
            symbol="BTC/USD",
            price=60000.0,
            size=1.5,
            side=Side.SELL,
            timestamp=now,
            sequence=42,
            exchange="COINBASE",
            conditions=["REGULAR"],
        )
        d = tick.to_dict()
        assert d["symbol"] == "BTC/USD"
        assert d["price"] == 60000.0
        assert d["size"] == 1.5
        assert d["side"] == "sell"
        assert d["sequence"] == 42
        assert d["exchange"] == "COINBASE"
        assert d["conditions"] == ["REGULAR"]
        assert d["notional"] == 90000.0
        assert d["timestamp"] == now.isoformat()

    def test_tick_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            MarketTick(symbol="AAPL", price=-1.0, size=10.0, side=Side.BUY)

        with pytest.raises(ValidationError):
            MarketTick(symbol="AAPL", price=0.0, size=10.0, side=Side.BUY)

        with pytest.raises(ValidationError):
            MarketTick(symbol="AAPL", price=100.0, size=-5.0, side=Side.BUY)

        with pytest.raises(ValidationError):
            MarketTick(symbol="INVALID$$", price=100.0, size=5.0, side=Side.BUY)

        with pytest.raises(ValidationError):
            MarketTick(symbol="AAPL", price=100.0, size=5.0, side=Side.BUY, sequence=-1)


class TestOrderBookLevel:
    """Test OrderBookLevel depth level domain model."""

    def test_level_valid_and_methods(self) -> None:
        lvl = OrderBookLevel(price=105.50, quantity=200.0, orders_count=4)
        assert lvl.price == 105.50
        assert lvl.quantity == 200.0
        assert lvl.orders_count == 4
        assert lvl.notional == 21100.0
        assert lvl.to_tuple() == (105.50, 200.0, 4)

        d = lvl.to_dict()
        assert d["price"] == 105.50
        assert d["quantity"] == 200.0
        assert d["orders_count"] == 4
        assert d["notional"] == 21100.0

    def test_level_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            OrderBookLevel(price=0.0, quantity=10.0)
        with pytest.raises(ValidationError):
            OrderBookLevel(price=-10.0, quantity=10.0)
        with pytest.raises(ValidationError):
            OrderBookLevel(price=100.0, quantity=-1.0)
        with pytest.raises(ValidationError):
            OrderBookLevel(price=100.0, quantity=10.0, orders_count=-1)


class TestOrderBook:
    """Test OrderBook depth model, imbalance, microprice, and cross detection."""

    def test_empty_order_book(self) -> None:
        ob = OrderBook(symbol="aapl")
        assert ob.symbol == "AAPL"
        assert ob.bids == []
        assert ob.asks == []
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.spread is None
        assert ob.spread_bps is None
        assert ob.mid_price is None
        assert ob.microprice is None
        assert ob.order_book_imbalance == 0.0
        assert ob.total_bid_volume == 0.0
        assert ob.total_ask_volume == 0.0
        assert ob.total_bid_notional == 0.0
        assert ob.total_ask_notional == 0.0
        assert ob.is_crossed is False

    def test_populated_order_book_metrics(self) -> None:
        bids = [
            OrderBookLevel(price=100.0, quantity=10.0),
            OrderBookLevel(price=99.5, quantity=20.0),
        ]
        asks = [
            OrderBookLevel(price=101.0, quantity=10.0),
            OrderBookLevel(price=101.5, quantity=30.0),
        ]
        ob = OrderBook(symbol="NVDA", bids=bids, asks=asks)

        assert ob.best_bid == 100.0
        assert ob.best_ask == 101.0
        assert ob.spread == 1.0
        assert ob.mid_price == 100.5
        assert pytest.approx(ob.spread_bps, 0.01) == 99.5025
        assert ob.microprice == 100.5
        assert ob.total_bid_volume == 30.0
        assert ob.total_ask_volume == 40.0
        assert ob.total_bid_notional == (100.0 * 10.0 + 99.5 * 20.0)
        assert ob.total_ask_notional == (101.0 * 10.0 + 101.5 * 30.0)
        assert ob.is_crossed is False

        # Imbalance: (30 - 40) / 70 = -10 / 70 = -0.1429
        assert ob.order_book_imbalance == -0.1429

    def test_order_book_crossed_detection(self) -> None:
        bids = [OrderBookLevel(price=105.0, quantity=10.0)]
        asks = [OrderBookLevel(price=104.0, quantity=10.0)]
        ob = OrderBook(symbol="AAPL", bids=bids, asks=asks)
        assert ob.is_crossed is True

    def test_order_book_depth_summary_and_dict(self) -> None:
        bids = [OrderBookLevel(price=100.0, quantity=15.0)]
        asks = [OrderBookLevel(price=101.0, quantity=25.0)]
        ob = OrderBook(symbol="GOOGL", bids=bids, asks=asks, sequence=12)

        summary = ob.depth_summary(levels=5)
        assert summary["symbol"] == "GOOGL"
        assert summary["best_bid"] == 100.0
        assert summary["best_ask"] == 101.0
        assert summary["total_bid_volume"] == 15.0
        assert summary["total_ask_volume"] == 25.0

        d = ob.to_dict()
        assert d["symbol"] == "GOOGL"
        assert d["sequence"] == 12
        assert len(d["bids"]) == 1
        assert len(d["asks"]) == 1
        assert d["best_bid"] == 100.0


class TestBar:
    """Test Bar candlestick domain model and geometry validations."""

    def test_valid_bullish_and_bearish_bars(self) -> None:
        # Bullish bar
        bull = Bar(
            symbol="spy",
            open=500.0,
            high=505.0,
            low=499.0,
            close=504.0,
            volume=100_000.0,
            timeframe=BarTimeframe.MIN_5,
            vwap=502.5,
            trades_count=1200,
        )
        assert bull.symbol == "SPY"
        assert bull.is_bullish is True
        assert bull.is_bearish is False
        assert bull.range == 6.0
        assert bull.body == 4.0
        assert bull.typical_price == round((505.0 + 499.0 + 504.0) / 3.0, 6)
        assert bull.timeframe == BarTimeframe.MIN_5

        # Bearish bar
        bear = Bar(
            symbol="SPY",
            open=504.0,
            high=505.0,
            low=495.0,
            close=497.0,
            volume=80_000.0,
        )
        assert bear.is_bullish is False
        assert bear.is_bearish is True
        assert bear.range == 10.0
        assert bear.body == 7.0

    def test_bar_geometry_validation_errors(self) -> None:
        # high < low
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", open=100.0, high=90.0, low=95.0, close=92.0, volume=100.0)

        # high < open
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", open=105.0, high=100.0, low=95.0, close=98.0, volume=100.0)

        # high < close
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", open=98.0, high=100.0, low=95.0, close=102.0, volume=100.0)

        # low > open
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", open=90.0, high=105.0, low=95.0, close=100.0, volume=100.0)

        # low > close
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", open=100.0, high=105.0, low=95.0, close=92.0, volume=100.0)

        # negative volume
        with pytest.raises(ValidationError):
            Bar(symbol="AAPL", open=100.0, high=105.0, low=95.0, close=102.0, volume=-10.0)

    def test_bar_to_dict(self) -> None:
        bar = Bar(symbol="AAPL", open=100.0, high=105.0, low=98.0, close=102.0, volume=500.0)
        d = bar.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["open"] == 100.0
        assert d["range"] == 7.0
        assert d["body"] == 2.0
        assert d["timeframe"] == "1m"


class TestQuote:
    """Test top-of-book Quote model and validations."""

    def test_valid_quote_and_metrics(self) -> None:
        q = Quote(
            symbol="msft",
            bid=420.0,
            ask=420.5,
            bid_size=100.0,
            ask_size=150.0,
            last_price=420.25,
            last_size=10.0,
        )
        assert q.symbol == "MSFT"
        assert q.spread == 0.5
        assert q.mid_price == 420.25
        assert pytest.approx(q.spread_bps, 0.01) == 11.8977

        d = q.to_dict()
        assert d["symbol"] == "MSFT"
        assert d["spread"] == 0.5
        assert d["mid_price"] == 420.25
        assert d["last_price"] == 420.25

    def test_quote_invalid_prices(self) -> None:
        # ask < bid
        with pytest.raises(ValidationError):
            Quote(symbol="MSFT", bid=420.0, ask=419.0, bid_size=10.0, ask_size=10.0)

        # bid <= 0
        with pytest.raises(ValidationError):
            Quote(symbol="MSFT", bid=0.0, ask=100.0, bid_size=10.0, ask_size=10.0)


class TestMarketDepthSnapshot:
    """Test MarketDepthSnapshot model."""

    def test_snapshot_lifecycle(self) -> None:
        bids = [OrderBookLevel(price=100.0, quantity=10.0)]
        asks = [OrderBookLevel(price=101.0, quantity=15.0)]
        snap = MarketDepthSnapshot(
            symbol="aapl",
            bids=bids,
            asks=asks,
            spread=1.0,
            mid_price=100.5,
            imbalance=-0.2,
        )
        assert snap.symbol == "AAPL"
        assert snap.spread == 1.0
        assert snap.imbalance == -0.2
        d = snap.to_dict()
        assert d["symbol"] == "AAPL"
        assert len(d["bids"]) == 1
        assert len(d["asks"]) == 1


class TestIndicatorConfig:
    """Test IndicatorConfig configuration parameters and validations."""

    def test_valid_indicator_config(self) -> None:
        cfg = IndicatorConfig(indicator=IndicatorType.SMA, period=30)
        assert cfg.indicator == IndicatorType.SMA
        assert cfg.period == 30

    def test_macd_parameter_validation(self) -> None:
        valid_macd = IndicatorConfig(
            indicator=IndicatorType.MACD,
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )
        assert valid_macd.fast_period == 12

        # fast_period >= slow_period must fail
        with pytest.raises(ValidationError):
            IndicatorConfig(indicator=IndicatorType.MACD, fast_period=26, slow_period=26)

        with pytest.raises(ValidationError):
            IndicatorConfig(indicator=IndicatorType.MACD, fast_period=30, slow_period=20)


class TestTechnicalIndicatorResult:
    """Test TechnicalIndicatorResult data model."""

    def test_result_values_and_getter(self) -> None:
        res = TechnicalIndicatorResult(
            symbol="btc/usd",
            indicator="bollinger",
            values={"upper": 65000.0, "middle": 62000.0, "lower": 59000.0},
            metadata={"period": 20, "std": 2.0},
        )
        assert res.symbol == "BTC/USD"
        assert res.get_value("upper") == 65000.0
        assert res.get_value("nonexistent") is None
        assert res.get_value("nonexistent", 0.0) == 0.0

        d = res.to_dict()
        assert d["symbol"] == "BTC/USD"
        assert d["values"]["upper"] == 65000.0
        assert d["metadata"]["period"] == 20


class TestRiskMetrics:
    """Test RiskMetrics quantitative contract."""

    def test_valid_risk_metrics(self) -> None:
        metrics = RiskMetrics(
            symbol="eth/usd",
            realized_volatility=0.65,
            sharpe_ratio=1.45,
            max_drawdown=-0.18,
            value_at_risk_95=-0.045,
            expected_shortfall_95=-0.062,
            sample_size=500,
            metadata={"annualized_days": 365},
        )
        assert metrics.symbol == "ETH/USD"
        assert metrics.realized_volatility == 0.65
        assert metrics.max_drawdown == -0.18

        d = metrics.to_dict()
        assert d["symbol"] == "ETH/USD"
        assert d["max_drawdown"] == -0.18
        assert d["sample_size"] == 500

    def test_risk_metrics_validation_errors(self) -> None:
        # Negative volatility
        with pytest.raises(ValidationError):
            RiskMetrics(
                symbol="AAPL",
                realized_volatility=-0.1,
                max_drawdown=-0.05,
                sample_size=100,
            )

        # Positive max drawdown (drawdown must be <= 0)
        with pytest.raises(ValidationError):
            RiskMetrics(
                symbol="AAPL",
                realized_volatility=0.2,
                max_drawdown=0.05,
                sample_size=100,
            )

        # Negative sample size
        with pytest.raises(ValidationError):
            RiskMetrics(
                symbol="AAPL",
                realized_volatility=0.2,
                max_drawdown=-0.05,
                sample_size=-10,
            )


class TestMarketTelemetrySummary:
    """Test MarketTelemetrySummary model."""

    def test_telemetry_summary_valid(self) -> None:
        summary = MarketTelemetrySummary(
            symbol="nvda",
            last_price=128.5,
            last_size=40.0,
            last_side=Side.BUY,
            volume_24h=5_000_000.0,
            vwap=127.8,
            high_24h=130.0,
            low_24h=125.0,
            spread=0.02,
            change_24h_pct=2.8,
            market_state=MarketState.OPEN,
        )
        assert summary.symbol == "NVDA"
        assert summary.last_price == 128.5
        assert summary.market_state == MarketState.OPEN

        d = summary.to_dict()
        assert d["symbol"] == "NVDA"
        assert d["last_side"] == "buy"
        assert d["market_state"] == "open"
        assert d["volume_24h"] == 5_000_000.0

    def test_telemetry_summary_validation(self) -> None:
        with pytest.raises(ValidationError):
            MarketTelemetrySummary(
                symbol="NVDA",
                last_price=-10.0,
                last_size=1.0,
                last_side=Side.BUY,
            )


class TestTradeExecution:
    """Test TradeExecution fill model."""

    def test_trade_execution_valid(self) -> None:
        trade = TradeExecution(
            trade_id="TRD-1001",
            symbol="googl",
            price=175.25,
            size=100.0,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            maker_order_id="ORD-01",
            taker_order_id="ORD-02",
            fee=0.35,
        )
        assert trade.symbol == "GOOGL"
        assert trade.notional == 17525.0
        d = trade.to_dict()
        assert d["trade_id"] == "TRD-1001"
        assert d["order_type"] == "limit"
        assert d["notional"] == 17525.0

    def test_trade_execution_validation(self) -> None:
        with pytest.raises(ValidationError):
            TradeExecution(
                trade_id="TRD-1",
                symbol="GOOGL",
                price=100.0,
                size=0.0,  # size must be gt 0
                side=Side.BUY,
            )


class TestBenchmarkRunResult:
    """Test BenchmarkRunResult performance contract."""

    def test_benchmark_result_valid(self) -> None:
        bench = BenchmarkRunResult(
            total_ticks=100_000,
            duration_seconds=0.25,
            ticks_per_second=400_000.0,
            latency_p50_us=1.2,
            latency_p95_us=3.8,
            latency_p99_us=6.5,
            buffer_utilization_pct=45.5,
        )
        assert bench.total_ticks == 100_000
        d = bench.to_dict()
        assert d["total_ticks"] == 100_000
        assert d["ticks_per_second"] == 400_000.0

    def test_benchmark_result_validation(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRunResult(
                total_ticks=-1,
                duration_seconds=1.0,
                ticks_per_second=1.0,
                latency_p50_us=1.0,
                latency_p95_us=1.0,
                latency_p99_us=1.0,
                buffer_utilization_pct=50.0,
            )
        with pytest.raises(ValidationError):
            BenchmarkRunResult(
                total_ticks=100,
                duration_seconds=1.0,
                ticks_per_second=1.0,
                latency_p50_us=1.0,
                latency_p95_us=1.0,
                latency_p99_us=1.0,
                buffer_utilization_pct=105.0,  # > 100%
            )


class TestSerializationRoundtrips:
    """Test JSON serialization and deserialization roundtrips across all domain types."""

    def test_market_tick_roundtrip(self) -> None:
        tick = MarketTick(symbol="NVDA", price=125.5, size=10.0, side=Side.BUY)
        json_str = tick.model_dump_json()
        loaded = MarketTick.model_validate_json(json_str)
        assert loaded.symbol == tick.symbol
        assert loaded.price == tick.price
        assert loaded.side == tick.side

    def test_order_book_roundtrip(self) -> None:
        bids = [OrderBookLevel(price=100.0, quantity=5.0)]
        asks = [OrderBookLevel(price=101.0, quantity=8.0)]
        ob = OrderBook(symbol="NVDA", bids=bids, asks=asks)
        json_str = ob.model_dump_json()
        loaded = OrderBook.model_validate_json(json_str)
        assert loaded.symbol == ob.symbol
        assert loaded.best_bid == 100.0
        assert loaded.best_ask == 101.0

    def test_bar_roundtrip(self) -> None:
        bar = Bar(symbol="AAPL", open=150.0, high=155.0, low=148.0, close=153.0, volume=1000.0)
        json_str = bar.model_dump_json()
        loaded = Bar.model_validate_json(json_str)
        assert loaded.symbol == "AAPL"
        assert loaded.high == 155.0

    def test_quote_roundtrip(self) -> None:
        q = Quote(symbol="AAPL", bid=150.0, ask=150.5, bid_size=10.0, ask_size=20.0)
        json_str = q.model_dump_json()
        loaded = Quote.model_validate_json(json_str)
        assert loaded.spread == 0.5
