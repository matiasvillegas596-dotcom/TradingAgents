"""The autonomous trader: ratings become sized orders, and the risk rules hold.

The graph and the prices are stand-ins, so these run offline and pin the
trading rules rather than any model's judgment.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tradingagents.autotrader import AutoTrader, AutoTraderConfig, PaperBroker
from tradingagents.autotrader.broker import AlpacaBroker
from tradingagents.autotrader.risk import MarketView, plan_rebalance

NY = ZoneInfo("America/New_York")


def _bars(last: float, daily_move: float = 0.01, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = last * np.cumprod(1 + rng.normal(0, daily_move, n))
    closes = closes * last / closes[-1]
    return pd.DataFrame({"Date": pd.date_range(end="2026-10-27", periods=n), "Close": closes,
                         "High": closes * 1.01, "Low": closes * 0.99})


class Market:
    def __init__(self, prices: dict[str, float]):
        self.prices = prices

    def bars_of(self, ticker, _as_of):
        return _bars(self.prices[ticker])

    def price_of(self, ticker):
        return self.prices[ticker]


def _trader(tmp_path, ratings, prices, **overrides):
    config = AutoTraderConfig(watchlist=list(ratings), state_dir=str(tmp_path), slippage_bps=0,
                              commission_bps=0, **overrides)
    market = Market(prices)
    broker = PaperBroker(tmp_path / "ledger.json", market.price_of, starting_cash=100_000,
                         commission_bps=0, slippage_bps=0)
    calls = []

    def propagate(ticker, date, asset_type, portfolio):
        calls.append((ticker, date, portfolio))
        return {}, ratings[ticker]

    trader = AutoTrader(config, broker, propagate, market.bars_of,
                        now=lambda: datetime(2026, 10, 27, 15, 30, tzinfo=NY))
    return trader, broker, market, calls


@pytest.mark.unit
def test_buy_rating_opens_a_capped_position_and_hold_trades_nothing(tmp_path):
    trader, broker, _, calls = _trader(tmp_path, {"AAA": "Buy", "BBB": "Hold"}, {"AAA": 100.0, "BBB": 50.0})
    plans = trader.run_analysis()
    assert [(p.ticker, p.side) for p in plans] == [("AAA", "buy")]
    held = broker.holdings["AAA"].quantity * 100.0
    assert 0 < held <= 0.15 * 100_000
    assert calls[0][2].cash == 100_000  # the agents saw the live book


@pytest.mark.unit
def test_sell_rating_closes_the_whole_position(tmp_path):
    trader, broker, _, _ = _trader(tmp_path, {"AAA": "Sell"}, {"AAA": 100.0})
    broker.submit_market_order("AAA", "buy", 50)
    plans = trader.run_analysis()
    assert [(p.side, p.quantity) for p in plans] == [("sell", 50)]
    assert "AAA" not in broker.holdings


@pytest.mark.unit
def test_review_is_never_traded(tmp_path):
    trader, broker, _, _ = _trader(tmp_path, {"AAA": "REVIEW"}, {"AAA": 100.0})
    assert trader.run_analysis() == []
    assert broker.holdings == {}


@pytest.mark.unit
def test_trailing_stop_exits_without_the_agents(tmp_path):
    trader, broker, market, calls = _trader(tmp_path, {"AAA": "Buy"}, {"AAA": 100.0})
    trader.run_analysis()
    calls.clear()
    market.prices["AAA"] = 80.0
    exits = trader.check_exits()
    assert [e.ticker for e in exits] == ["AAA"] and "trailing stop" in exits[0].reason
    assert "AAA" not in broker.holdings and calls == []


@pytest.mark.unit
def test_drawdown_halts_new_buys_until_reset(tmp_path):
    trader, broker, market, _ = _trader(tmp_path, {"AAA": "Hold", "BBB": "Buy"},
                                        {"AAA": 100.0, "BBB": 10.0}, max_drawdown_pct=0.1,
                                        max_daily_loss_pct=0.5)
    broker.submit_market_order("AAA", "buy", 900)
    trader.check_exits()  # records the peak
    market.prices["AAA"] = 85.0  # equity falls about 13.5%
    assert trader.run_analysis() == []
    assert trader.state["halted"] is True
    trader.reset_halt()
    assert [p.ticker for p in trader.run_analysis()] == ["BBB"]


@pytest.mark.unit
def test_paper_ledger_survives_a_restart(tmp_path):
    prices = {"AAA": 100.0}
    broker = PaperBroker(tmp_path / "ledger.json", prices.get, starting_cash=10_000, slippage_bps=10)
    broker.submit_market_order("AAA", "buy", 10)
    reopened = PaperBroker(tmp_path / "ledger.json", prices.get, starting_cash=1)
    assert reopened.holdings["AAA"].quantity == 10
    assert reopened.cash == pytest.approx(broker.cash)
    assert reopened.cash < 10_000 - 1000  # slippage and commission were charged
    with pytest.raises(ValueError):
        reopened.submit_market_order("AAA", "sell", 11)


@pytest.mark.unit
def test_volatile_names_get_smaller_positions():
    config = AutoTraderConfig(watchlist=["X"])
    calm = MarketView.from_bars(_bars(100.0, daily_move=0.005))
    wild = MarketView.from_bars(_bars(100.0, daily_move=0.05))
    size = {name: plan_rebalance("X", "Buy", 0, 100_000, 100_000, 0, view, config, True).quantity
            for name, view in {"calm": calm, "wild": wild}.items()}
    assert size["wild"] < size["calm"]


@pytest.mark.unit
def test_live_trading_needs_the_environment_guard(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.delenv("TRADINGAGENTS_ALLOW_LIVE", raising=False)
    with pytest.raises(ValueError, match="TRADINGAGENTS_ALLOW_LIVE"):
        AlpacaBroker(live=True)
    assert AlpacaBroker(live=False).base_url.startswith("https://paper-api")


@pytest.mark.unit
def test_example_config_is_valid():
    from pathlib import Path

    from tradingagents.autotrader import load_autotrader_config

    path = Path(__file__).resolve().parents[1] / "autotrader.example.json"
    config = load_autotrader_config(path)
    assert config.broker == "paper" and not config.live
    assert json.loads(path.read_text())["watchlist"] == config.watchlist
