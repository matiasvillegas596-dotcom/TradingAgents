"""Turning a rating into an order, and deciding when to get out without one.

Deterministic and LLM-free: the agents choose a direction, these rules choose
how much and when to cut. Sizing is volatility-scaled so a quiet name and a
wild one carry similar risk, and capped per name and in total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from tradingagents.autotrader.config import AutoTraderConfig

TRADING_DAYS = 252


@dataclass
class MarketView:
    """What the risk rules need about one instrument, from its daily bars."""

    price: float
    atr: float
    annual_vol: float

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, window: int = 14) -> MarketView:
        bars = bars.dropna(subset=["Close", "High", "Low"]).tail(max(window, 60) + 1)
        if len(bars) < window + 1:
            raise ValueError(f"need {window + 1} daily bars to size a trade, have {len(bars)}")
        close, prev = bars["Close"], bars["Close"].shift()
        true_range = pd.concat([bars["High"] - bars["Low"], (bars["High"] - prev).abs(),
                                (bars["Low"] - prev).abs()], axis=1).max(axis=1)
        returns = close.pct_change().dropna()
        return cls(price=float(close.iloc[-1]), atr=float(true_range.tail(window).mean()),
                   annual_vol=float(returns.std() * math.sqrt(TRADING_DAYS)))


@dataclass
class OrderPlan:
    ticker: str
    side: str
    quantity: float
    reason: str


def target_value(rating: str, equity: float, view: MarketView, config: AutoTraderConfig) -> float | None:
    """Dollar value the rating aims to hold, or None to leave the position as is (Hold)."""
    weight = config.target_weights.get(rating)
    if weight is None:
        return None
    vol_scale = min(1.0, config.target_annual_vol / view.annual_vol) if view.annual_vol > 0 else 1.0
    return equity * config.max_position_pct * weight * vol_scale


def plan_rebalance(ticker: str, rating: str, held_quantity: float, equity: float, cash: float,
                   gross_exposure: float, view: MarketView, config: AutoTraderConfig,
                   buys_allowed: bool) -> OrderPlan | None:
    """The order that moves the holding toward the rating's target, or None."""
    target = target_value(rating, equity, view, config)
    if target is None:
        return None
    current = held_quantity * view.price
    delta = target - current
    if abs(delta) < max(config.min_trade_value, config.rebalance_band_pct * max(target, current)):
        return None

    if delta > 0:
        if not buys_allowed:
            return None
        room = config.max_gross_exposure_pct * equity - gross_exposure
        # Keep a sliver of cash back for the fill's slippage and commission.
        delta = min(delta, room, cash * 0.99)
        if delta < config.min_trade_value:
            return None
        quantity = _round_qty(delta / view.price, config)
        side = "buy"
    else:
        quantity = held_quantity if target == 0 else min(held_quantity, _round_qty(-delta / view.price, config))
        side = "sell"
    if quantity <= 0:
        return None
    return OrderPlan(ticker, side, quantity, f"rating {rating}: target {target:,.0f}, held {current:,.0f}")


def exit_reason(view: MarketView, entry_price: float, high_since_entry: float, atr_at_entry: float,
                config: AutoTraderConfig) -> str | None:
    """Why a held position must close now, from its stop or profit target, or None."""
    atr = atr_at_entry or view.atr
    stop = high_since_entry - config.stop_loss_atr * atr
    if view.price <= stop:
        return f"trailing stop: {view.price:,.2f} <= {stop:,.2f}"
    if config.take_profit_atr is not None:
        target = entry_price + config.take_profit_atr * atr
        if view.price >= target:
            return f"take profit: {view.price:,.2f} >= {target:,.2f}"
    return None


def _round_qty(quantity: float, config: AutoTraderConfig) -> float:
    return math.floor(quantity * 10_000) / 10_000 if config.allow_fractional else float(math.floor(quantity))
