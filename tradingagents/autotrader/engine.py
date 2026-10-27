"""The autonomous loop: analyze the watchlist, trade the ratings, guard the book.

Each trading day at ``analysis_time`` the agents analyze every watchlist name
with the live book as context, and the ratings become orders through the risk
rules. Between analyses, stops and profit targets are checked on a short
interval with no LLM involved. Circuit breakers stop new buys after a bad day
and halt them entirely after a deep drawdown, until a person resets the halt.

Every order, exit and decision is appended to ``journal.jsonl`` in the state
directory, so what the trader did, and why, can be audited afterwards.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.agents.rating import is_review
from tradingagents.autotrader.broker import Account, Broker, write_json_atomic
from tradingagents.autotrader.config import AutoTraderConfig
from tradingagents.autotrader.risk import MarketView, OrderPlan, exit_reason, plan_rebalance
from tradingagents.portfolio import PortfolioContext, Position

logger = logging.getLogger(__name__)

SESSION_OPEN, SESSION_CLOSE = (9, 30), (16, 0)


class AutoTrader:
    def __init__(self, config: AutoTraderConfig, broker: Broker,
                 propagate: Callable[..., tuple[dict, str]],
                 bars_of: Callable[[str, str], pd.DataFrame],
                 now: Callable[[], datetime] | None = None):
        self.config = config
        self.broker = broker
        self.propagate = propagate
        self.bars_of = bars_of
        self.tz = ZoneInfo(config.timezone)
        self.now = now or (lambda: datetime.now(self.tz))
        self.state_file = config.state_path / "state.json"
        self.journal_file = config.state_path / "journal.jsonl"
        self.state = self._load_state()
        self._views: dict[str, MarketView] = {}

    # -- public cycle steps -------------------------------------------------

    def check_exits(self) -> list[OrderPlan]:
        """Close every position whose trailing stop or profit target was hit."""
        self._views.clear()
        account = self._sync()
        exits = []
        for ticker, holding in account.holdings.items():
            if holding.quantity <= 0:
                continue
            view = self._view(ticker)
            if view is None:
                continue
            tracked = self.state["positions"][ticker]
            tracked["high"] = max(tracked["high"], view.price)
            reason = exit_reason(view, tracked["entry_price"], tracked["high"], tracked["atr_at_entry"], self.config)
            if reason:
                exits.append(OrderPlan(ticker, "sell", holding.quantity, reason))
        self._execute(exits, kind="exit")
        self._save_state()
        return exits

    def run_analysis(self) -> list[OrderPlan]:
        """Run the agents on the watchlist and trade toward their ratings."""
        self._views.clear()
        account = self._sync()
        trade_date = self.now().strftime("%Y-%m-%d")
        book = PortfolioContext(cash=account.cash, currency="USD", positions=[
            Position(ticker=h.ticker, quantity=h.quantity, average_price=h.average_price)
            for h in account.holdings.values()
        ])
        ratings = {}
        for ticker in self.config.watchlist:
            try:
                _, rating = self.propagate(ticker, trade_date, asset_type=self.config.asset_type, portfolio=book)
            except Exception as exc:  # one name's failure must not stop the rest
                logger.exception("analysis of %s failed", ticker)
                self._journal("analysis_failed", ticker=ticker, error=str(exc))
                continue
            ratings[ticker] = rating
            self._journal("decision", ticker=ticker, date=trade_date, rating=rating)

        buys_allowed = self._buys_allowed(account)
        plans = []
        gross = sum(h.quantity * v.price for t, h in account.holdings.items() if (v := self._view(t)))
        for ticker, rating in ratings.items():
            if is_review(rating):
                continue
            view = self._view(ticker)
            if view is None:
                continue
            held = account.holdings[ticker].quantity if ticker in account.holdings else 0.0
            plan = plan_rebalance(ticker, rating, held, account.equity, account.cash, gross, view,
                                  self.config, buys_allowed)
            if plan:
                plans.append(plan)
                if plan.side == "buy":
                    gross += plan.quantity * view.price
                    account.cash -= plan.quantity * view.price
        # Sells first, so their proceeds fund the buys.
        plans.sort(key=lambda p: p.side != "sell")
        self._execute(plans, kind="rebalance")
        self.state["last_analysis_date"] = trade_date
        self._save_state()
        return plans

    def run_forever(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """Trade on the schedule until interrupted."""
        logger.info("autotrader started: %s via %s%s", ", ".join(self.config.watchlist),
                    self.config.broker, " (LIVE)" if self.config.live else "")
        while True:
            now = self.now()
            try:
                if self._session_open(now):
                    self.check_exits()
                    if self._analysis_due(now):
                        self.run_analysis()
            except Exception as exc:  # keep trading tomorrow after a bad cycle
                logger.exception("cycle failed")
                self._journal("cycle_failed", error=str(exc))
            sleep(self.config.stop_check_minutes * 60)

    # -- schedule ------------------------------------------------------------

    def _session_open(self, now: datetime) -> bool:
        if self.config.asset_type == "crypto":
            return True
        if now.weekday() >= 5 or not SESSION_OPEN <= (now.hour, now.minute) < SESSION_CLOSE:
            return False
        return self.broker.is_market_open()

    def _analysis_due(self, now: datetime) -> bool:
        hour, minute = map(int, self.config.analysis_time.split(":"))
        return ((now.hour, now.minute) >= (hour, minute)
                and self.state.get("last_analysis_date") != now.strftime("%Y-%m-%d"))

    # -- risk state ------------------------------------------------------------

    def _sync(self) -> Account:
        """Read the account, track new holdings, and update the circuit breakers."""
        account = self.broker.account()
        positions = self.state["positions"]
        for ticker in list(positions):
            if ticker not in account.holdings:
                del positions[ticker]
        for ticker, holding in account.holdings.items():
            if ticker not in positions:
                view = self._view(ticker)
                positions[ticker] = {
                    "entry_price": holding.average_price,
                    "high": max(holding.average_price, view.price if view else 0.0),
                    "atr_at_entry": view.atr if view else 0.0,
                }
        today = self.now().strftime("%Y-%m-%d")
        if self.state.get("day") != today:
            self.state["day"], self.state["day_start_equity"] = today, account.equity
        self.state["peak_equity"] = max(self.state.get("peak_equity") or 0.0, account.equity)
        drawdown = 1 - account.equity / self.state["peak_equity"]
        if drawdown >= self.config.max_drawdown_pct and not self.state.get("halted"):
            self.state["halted"] = True
            self._journal("halted", equity=account.equity, peak=self.state["peak_equity"], drawdown=drawdown)
            logger.error("drawdown %.1f%% reached the limit: new buys halted until reset", drawdown * 100)
        self._journal("account", cash=account.cash, equity=account.equity,
                      holdings={t: h.quantity for t, h in account.holdings.items()})
        return account

    def _buys_allowed(self, account: Account) -> bool:
        if self.state.get("halted"):
            return False
        day_loss = 1 - account.equity / self.state["day_start_equity"]
        if day_loss >= self.config.max_daily_loss_pct:
            self._journal("buys_paused", reason=f"daily loss {day_loss:.1%}")
            return False
        return True

    def reset_halt(self) -> None:
        self.state["halted"] = False
        self.state["peak_equity"] = None
        self._save_state()
        self._journal("halt_reset")

    # -- plumbing ------------------------------------------------------------------

    def _execute(self, plans: list[OrderPlan], kind: str) -> None:
        for plan in plans:
            try:
                fill = self.broker.submit_market_order(plan.ticker, plan.side, plan.quantity)
            except Exception as exc:
                logger.exception("order for %s failed", plan.ticker)
                self._journal("order_failed", kind=kind, ticker=plan.ticker, side=plan.side,
                              quantity=plan.quantity, reason=plan.reason, error=str(exc))
                continue
            logger.info("%s %s %s %s @ %s (%s)", kind, plan.side, plan.quantity, plan.ticker,
                        fill.price, plan.reason)
            self._journal("order", kind=kind, ticker=plan.ticker, side=plan.side, quantity=plan.quantity,
                          price=fill.price, order_id=fill.order_id, reason=plan.reason)
            if plan.side == "buy" and (view := self._view(plan.ticker)):
                tracked = self.state["positions"].setdefault(
                    plan.ticker, {"entry_price": fill.price or view.price, "high": 0.0, "atr_at_entry": view.atr})
                tracked["high"] = max(tracked["high"], fill.price or view.price)
                tracked["atr_at_entry"] = view.atr

    def _view(self, ticker: str) -> MarketView | None:
        if ticker not in self._views:
            try:
                self._views[ticker] = MarketView.from_bars(self.bars_of(ticker, self.now().strftime("%Y-%m-%d")))
            except Exception as exc:
                logger.warning("no usable prices for %s: %s", ticker, exc)
                self._journal("no_prices", ticker=ticker, error=str(exc))
                return None
        return self._views[ticker]

    def _load_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        return {"positions": {}, "peak_equity": None, "halted": False}

    def _save_state(self) -> None:
        write_json_atomic(self.state_file, self.state)

    def _journal(self, event: str, **fields) -> None:
        self.journal_file.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": self.now().isoformat(timespec="seconds"), "event": event, **fields}) + "\n")
