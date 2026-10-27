"""Where orders go: a local paper ledger, or Alpaca (paper or live).

The engine sees only :class:`Broker`, so the same decisions run against either.
The paper broker fills market orders at the latest close plus slippage and
commission, and keeps its ledger in a JSON file so a restart resumes the book.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import requests

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
LIVE_GUARD_ENV = "TRADINGAGENTS_ALLOW_LIVE"


@dataclass
class Holding:
    ticker: str
    quantity: float
    average_price: float


@dataclass
class Account:
    cash: float
    equity: float
    holdings: dict[str, Holding]


@dataclass
class Fill:
    ticker: str
    side: str
    quantity: float
    price: float | None
    order_id: str
    submitted_at: str


class Broker(Protocol):
    def account(self) -> Account: ...

    def submit_market_order(self, ticker: str, side: str, quantity: float) -> Fill: ...

    def is_market_open(self) -> bool: ...


class PaperBroker:
    """A simulated account: fills at the latest price, charged slippage and commission."""

    def __init__(self, ledger_path: Path, price_of: Callable[[str], float], starting_cash: float,
                 commission_bps: float = 1.0, slippage_bps: float = 5.0):
        self.ledger_path = ledger_path
        self.price_of = price_of
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        if ledger_path.exists():
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.cash = float(data["cash"])
            self.holdings = {t: Holding(**h) for t, h in data["holdings"].items()}
            self.next_id = int(data.get("next_id", 1))
        else:
            self.cash, self.holdings, self.next_id = float(starting_cash), {}, 1

    def account(self) -> Account:
        equity = self.cash + sum(h.quantity * self.price_of(t) for t, h in self.holdings.items())
        return Account(cash=self.cash, equity=equity, holdings=dict(self.holdings))

    def submit_market_order(self, ticker: str, side: str, quantity: float) -> Fill:
        if quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {quantity}")
        mid = self.price_of(ticker)
        held = self.holdings.get(ticker)
        if side == "buy":
            price = mid * (1 + self.slippage)
            cost = quantity * price * (1 + self.commission)
            if cost > self.cash + 1e-9:
                raise ValueError(f"insufficient cash for {quantity} {ticker}: need {cost:,.2f}, have {self.cash:,.2f}")
            self.cash -= cost
            if held:
                total = held.quantity + quantity
                held.average_price = (held.quantity * held.average_price + quantity * price) / total
                held.quantity = total
            else:
                self.holdings[ticker] = Holding(ticker, quantity, price)
        elif side == "sell":
            if not held or quantity > held.quantity + 1e-9:
                raise ValueError(f"cannot sell {quantity} {ticker}: holding {held.quantity if held else 0}")
            price = mid * (1 - self.slippage)
            self.cash += quantity * price * (1 - self.commission)
            held.quantity -= quantity
            if held.quantity <= 1e-9:
                del self.holdings[ticker]
        else:
            raise ValueError(f"side must be buy or sell, got {side!r}")
        fill = Fill(ticker, side, quantity, price, f"paper-{self.next_id}", _now())
        self.next_id += 1
        self._save()
        return fill

    def is_market_open(self) -> bool:
        return True

    def _save(self) -> None:
        write_json_atomic(self.ledger_path, {
            "cash": self.cash, "next_id": self.next_id,
            "holdings": {t: asdict(h) for t, h in self.holdings.items()},
        })


class AlpacaBroker:
    """Alpaca's trading API. Paper unless ``live`` is set and the env guard allows it."""

    def __init__(self, live: bool = False, session: requests.Session | None = None):
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError("set ALPACA_API_KEY and ALPACA_SECRET_KEY to trade through Alpaca")
        if live and os.environ.get(LIVE_GUARD_ENV) != "1":
            raise ValueError(f"live trading also needs {LIVE_GUARD_ENV}=1 in the environment")
        self.base_url = ALPACA_LIVE_URL if live else ALPACA_PAPER_URL
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        response = self.session.request(method, self.base_url + path, timeout=30, **kwargs)
        if not response.ok:
            raise RuntimeError(f"Alpaca {method} {path} failed ({response.status_code}): {response.text[:300]}")
        return response.json()

    def account(self) -> Account:
        acct = self._request("GET", "/v2/account")
        holdings = {
            p["symbol"]: Holding(p["symbol"], float(p["qty"]), float(p["avg_entry_price"]))
            for p in self._request("GET", "/v2/positions")
        }
        return Account(cash=float(acct["cash"]), equity=float(acct["equity"]), holdings=holdings)

    def submit_market_order(self, ticker: str, side: str, quantity: float) -> Fill:
        order = self._request("POST", "/v2/orders", json={
            "symbol": ticker, "qty": str(quantity), "side": side,
            "type": "market", "time_in_force": "day",
        })
        price = order.get("filled_avg_price")
        return Fill(ticker, side, quantity, float(price) if price else None, order["id"], _now())

    def is_market_open(self) -> bool:
        return bool(self._request("GET", "/v2/clock")["is_open"])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json_atomic(path: Path, data) -> None:
    """Write through a temp file and rename, so a crash never leaves half a ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
