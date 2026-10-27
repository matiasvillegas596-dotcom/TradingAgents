"""An autonomous trader built on the agents' ratings.

The analysis graph decides direction (Buy ... Sell); this package turns those
ratings into sized orders, guards them with stops and circuit breakers, and runs
the loop on a schedule against a paper ledger or an Alpaca account.
"""

from __future__ import annotations

from tradingagents.autotrader.broker import AlpacaBroker, PaperBroker
from tradingagents.autotrader.config import AutoTraderConfig, load_autotrader_config
from tradingagents.autotrader.engine import AutoTrader


def build_autotrader(config: AutoTraderConfig, graph_config: dict | None = None) -> AutoTrader:
    """Wire the trader to real prices, a broker and the agents graph."""
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    def bars_of(ticker: str, as_of: str):
        return load_ohlcv(ticker, as_of)

    if config.broker == "alpaca":
        broker = AlpacaBroker(live=config.live)
    else:
        if config.live:
            raise ValueError("live trading needs the alpaca broker; the paper broker never sends orders")
        from datetime import date

        broker = PaperBroker(
            config.state_path / "paper_ledger.json",
            price_of=lambda t: float(bars_of(t, date.today().isoformat())["Close"].dropna().iloc[-1]),
            starting_cash=config.starting_cash,
            commission_bps=config.commission_bps, slippage_bps=config.slippage_bps,
        )

    analysts = config.selected_analysts or ["market", "social", "news", "fundamentals"]
    if config.asset_type == "crypto":
        analysts = [a for a in analysts if a != "fundamentals"]
    graph = TradingAgentsGraph(selected_analysts=analysts, config=dict(graph_config or DEFAULT_CONFIG))
    return AutoTrader(config, broker, propagate=graph.propagate, bars_of=bars_of)


__all__ = ["AutoTrader", "AutoTraderConfig", "AlpacaBroker", "PaperBroker",
           "build_autotrader", "load_autotrader_config"]
