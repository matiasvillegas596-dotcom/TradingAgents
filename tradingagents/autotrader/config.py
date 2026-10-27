"""Settings for the autonomous trader, loaded from one JSON file.

Every limit has a conservative default so a config naming only a watchlist is
already safe to run on paper. Live trading is never a default: it needs both
``"live": true`` here and ``TRADINGAGENTS_ALLOW_LIVE=1`` in the environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tradingagents.agents.rating import RATINGS_5_TIER


class AutoTraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    watchlist: list[str] = Field(min_length=1, description="Tickers the agents analyze each cycle")
    asset_type: Literal["stock", "crypto"] = "stock"
    selected_analysts: list[str] | None = Field(
        default=None, description="Analysts to run; None runs every one the asset type allows")

    broker: Literal["paper", "alpaca"] = "paper"
    live: bool = Field(default=False, description="Send real orders (alpaca only); also needs the env guard")
    starting_cash: float = Field(default=100_000.0, gt=0, description="Paper broker's opening cash")
    commission_bps: float = Field(default=1.0, ge=0, description="Paper broker's cost per fill, in bps")
    slippage_bps: float = Field(default=5.0, ge=0, description="Paper broker's adverse fill offset, in bps")
    allow_fractional: bool = False

    # Sizing: what share of equity each rating aims to hold in one name.
    target_weights: dict[str, float] = Field(default_factory=lambda: {
        "Buy": 1.0, "Overweight": 0.6, "Underweight": 0.0, "Sell": 0.0,
    })
    max_position_pct: float = Field(default=0.15, gt=0, le=1, description="Cap on one name, share of equity")
    max_gross_exposure_pct: float = Field(default=0.9, gt=0, le=1, description="Cap on all longs, share of equity")
    target_annual_vol: float = Field(default=0.25, gt=0, description="Positions in more volatile names shrink to this")
    rebalance_band_pct: float = Field(default=0.2, ge=0, description="Skip trades smaller than this share of the target")
    min_trade_value: float = Field(default=100.0, ge=0)

    # Exits that run between analyses, without the LLM.
    stop_loss_atr: float = Field(default=2.5, gt=0, description="Trailing stop, in ATRs below the high since entry")
    take_profit_atr: float | None = Field(default=None, gt=0, description="Take profit, in ATRs above entry")

    # Circuit breakers.
    max_daily_loss_pct: float = Field(default=0.03, gt=0, le=1, description="No new buys for the day past this loss")
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1, description="Halt all buys past this drawdown from peak")

    # Schedule (market time zone).
    timezone: str = "America/New_York"
    analysis_time: str = Field(default="15:30", description="HH:MM the agents run on trading days")
    stop_check_minutes: int = Field(default=15, ge=1)
    state_dir: str = "~/.tradingagents/autotrader"

    @field_validator("watchlist")
    @classmethod
    def _tickers(cls, value: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in value if t.strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("watchlist names a ticker twice")
        return cleaned

    @field_validator("target_weights")
    @classmethod
    def _weights(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - set(RATINGS_5_TIER)
        if unknown:
            raise ValueError(f"target_weights has unknown ratings: {sorted(unknown)}")
        if any(not 0 <= w <= 1 for w in value.values()):
            raise ValueError("target_weights must be between 0 and 1")
        return value

    @field_validator("analysis_time")
    @classmethod
    def _hhmm(cls, value: str) -> str:
        hours, _, minutes = value.partition(":")
        if not (hours.isdigit() and minutes.isdigit() and int(hours) < 24 and int(minutes) < 60):
            raise ValueError(f"analysis_time must be HH:MM, got {value!r}")
        return f"{int(hours):02d}:{int(minutes):02d}"

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir).expanduser()


def load_autotrader_config(path: str | Path) -> AutoTraderConfig:
    """Read the config file, failing before any order rather than mid-cycle."""
    try:
        return AutoTraderConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"autotrader config {path} is not usable: {exc}") from exc
