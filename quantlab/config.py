"""Typed, hashable backtest configuration. The hash is the identity of a run: same hash,
same data => same result."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from quantlab.backtest.costs import CostModel
from quantlab.backtest.schedule import Rebalance


class CostConfig(BaseModel):
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 5.0

    def to_model(self) -> CostModel:
        return CostModel(**self.model_dump())


class BacktestConfig(BaseModel):
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    universe: str
    start: date
    end: date
    rebalance: Rebalance = "monthly"
    cost: CostConfig = Field(default_factory=CostConfig)
    initial_capital: float = 100_000.0
    execution_lag: int = 1
    benchmark: str = "SPY"
    provider: str = "yfinance"

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]

    def cost_model(self) -> CostModel:
        return self.cost.to_model()

    @classmethod
    def from_yaml(cls, path: Path | str) -> BacktestConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
