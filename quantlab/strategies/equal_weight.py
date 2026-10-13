"""The control strategy: 1/n over everything that trades. If a 'clever' strategy cannot
beat this after costs, it is not clever."""

from __future__ import annotations

import pandas as pd

from quantlab.backtest.benchmarks import equal_weight_weights
from quantlab.data.panel import PricePanel
from quantlab.strategies.base import Strategy


class EqualWeight(Strategy):
    name = "equal_weight"

    def target_weights(self, panel: PricePanel) -> pd.DataFrame:
        return equal_weight_weights(panel)
