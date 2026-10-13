"""Reference allocations every strategy is compared against."""

from __future__ import annotations

import pandas as pd

from quantlab.data.panel import PricePanel


def buy_and_hold_weights(panel: PricePanel, symbol: str) -> pd.DataFrame:
    """100 % in ``symbol`` whenever it has a price, else cash."""
    if symbol not in panel.symbols:
        raise KeyError(symbol)
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    w[symbol] = panel.close[symbol].notna().astype(float)
    return w


def equal_weight_weights(panel: PricePanel) -> pd.DataFrame:
    """1/n over the symbols that have a price at each date."""
    available = panel.close.notna().astype(float)
    n = available.sum(axis=1).replace(0.0, float("nan"))
    return available.div(n, axis=0).fillna(0.0)
