"""Dual momentum on a basket of ETFs (Antonacci, 2014).

Economic rationale: trailing 6-12 month returns persist for a few months (cross-sectional
momentum, Jegadeesh & Titman 1993; time-series momentum, Moskowitz, Ooi & Pedersen 2012),
usually attributed to under-reaction and slow capital flows. The *absolute* filter avoids
holding assets in a downtrend, which is where the worst drawdowns of relative momentum
come from.

Rules at each date t (the engine only reads them on rebalance dates):
1. score = momentum(close, lookback, skip) for every symbol with enough history;
2. rank descending, keep the ``top_n`` best;
3. if ``absolute_filter``: drop picks whose score is not above ``threshold``
   (0, or the safe asset's own score if ``safe_asset`` is set);
4. equal weight among surviving picks; the remainder goes to ``safe_asset`` if given,
   else cash.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.panel import PricePanel
from quantlab.features.momentum import momentum
from quantlab.strategies.base import Strategy


class DualMomentum(Strategy):
    name = "dual_momentum"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "lookback": 252,
            "skip": 21,
            "top_n": 3,
            "absolute_filter": True,
            "safe_asset": None,
        }

    @classmethod
    def param_grid_default(cls) -> dict[str, list]:
        return {"lookback": [126, 189, 252], "skip": [0, 21], "top_n": [1, 2, 3]}

    def target_weights(self, panel: PricePanel) -> pd.DataFrame:
        p = self.params
        lookback, skip, top_n = int(p["lookback"]), int(p["skip"]), int(p["top_n"])
        safe = p["safe_asset"]
        if safe is not None and safe not in panel.symbols:
            raise KeyError(f"safe_asset {safe!r} not in panel")

        score = momentum(panel.close, lookback=lookback, skip=skip)
        risky = score.drop(columns=[safe]) if safe is not None else score

        # Rank among non-NaN scores, 1 = best.
        ranks = risky.rank(axis=1, ascending=False, method="first")
        picked = ranks <= top_n

        if p["absolute_filter"]:
            threshold = score[safe] if safe is not None else 0.0
            picked &= risky.gt(threshold, axis=0)

        picked = picked.fillna(False)
        n_picked = picked.sum(axis=1)
        risky_w = picked.astype(float).div(n_picked.where(n_picked > 0, np.nan), axis=0)
        risky_w = risky_w.fillna(0.0) * (n_picked.clip(upper=top_n) / top_n).to_numpy()[:, None]
        # Each pick gets 1/top_n; unfilled slots go to the safe asset (or cash).

        weights = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        weights[risky_w.columns] = risky_w
        if safe is not None:
            remainder = 1.0 - weights.sum(axis=1)
            safe_available = panel.close[safe].notna()
            weights[safe] = remainder.where(safe_available, 0.0)
        return weights
