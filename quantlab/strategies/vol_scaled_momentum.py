"""Volatility-scaled dual momentum (H2).

Three changes to plain dual momentum, each with a documented rationale:

1. **Risk-adjusted ranking.** Assets are ranked on ``momentum / realized volatility``
   rather than raw momentum. Raw momentum mechanically favours the most volatile assets
   (equities, commodities), so a raw ranking is partly a bet on volatility rather than on
   trend persistence. Scaling by volatility is the standard fix (see "volatility-scaled
   momentum", Barroso & Santa-Clara 2015).
2. **Inverse-volatility weights among picks.** Instead of 1/top_n each, picks are weighted
   by 1/vol so each contributes a similar amount of risk (risk parity within the sleeve).
3. **Portfolio volatility target.** The risky sleeve is scaled down when its estimated
   volatility exceeds ``target_vol`` (annualised); the remainder goes to the safe asset or
   cash. Volatility-managed portfolios raise Sharpe and cut drawdowns because volatility
   is persistent while returns are not (Moreira & Muir 2017). The estimate is the
   weighted sum of asset volatilities, i.e. it assumes perfect correlation: a deliberately
   conservative upper bound. No leverage: the scale is capped at 1.

The absolute trend filter still uses *raw* momentum (sign of the trend), as in H1.
``target_vol=1.0`` disables the target in practice (no sleeve has 100 % volatility).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.panel import PricePanel
from quantlab.features.momentum import momentum
from quantlab.features.volatility import realized_volatility
from quantlab.strategies.base import Strategy


class VolScaledMomentum(Strategy):
    name = "vol_scaled_momentum"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "lookback": 252,
            "skip": 21,
            "vol_window": 63,
            "top_n": 3,
            "absolute_filter": True,
            "safe_asset": None,
            "target_vol": 0.10,
        }

    @classmethod
    def param_grid_default(cls) -> dict[str, list]:
        return {"lookback": [126, 189, 252], "top_n": [2, 3, 4], "target_vol": [0.08, 0.12, 1.0]}

    def target_weights(self, panel: PricePanel) -> pd.DataFrame:
        p = self.params
        lookback, skip, top_n = int(p["lookback"]), int(p["skip"]), int(p["top_n"])
        vol_window, target_vol, safe = int(p["vol_window"]), float(p["target_vol"]), p["safe_asset"]
        if safe is not None and safe not in panel.symbols:
            raise KeyError(f"safe_asset {safe!r} not in panel")

        mom = momentum(panel.close, lookback=lookback, skip=skip)
        vol = realized_volatility(panel.returns(), window=vol_window, annualize=True)
        vol = vol.where(vol > 0)  # a zero-vol asset cannot be ranked or sized
        score = mom / vol

        risky_cols = [s for s in panel.symbols if s != safe]
        risky_score = score[risky_cols]
        ranks = risky_score.rank(axis=1, ascending=False, method="first")
        picked = ranks <= top_n
        if p["absolute_filter"]:
            threshold = mom[safe] if safe is not None else 0.0
            picked &= mom[risky_cols].gt(threshold, axis=0)
        picked = picked.fillna(False)

        inv_vol = (1.0 / vol[risky_cols]).where(picked, 0.0).fillna(0.0)
        inv_sum = inv_vol.sum(axis=1)
        sleeve_w = inv_vol.div(inv_sum.where(inv_sum > 0, np.nan), axis=0).fillna(0.0)

        # Sleeve size: n_picked / top_n, as in H1 (unfilled slots are defensive).
        n_picked = picked.sum(axis=1)
        sleeve_size = (n_picked.clip(upper=top_n) / top_n).astype(float)
        # Volatility target on the sleeve (perfect-correlation upper bound).
        sleeve_vol = (sleeve_w * vol[risky_cols].fillna(0.0)).sum(axis=1)
        scale = (target_vol / sleeve_vol.where(sleeve_vol > 0, np.nan)).clip(upper=1.0).fillna(1.0)
        risky_w = sleeve_w.mul(sleeve_size * scale, axis=0)

        weights = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
        weights[risky_cols] = risky_w
        if safe is not None:
            remainder = 1.0 - weights.sum(axis=1)
            weights[safe] = remainder.where(panel.close[safe].notna(), 0.0)
        return weights.clip(lower=0.0)
