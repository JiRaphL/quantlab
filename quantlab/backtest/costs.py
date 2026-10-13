"""Transaction cost model, approximating Interactive Brokers' US stock pricing.

Two usages:

* fractional backtests (the default engine) charge ``turnover_cost_rate`` per unit of
  traded notional: slippage in basis points plus the per-share commission expressed as a
  fraction of the price. Minimum commissions cannot be modelled without share counts.
* the real-capital simulation charges ``order_commission`` per order with the minimum,
  plus slippage on notional.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 5.0

    def turnover_cost_rate(self, prices: pd.Series) -> pd.Series:
        """Cost per unit of traded notional, per symbol (``prices`` are raw closes)."""
        commission = (self.commission_per_share / prices).fillna(0.0)
        return commission + self.slippage_bps / 1e4

    def order_commission(self, shares: int | float) -> float:
        if shares == 0:
            return 0.0
        return max(self.min_commission, abs(shares) * self.commission_per_share)

    def slippage_rate(self) -> float:
        return self.slippage_bps / 1e4


ZERO_COST = CostModel(commission_per_share=0.0, min_commission=0.0, slippage_bps=0.0)

IBKR_DEFAULT = CostModel()
