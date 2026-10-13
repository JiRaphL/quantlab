"""From ideal weights to whole shares.

With a few thousand euros, a 300 $ ETF and a 1 $ minimum commission, the fractional
backtest is a fiction. ``discretize_weights`` turns target weights into integer share
counts (rounding down), reports the resulting *actual* weights and the cash left over,
and prices the orders with the per-order commission schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel


@dataclass
class LotAllocation:
    shares: pd.Series  # integer shares per symbol after the trade
    weights: pd.Series  # actual weights given whole shares
    cash: float  # cash left after buying the shares and paying commissions
    commissions: float  # total commissions of the orders
    orders: pd.Series  # share delta per symbol (signed)


def discretize_weights(
    target: pd.Series,
    prices_raw: pd.Series,
    capital: float,
    cost_model: CostModel,
    current_shares: pd.Series | None = None,
) -> LotAllocation:
    """``capital`` is the total portfolio value (cash + holdings) before the trade."""
    prices = prices_raw.reindex(target.index)
    current = (
        current_shares.reindex(target.index).fillna(0.0)
        if current_shares is not None
        else pd.Series(0.0, index=target.index)
    )
    tradable = prices.notna() & (prices > 0)
    budget = target.clip(lower=0.0) * capital
    shares = np.floor(budget / prices.where(tradable)).fillna(0.0)
    shares = shares.where(tradable, current)  # cannot trade what has no price: keep it
    orders = shares - current
    commissions = float(sum(cost_model.order_commission(int(o)) for o in orders))
    slippage = float((orders.abs() * prices.fillna(0.0)).sum() * cost_model.slippage_rate())
    invested = float((shares * prices.fillna(0.0)).sum())
    cash = capital - invested - commissions - slippage
    weights = (shares * prices.fillna(0.0)) / capital
    return LotAllocation(
        shares=shares.astype(int),
        weights=weights,
        cash=cash,
        commissions=commissions + slippage,
        orders=orders.astype(int),
    )
