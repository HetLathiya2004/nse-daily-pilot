"""Time-series momentum portfolio -- long-only India adaptation
(Moskowitz, Ooi & Pedersen, 2012, JFE).

Each month: hold (equal-weighted) every universe name whose own trailing
`lookback`-month return, skipping the most recent month, is positive; hold
cash otherwise. Rebalanced monthly.

Same execution honesty as xsmom: the signal uses data through month m-1 and
earns month m's return; turnover pays the full Indian delivery cost model.
The lookback is chosen by walk-forward, never hard-coded.
"""
import pandas as pd

from engine.backtest import _metrics
from engine.costs import CostModel
from .xsmom import _last_complete_month, _monthly

NAME = "ts_mom"
DESCRIPTION = ("Time-series momentum: hold names with positive trailing "
               "lookback-month return (skip 1m), equal weight, else cash.")
PARAM_GRID = {"lookback": [3, 6, 9, 12]}
WARMUP_MONTHS = 16  # >= max lookback (12) + skip (1) + margin


def run(closes_daily: pd.DataFrame, params: dict, capital: float = 100_000.0,
        costs: CostModel = None, start_pos: int = None) -> dict:
    costs = costs or CostModel()
    lookback = params["lookback"]
    monthly = _monthly(closes_daily)
    rets = monthly.pct_change()
    n = len(monthly)
    if n < 4:
        raise ValueError("Not enough monthly history for ts_mom.")
    if start_pos is None:
        start_pos = WARMUP_MONTHS
    if start_pos < lookback + 2:
        raise ValueError(f"start_pos {start_pos} < lookback+2 "
                         f"({lookback + 2}): indicators would be cold.")

    equity, eq_idx = [capital], [monthly.index[start_pos - 1]]
    picks_prev, n_rebal, total_cost = [], 0, 0.0
    m = start_pos
    while m < n:
        # signal on data through month m-1 only
        trail = (monthly.iloc[m - 1] / monthly.iloc[m - 1 - lookback]
                 - 1).dropna()
        picks = list(trail[trail > 0].index)
        mr = rets.iloc[m][picks].mean() if picks else 0.0
        mr = 0.0 if pd.isna(mr) else mr
        val = equity[-1]
        replaced = (len(set(picks) - set(picks_prev))
                    / max(len(picks), 1)) if picks else 0.0
        traded_val = replaced * val
        fee = (costs.leg(traded_val / 2, "sell", False)
               + costs.leg(traded_val / 2, "buy", False)
               ) if traded_val > 0 else 0.0
        total_cost += fee
        equity.append(val * (1 + mr) - fee)
        eq_idx.append(monthly.index[m])
        picks_prev, n_rebal = picks, n_rebal + 1
        m += 1

    eq = pd.Series(equity, index=eq_idx, name="equity")
    trips = pd.DataFrame(columns=["entry", "exit", "pnl", "cost"])
    metrics = _metrics(eq, trips, bars_per_year=12)
    metrics["total_costs"] = round(total_cost, 2)
    metrics["num_rebalances"] = n_rebal
    return {"equity": eq, "metrics": metrics, "params": dict(params)}


def current_picks(closes_daily: pd.DataFrame, params: dict) -> dict:
    """Names the strategy would hold for the upcoming month, ranked on the
    last complete month-end."""
    lookback = params["lookback"]
    monthly = _monthly(closes_daily)
    as_of = _last_complete_month(closes_daily, monthly)
    m = monthly.index.get_loc(as_of) + 1
    trail = (monthly.iloc[m - 1] / monthly.iloc[m - 1 - lookback]
             - 1).dropna()
    picks = list(trail[trail > 0].index)
    next_rebal = monthly.index[m] if m < len(monthly) else None
    return {
        "strategy": NAME,
        "params": dict(params),
        "as_of_month": as_of.strftime("%Y-%m"),
        "next_rebalance_month": (next_rebal.strftime("%Y-%m")
                                 if next_rebal is not None else None),
        "universe_ranked": int(len(trail)),
        "picks": [
            {"symbol": s, "weight": round(1 / max(len(picks), 1), 4),
             "formation_return": round(float(trail[s]), 4)}
            for s in picks
        ],
    }
