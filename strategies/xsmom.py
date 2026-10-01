"""Cross-sectional momentum portfolio -- long-only India adaptation
(Jegadeesh & Titman, 1993, JF; 52-week-high refinement: George & Hwang, 2004).

Each rebalance: rank the universe by trailing `formation`-month return
(skipping the most recent month -- the standard reversal-avoidance skip),
hold the top decile (min 3 names) equal-weighted for `hold` months.

Execution honesty: ranks are computed on data through month m-1 and the
portfolio earns month m's return -- i.e. executable at month m's open, no
lookahead. Turnover is charged explicitly via the Indian delivery cost model
(STT 0.1% both legs, stamp duty, exchange, GST, SEBI + slippage), because
turnover is where momentum usually dies.

India evidence: Sehgal & Balakrishnan (2002) found momentum on Indian
stocks; the canonical 6/6 is not settled for India, so (formation, hold)
is chosen by walk-forward, never hard-coded.
"""
import pandas as pd

from engine.backtest import _metrics
from engine.costs import CostModel

NAME = "xsmom"
DESCRIPTION = ("Cross-sectional momentum: hold top-decile names by trailing "
               "formation-month return (skip 1m), equal weight.")
PARAM_GRID = {"formation": [3, 6, 9, 12], "hold": [1, 3]}
WARMUP_MONTHS = 16  # >= max formation (12) + skip (1) + margin
MIN_NAMES = 3


def _monthly(closes_daily: pd.DataFrame) -> pd.DataFrame:
    m = closes_daily.resample("ME").last()
    return m.dropna(axis=1, how="all")


def _rank(formation_rets: pd.Series):
    """Top-decile picks from a formation-return series (NaNs already dropped
    point-in-time: names lacking formation history are simply not ranked)."""
    valid = formation_rets.sort_values(ascending=False)
    if len(valid) < MIN_NAMES:
        return []
    k = max(MIN_NAMES, int(len(valid) * 0.1))
    return list(valid.index[:k])


def run(closes_daily: pd.DataFrame, params: dict, capital: float = 100_000.0,
        costs: CostModel = None, start_pos: int = None) -> dict:
    """Backtest from monthly position `start_pos` (equity starts at capital
    at month start_pos-1). All indicator inputs before start_pos are pure
    past data -- warmup, not leakage."""
    costs = costs or CostModel()
    formation, hold = params["formation"], params["hold"]
    monthly = _monthly(closes_daily)
    rets = monthly.pct_change()
    n = len(monthly)
    if n < 4:
        raise ValueError("Not enough monthly history for xsmom.")
    if start_pos is None:
        start_pos = WARMUP_MONTHS
    if start_pos < formation + 2:
        raise ValueError(f"start_pos {start_pos} < formation+2 "
                         f"({formation + 2}): indicators would be cold.")

    equity, eq_idx = [capital], [monthly.index[start_pos - 1]]
    picks_prev, n_rebal, total_cost = [], 0, 0.0
    m = start_pos
    while m < n:
        # rank on data through month m-1 only
        form_ret = (monthly.iloc[m - 1] / monthly.iloc[m - 1 - formation]
                    - 1).dropna()
        picks = _rank(form_ret)
        for h in range(hold):
            if m + h >= n:
                break
            mr = rets.iloc[m + h][picks].mean() if picks else 0.0
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
            eq_idx.append(monthly.index[m + h])
        picks_prev, n_rebal = picks, n_rebal + 1
        m += hold

    eq = pd.Series(equity, index=eq_idx, name="equity")
    trips = pd.DataFrame(columns=["entry", "exit", "pnl", "cost"])
    metrics = _metrics(eq, trips, bars_per_year=12)
    metrics["total_costs"] = round(total_cost, 2)
    metrics["num_rebalances"] = n_rebal
    return {"equity": eq, "metrics": metrics, "params": dict(params)}


def _last_complete_month(daily: pd.DataFrame, monthly: pd.DataFrame):
    """Month-end through which data is complete. A partial current month is
    never used for ranking."""
    last_day = daily.index[-1].normalize()
    cutoff = last_day if last_day.is_month_end else last_day - pd.offsets.MonthEnd(1)
    past = monthly.index[monthly.index <= cutoff]
    if len(past) == 0:
        raise ValueError("No complete month of history available.")
    return past[-1]


def current_picks(closes_daily: pd.DataFrame, params: dict) -> dict:
    """Portfolio for the upcoming month, ranked on the last complete
    month-end. Returns picks + formation stats for the signal plan."""
    formation = params["formation"]
    monthly = _monthly(closes_daily)
    as_of = _last_complete_month(closes_daily, monthly)
    m = monthly.index.get_loc(as_of) + 1  # rebalance month position
    form_ret = (monthly.iloc[m - 1] / monthly.iloc[m - 1 - formation]
                - 1).dropna()
    picks = _rank(form_ret)
    next_rebal = monthly.index[m] if m < len(monthly) else None
    return {
        "strategy": NAME,
        "params": dict(params),
        "as_of_month": as_of.strftime("%Y-%m"),
        "next_rebalance_month": (next_rebal.strftime("%Y-%m")
                                 if next_rebal is not None else None),
        "universe_ranked": int(len(form_ret)),
        "picks": [
            {"symbol": s, "weight": round(1 / max(len(picks), 1), 4),
             "formation_return": round(float(form_ret[s]), 4)}
            for s in picks
        ],
    }
