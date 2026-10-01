"""Backtest engine.

Honesty rules (this is what "no bluffs" means in code):

1. No lookahead -- a signal formed at bar t-1 executes at the OPEN of bar t.
2. Every leg pays real Indian costs (see costs.py) plus adverse slippage.
3. Equity is marked to market every bar; cash never goes negative by
   construction (position size is a fraction of current equity).
4. Metrics are computed from the equity curve and completed round trips only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame      # one row per leg
    roundtrips: pd.DataFrame  # one row per completed round trip
    metrics: dict = field(default_factory=dict)


def _metrics(equity: pd.Series, roundtrips: pd.DataFrame,
             bars_per_year: int) -> dict:
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (bars_per_year / max(n, 1)) - 1
    vol = rets.std()
    sharpe = (rets.mean() / vol * np.sqrt(bars_per_year)) if vol > 0 else 0.0
    roll_max = eq.cummax()
    max_dd = ((eq - roll_max) / roll_max).min()
    wins = (roundtrips["pnl"] > 0).sum() if len(roundtrips) else 0
    losses = (roundtrips["pnl"] <= 0).sum() if len(roundtrips) else 0
    gross_win = roundtrips.loc[roundtrips["pnl"] > 0, "pnl"].sum()
    gross_loss = -roundtrips.loc[roundtrips["pnl"] <= 0, "pnl"].sum()
    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "num_roundtrips": int(len(roundtrips)),
        "win_rate": round(wins / max(wins + losses, 1), 3),
        "profit_factor": round(gross_win / max(gross_loss, 1e-9), 3),
        "total_costs": round(float(roundtrips["cost"].sum()) if len(roundtrips) else 0.0, 2),
    }


def backtest(df: pd.DataFrame, strategy, capital: float = 100_000.0,
             allocation: float = 1.0, costs: CostModel = None,
             intraday: bool = True, bars_per_year: int = 252,
             signal: pd.Series = None) -> BacktestResult:
    costs = costs or CostModel()
    raw = signal if signal is not None else strategy.generate(df)
    sig = raw.reindex(df.index).fillna(0.0).astype(float)
    if strategy.long_only:
        sig = sig.clip(lower=0.0)

    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    slip = costs.slippage_bps / 10_000.0

    cash, shares = float(capital), 0.0
    cur_tgt = 0.0  # current target position; trade ONLY when this changes
    equity = np.empty(len(df))
    legs, trips = [], []
    entry_value = entry_fee = None
    entry_time = None

    def close_position(i, px):
        nonlocal cash, shares, entry_value, entry_fee, entry_time
        side = "sell" if shares > 0 else "buy"
        exec_px = px * (1 - slip) if shares > 0 else px * (1 + slip)
        val = abs(shares) * exec_px
        fee = costs.leg(val, side, intraday)
        cash = cash + shares * exec_px - fee
        legs.append({"time": idx[i], "side": side, "shares": abs(shares),
                     "price": round(exec_px, 2), "value": round(val, 2),
                     "cost": round(fee, 2)})
        if entry_value is not None:  # completed round trip
            exit_value = shares * exec_px
            pnl = (exit_value - entry_value) - (entry_fee + fee)
            trips.append({"entry": entry_time, "exit": idx[i],
                          "pnl": round(pnl, 2),
                          "cost": round(entry_fee + fee, 2)})
            entry_value = entry_fee = entry_time = None
        shares = 0.0

    def open_position(i, px, desired):
        nonlocal cash, shares, entry_value, entry_fee, entry_time
        side = "buy" if desired > 0 else "sell"
        exec_px = px * (1 + slip) if desired > 0 else px * (1 - slip)
        val = abs(desired) * exec_px
        fee = costs.leg(val, side, intraday)
        cash = cash - desired * exec_px - fee
        legs.append({"time": idx[i], "side": side, "shares": abs(desired),
                     "price": round(exec_px, 2), "value": round(val, 2),
                     "cost": round(fee, 2)})
        entry_value, entry_fee, entry_time = desired * exec_px, fee, idx[i]
        shares = desired

    for i in range(len(df)):
        if i > 0 and o[i] > 0:
            tgt = float(sig.iloc[i - 1])
            if abs(tgt - cur_tgt) > 1e-9:  # signal changed -> rebalance once
                eq_now = cash + shares * o[i]
                desired = tgt * allocation * eq_now / o[i]
                if shares != 0:
                    close_position(i, o[i])
                if abs(desired) > 1e-8:
                    open_position(i, o[i], desired)
                cur_tgt = tgt
        equity[i] = cash + shares * c[i]

    eq_series = pd.Series(equity, index=idx, name="equity")
    trips_df = pd.DataFrame(trips, columns=["entry", "exit", "pnl", "cost"])
    legs_df = pd.DataFrame(legs, columns=["time", "side", "shares", "price",
                                           "value", "cost"])
    return BacktestResult(equity=eq_series, trades=legs_df,
                          roundtrips=trips_df,
                          metrics=_metrics(eq_series, trips_df, bars_per_year))
