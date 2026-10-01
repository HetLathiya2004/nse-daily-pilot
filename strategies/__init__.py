"""Strategy plugins for the daily pilot.

Each strategy is a monthly-rebalanced, long-only portfolio module exposing:

- NAME: registry name
- DESCRIPTION: one-liner
- PARAM_GRID: small grid for walk-forward tuning
- WARMUP_MONTHS: months of history needed before the first evaluated month
- run(closes_daily, params, capital, costs, start_pos) -> dict with
  "equity" (monthly Series starting at capital) and "metrics"
- current_picks(closes_daily, params) -> dict describing the portfolio the
  strategy would hold for the upcoming month, ranked on data that is already
  public (last complete month-end) -- no lookahead.

Monthly cadence is deliberate: daily-rebalanced portfolios die on Indian
delivery costs; the research (Jegadeesh-Titman; Moskowitz-Ooi-Pedersen)
rebalances monthly too.
"""
from . import ts_mom, xsmom

ALL = {"xsmom": xsmom, "ts_mom": ts_mom}
