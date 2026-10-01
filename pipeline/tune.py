"""tune.py -- daily walk-forward champion/challenger self-improvement loop.

Every run (once a day, after the NSE close):

1. Builds monthly closes for the universe from data/.
2. Defines a rolling window: IS = last `is_months` months, OOS = the most
   recent `oos_months` months. Every backtest slice is preceded by
   WARMUP_MONTHS of pure past history -- cold indicators silently produce
   fake results otherwise, so the warmup is mandatory and explicit.
3. For each strategy (champion's and challengers'):
     - challenger: grid-search params on IS, keep the best by Sharpe
       (minimum-rebalance guard against flukes), then evaluate on OOS.
     - champion: evaluate its registry params on the same OOS window.
   All net of the full Indian cost model (STT, stamp, exchange, GST, SEBI,
   brokerage, slippage).
4. Promote the best challenger to champion ONLY if, on the SAME OOS window:
     - its OOS Sharpe beats the champion's by >= `promote_sharpe_edge`
       (default +0.10), AND
     - its OOS net total return > 0, AND
     - its OOS max drawdown is within `max_drawdown` cap, AND
     - both sides have enough OOS rebalances to mean something.
   Otherwise the champion stays. This is the anti-overfit gate: the model
   can only improve by proving itself on unseen data.
5. Appends the run to model_registry.json (date, windows, champion params,
   challenger params, IS/OOS metrics, promoted, reason).

Usage:
    python -m pipeline.tune [--tiny]   # --tiny: smoke-test grid + windows
"""
import argparse
from itertools import product

from engine.costs import CostModel
from engine.data import closes_frame, load_universe
from strategies import ALL as STRATEGIES
from .common import (DATA_DIR, load_pipeline_config, load_registry,
                     load_universe as load_universe_cfg, now_utc,
                     save_registry, setup_logging)

log = setup_logging("tune")


def _param_combos(grid: dict):
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def _windows(n_months: int, is_months: int, oos_months: int, warmup: int):
    """Return (is_start_pos, oos_start_pos), shrinking IS if history is short.
    Raises if there isn't enough history for an honest test."""
    oos = min(oos_months, max(6, n_months // 4))
    is_m = min(is_months, n_months - oos - warmup)
    if n_months < warmup + 12 or is_m < 6:
        raise ValueError(
            f"Only {n_months} months of history; need at least "
            f"{warmup + 12} for warmup + a minimal IS/OOS split.")
    return n_months - oos - is_m, n_months - oos, is_m, oos


def _grid_search(mod, closes, is_start, capital, costs, min_rebalances):
    grid = dict(mod.PARAM_GRID)
    best, best_sharpe, best_m = None, float("-inf"), None
    tried = 0
    for params in _param_combos(grid):
        tried += 1
        r = mod.run(closes, params, capital=capital, costs=costs,
                    start_pos=is_start)
        m = r["metrics"]
        score = (m["sharpe"] if m["num_rebalances"] >= min_rebalances
                 else float("-inf"))
        if score > best_sharpe:
            best, best_sharpe, best_m = params, score, m
    if best is None:
        raise ValueError(f"{mod.NAME}: no param set cleared the "
                         f"min-rebalance guard on IS.")
    log.info("%s IS: %d combos -> %s sharpe=%.3f", mod.NAME, tried, best,
             best_sharpe)
    return best, best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                    help="smoke test: tiny grids and short windows")
    args = ap.parse_args()

    cfg = load_pipeline_config()
    capital = float(cfg.get("capital", 100_000))
    tune_cfg = cfg.get("tune", {})
    is_months = int(tune_cfg.get("is_months", 48))
    oos_months = int(tune_cfg.get("oos_months", 12))
    min_is_reb = int(tune_cfg.get("min_is_rebalances", 8))
    min_oos_reb = int(tune_cfg.get("min_oos_rebalances", 4))
    edge = float(tune_cfg.get("promote_sharpe_edge", 0.10))
    max_dd = float(tune_cfg.get("max_drawdown_cap", 0.40))
    costs = CostModel(**cfg.get("costs", {}))

    if args.tiny:
        STRATEGIES["xsmom"].PARAM_GRID = {"formation": [3, 6], "hold": [1]}
        STRATEGIES["ts_mom"].PARAM_GRID = {"lookback": [3, 6]}
        is_months, oos_months = 18, 6
        min_is_reb, min_oos_reb = 3, 2

    bars = load_universe(load_universe_cfg(), cfg.get("data_start", "2018-01-01"),
                         "2100-01-01", DATA_DIR)
    if not bars:
        raise SystemExit("No symbol data found -- run pipeline.fetch first.")
    closes = closes_frame(bars)
    monthly_n = len(closes.resample("ME").last())
    log.info("universe: %d symbols, %d months of history",
             len(bars), monthly_n)

    warmup = max(m.WARMUP_MONTHS for m in STRATEGIES.values())
    is_start, oos_start, is_m, oos_m = _windows(monthly_n, is_months,
                                                oos_months, warmup)
    log.info("windows: IS months [%d:%d] (%d), OOS months [%d:%d] (%d), "
             "warmup %d", is_start, oos_start, is_m, oos_start, monthly_n,
             oos_m, warmup)

    registry = load_registry()
    champ = registry.get("champion")

    # --- evaluate champion on the same OOS window -----------------------
    champ_oos = None
    if champ:
        mod = STRATEGIES[champ["strategy"]]
        champ_oos = mod.run(closes, champ["params"], capital=capital,
                            costs=costs, start_pos=oos_start)["metrics"]
        log.info("champion %s %s OOS sharpe=%.3f ret=%.3f dd=%.3f",
                 champ["strategy"], champ["params"], champ_oos["sharpe"],
                 champ_oos["total_return"], champ_oos["max_drawdown"])

    # --- challengers: grid-search on IS, evaluate on OOS ----------------
    contenders = []
    for name, mod in STRATEGIES.items():
        try:
            best_params, is_mets = _grid_search(mod, closes, is_start,
                                                capital, costs, min_is_reb)
        except ValueError as e:
            log.warning("skipping %s: %s", name, e)
            continue
        oos_mets = mod.run(closes, best_params, capital=capital,
                           costs=costs, start_pos=oos_start)["metrics"]
        contenders.append({"strategy": name, "params": best_params,
                           "is_metrics": is_mets, "oos_metrics": oos_mets})
        log.info("challenger %s %s OOS sharpe=%.3f ret=%.3f dd=%.3f", name,
                 best_params, oos_mets["sharpe"], oos_mets["total_return"],
                 oos_mets["max_drawdown"])

    if not contenders:
        raise SystemExit("No challenger produced a tradeable IS result.")

    best = max(contenders, key=lambda c: c["oos_metrics"]["sharpe"])
    bm = best["oos_metrics"]

    # --- promotion gate -------------------------------------------------
    def gates_ok(m):
        return (m["num_rebalances"] >= min_oos_reb
                and m["total_return"] > 0
                and m["max_drawdown"] >= -max_dd)

    promoted, reason = False, ""
    if champ is None:
        promoted = True
        reason = ("initial seeding: " + best["strategy"] + " " +
                  str(best["params"]) +
                  (" (passed minimum OOS gates)" if gates_ok(bm)
                   else " (BELOW minimum OOS gates -- weak start, "
                        "will be replaced when something proves itself)"))
    else:
        cm = champ_oos
        champ_ok = cm["num_rebalances"] >= min_oos_reb
        if not champ_ok:
            reason = (f"champion {champ['strategy']} has too few OOS "
                      f"rebalances ({cm['num_rebalances']}) -- keeping it, "
                      f"no promotion this run")
        elif not gates_ok(bm):
            reason = (f"challenger {best['strategy']} failed minimum OOS "
                      f"gates (ret={bm['total_return']}, "
                      f"dd={bm['max_drawdown']}, "
                      f"rebal={bm['num_rebalances']}) -- champion stays")
        elif bm["sharpe"] >= cm["sharpe"] + edge:
            promoted = True
            reason = (f"challenger {best['strategy']} {best['params']} "
                      f"OOS Sharpe {bm['sharpe']:.3f} beats champion "
                      f"{champ['strategy']} {champ['params']} "
                      f"({cm['sharpe']:.3f}) by >= {edge}")
        else:
            reason = (f"challenger {best['strategy']} OOS Sharpe "
                      f"{bm['sharpe']:.3f} did not beat champion "
                      f"{cm['sharpe']:.3f} by {edge} -- champion stays")

    if promoted:
        registry["champion"] = {
            "strategy": best["strategy"],
            "params": best["params"],
            "promoted_on": now_utc().date().isoformat(),
        }
        log.info("PROMOTED: %s", reason)
    else:
        log.info("no promotion: %s", reason)

    registry.setdefault("runs", []).append({
        "date": now_utc().date().isoformat(),
        "is_months": is_m,
        "oos_months": oos_m,
        "warmup_months": warmup,
        "capital": capital,
        "champion": ({"strategy": champ["strategy"], "params": champ["params"],
                      "oos_metrics": champ_oos} if champ else None),
        "contenders": contenders,
        "promoted": promoted,
        "reason": reason,
    })
    save_registry(registry)
    log.info("registry updated: champion=%s", registry["champion"])


if __name__ == "__main__":
    main()
