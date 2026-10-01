"""signals.py -- next-session trading plan from the current champion.

Reads model_registry.json, takes the champion strategy + params, ranks the
universe on the last COMPLETE month-end (a partial current month is never
used), and emits a per-symbol plan into site/data/:

- site/data/signals.json     -- next session's entries, stops, targets, sizes
- site/data/model_card.json  -- champion params, OOS evidence, sparkline data
- site/data/last_run.json    -- timestamps, coverage, data-through date

Signal construction rules (also documented in README.md):
- Direction is LONG ONLY (explicit; the strategies are long-only).
- entry_ref = previous close. The backtest executes at the NEXT bar's open,
  so treat entry_ref as a reference -- the real fill is the next open.
- stop-loss = entry - 2 x ATR(14); target = entry + 4 x ATR(14), i.e. a 2R
  target. ATR is Wilder's, on adjusted daily bars.
- Position size = min(risk_per_trade x capital / (entry - stop),
  max_position_pct x capital / entry), rounded down to whole shares.
  Picks that would need < 1 share are skipped, not fudged.
- Picks refresh MONTHLY (strategy rebalance cadence); entry/stop/target
  levels refresh DAILY from the latest bars.

Per-symbol "evidence" stats are trailing-12m context from daily bars
(total return, Sharpe, max drawdown) -- the strategy-level OOS metrics in
the model card are the actual backtested evidence.

Usage:
    python -m pipeline.signals
"""
import json
from datetime import timedelta, timezone

import pandas as pd

from engine.costs import CostModel
from engine.data import closes_frame, load_universe
from strategies import ALL as STRATEGIES
from .common import (DATA_DIR, SITE_DATA_DIR, load_pipeline_config,
                     load_registry, load_universe as load_universe_cfg,
                     now_utc, setup_logging)
from .tune import _windows

log = setup_logging("signals")
IST = timezone(timedelta(hours=5, minutes=30))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def _trailing_12m_stats(close: pd.Series) -> dict:
    s = close.dropna().iloc[-252:]
    if len(s) < 60:
        return {"total_return": None, "sharpe": None, "max_drawdown": None,
                "note": "insufficient history"}
    rets = s.pct_change().dropna()
    vol = rets.std()
    sharpe = float(rets.mean() / vol * (252 ** 0.5)) if vol > 0 else 0.0
    dd = float(((s - s.cummax()) / s.cummax()).min())
    return {"total_return": round(float(s.iloc[-1] / s.iloc[0] - 1), 4),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(dd, 4)}


def main():
    cfg = load_pipeline_config()
    capital = float(cfg.get("capital", 100_000))
    risk_pct = float(cfg.get("risk_per_trade", 0.01))
    max_pos_pct = float(cfg.get("max_position_pct", 0.25))
    atr_n = int(cfg.get("atr_period", 14))
    atr_mult = float(cfg.get("atr_stop_multiple", 2.0))
    rr = float(cfg.get("target_rr", 2.0))
    tune_cfg = cfg.get("tune", {})
    costs = CostModel(**cfg.get("costs", {}))

    registry = load_registry()
    champ = registry.get("champion")
    if not champ:
        raise SystemExit("No champion in model_registry.json -- "
                         "run pipeline.tune first.")
    mod = STRATEGIES[champ["strategy"]]
    params = champ["params"]
    log.info("champion: %s %s", champ["strategy"], params)

    symbols = load_universe_cfg()
    bars = load_universe(symbols, cfg.get("data_start", "2018-01-01"),
                         "2100-01-01", DATA_DIR)
    if not bars:
        raise SystemExit("No symbol data found -- run pipeline.fetch first.")
    closes = closes_frame(bars)
    data_through = closes.index[-1].date().isoformat()

    plan = mod.current_picks(closes, params)

    # next trading session (weekday step; NSE holidays not modelled --
    # verify the date is a trading day before acting)
    next_session = (pd.Timestamp(data_through)
                    + pd.tseries.offsets.BDay(1)).date().isoformat()

    positions, skipped = [], []
    for p in plan["picks"]:
        sym = p["symbol"]
        df = bars.get(sym)
        if df is None or df.empty:
            skipped.append({"symbol": sym, "reason": "no daily bars"})
            continue
        entry = float(df["close"].iloc[-1])
        atr = _atr(df, atr_n).iloc[-1]
        if not (entry > 0 and atr > 0 and pd.notna(atr)):
            skipped.append({"symbol": sym, "reason": "bad ATR/entry"})
            continue
        atr = float(atr)
        risk_dist = atr_mult * atr
        stop = round(entry - risk_dist, 2)
        target = round(entry + rr * risk_dist, 2)
        qty_risk = int((risk_pct * capital) // risk_dist)
        qty_cap = int((max_pos_pct * capital) // entry)
        qty = min(qty_risk, qty_cap)
        if qty < 1:
            skipped.append({"symbol": sym,
                            "reason": "position < 1 share at risk limits"})
            continue
        notional = round(qty * entry, 2)
        positions.append({
            "symbol": sym,
            "direction": "long",  # long-only, explicit
            "strategy_weight": p["weight"],
            "formation_return": p["formation_return"],
            "entry_ref": round(entry, 2),
            "entry_note": ("previous close; backtest fills at next "
                           "session open"),
            "atr14": round(atr, 2),
            "stop_loss": stop,
            "target": target,
            "qty": qty,
            "notional_rs": notional,
            "risk_rs": round(qty * risk_dist, 2),
            "trailing_12m": _trailing_12m_stats(df["close"]),
        })

    now = now_utc()
    signals = {
        "signal_date": next_session,
        "generated_at_utc": now.isoformat(),
        "generated_at_ist": now.astimezone(IST).isoformat(),
        "data_through": data_through,
        "strategy": champ["strategy"],
        "params": params,
        "as_of_month": plan["as_of_month"],
        "next_rebalance_month": plan["next_rebalance_month"],
        "capital_rs": capital,
        "risk_per_trade": risk_pct,
        "max_position_pct": max_pos_pct,
        "direction": "long",
        "rules": {
            "entry": "previous close (reference; fills at next session open)",
            "stop_loss": f"entry - {atr_mult} x ATR({atr_n})",
            "target": f"entry + {rr}R (R = {atr_mult} x ATR({atr_n}))",
            "sizing": ("min(risk_per_trade x capital / (entry - stop), "
                       "max_position_pct x capital / entry), whole shares"),
            "rebalance": ("picks refresh monthly at month-end; "
                          "levels refresh daily"),
        },
        "universe": {"ranked": plan["universe_ranked"],
                     "loaded": len(bars), "listed": len(symbols)},
        "positions": positions,
        "skipped": skipped,
    }

    # --- model card: champion + OOS evidence + sparkline -----------------
    monthly_n = len(closes.resample("ME").last())
    warmup = max(m.WARMUP_MONTHS for m in STRATEGIES.values())
    _, oos_start, _, oos_m = _windows(
        monthly_n, int(tune_cfg.get("is_months", 48)),
        int(tune_cfg.get("oos_months", 12)), warmup)
    oos = mod.run(closes, params, capital=capital, costs=costs,
                  start_pos=oos_start)
    eq = oos["equity"]
    spark = [[int(ts.value // 10 ** 6), round(float(v), 2)]
             for ts, v in eq.items()]
    runs = registry.get("runs", [])
    promotions = sum(1 for r in runs if r.get("promoted"))

    def _summarize_run(r):
        contenders = r.get("contenders", [])
        best = (max(contenders, key=lambda c: c["oos_metrics"]["sharpe"])
                if contenders else None)
        ch = r.get("champion")
        return {
            "date": r["date"],
            "champion": [ch["strategy"], ch["params"]] if ch else None,
            "challenger": ([best["strategy"], best["params"],
                            round(best["oos_metrics"]["sharpe"], 3)]
                           if best else None),
            "promoted": r["promoted"],
            "reason": r["reason"],
        }
    model_card = {
        "champion": {
            "strategy": champ["strategy"],
            "params": params,
            "description": mod.DESCRIPTION,
            "promoted_on": champ.get("promoted_on"),
            "promotions_to_date": promotions,
            "runs_to_date": len(runs),
        },
        "oos_evidence": {
            "window_months": oos_m,
            "note": ("walk-forward out-of-sample, net of full Indian costs; "
                     "past OOS performance does not predict future returns"),
            **oos["metrics"],
        },
        "oos_equity": spark,
        "history": [_summarize_run(r) for r in runs[-30:]],
    }

    manifest = {}
    try:
        with open(DATA_DIR / "_manifest.json") as f:
            manifest = json.load(f)
    except Exception:
        pass

    last_run = {
        "generated_at_utc": now.isoformat(),
        "generated_at_ist": now.astimezone(IST).isoformat(),
        "data_through": data_through,
        "next_session": next_session,
        "universe_listed": len(symbols),
        "symbols_loaded": len(bars),
        "symbols_failed": manifest.get("failed", []),
        "positions": len(positions),
        "pipeline": "fetch -> tune -> signals: ok",
    }

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (("signals.json", signals),
                          ("model_card.json", model_card),
                          ("last_run.json", last_run)):
        with open(SITE_DATA_DIR / name, "w") as f:
            json.dump(payload, f, indent=2)
    log.info("wrote site/data: %d positions, %d skipped; signal date %s",
             len(positions), len(skipped), next_session)


if __name__ == "__main__":
    main()
