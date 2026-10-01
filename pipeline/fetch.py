"""fetch.py -- download daily bars for the universe into data/.

Reads config/universe.yaml, downloads each symbol's full daily history from
Yahoo Finance (free) via engine.data.YFinanceProvider, and writes one
parquet per symbol: data/daily_<SYMBOL>.parquet.

Failures are graceful: a symbol that fails keeps its previous file (if any)
and is recorded in data/_manifest.json, so tune/signals can proceed with
whatever is available -- delisted or flaky names never silently poison the
universe.

Usage:
    python -m pipeline.fetch [--symbols RELIANCE.NS,INFY.NS] [--start 2018-01-01]
"""
import argparse
import json
import time
from datetime import date

import pandas as pd

from engine.data import YFinanceProvider
from .common import (DATA_DIR, load_pipeline_config, load_universe,
                     setup_logging)

log = setup_logging("fetch")


def _save(df: pd.DataFrame, symbol: str):
    key = symbol.replace(".", "_")
    fp = DATA_DIR / f"daily_{key}.parquet"
    try:
        df.to_parquet(fp)
    except Exception:
        df.to_csv(DATA_DIR / f"daily_{key}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="comma-separated override, e.g. RELIANCE.NS,INFY.NS")
    ap.add_argument("--start", default=None,
                    help="history start date (default from pipeline.yaml)")
    args = ap.parse_args()

    cfg = load_pipeline_config()
    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else load_universe())
    start = args.start or cfg.get("data_start", "2018-01-01")
    end = date.today().isoformat()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    provider = YFinanceProvider()
    manifest, ok, failed = {}, [], []

    for i, sym in enumerate(symbols, 1):
        try:
            # fresh download each day: always consistent, no seam artifacts
            # from incremental appends across splits/dividends
            df = provider.daily(sym, start, end, use_cache=False)
            _save(df, sym)
            manifest[sym] = {"rows": len(df),
                             "start": str(df.index[0].date()),
                             "end": str(df.index[-1].date()),
                             "status": "ok"}
            ok.append(sym)
        except Exception as e:
            manifest[sym] = {"status": f"failed: {e}"}
            failed.append(sym)
            log.warning("fetch failed for %s: %s", sym, e)
        if i % 10 == 0:
            log.info("progress %d/%d", i, len(symbols))
        time.sleep(0.4)  # be polite to the free endpoint

    with open(DATA_DIR / "_manifest.json", "w") as f:
        json.dump({"as_of": end, "ok": ok, "failed": failed,
                   "symbols": manifest}, f, indent=2)
    log.info("done: %d ok, %d failed of %d", len(ok), len(failed),
             len(symbols))
    if failed:
        log.info("failed: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
