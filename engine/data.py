"""Free, REAL market data for NSE with a local disk cache.

Zero-cost data: daily bars are downloaded from Yahoo Finance's public chart
API (free, no key), cached under data/cache/, and reused. No API keys, no
subscriptions, no mock or synthetic data -- if the cache is empty and the
network is down, the provider raises instead of inventing prices.

Two fetch paths: a direct v8 chart-API client (robust, minimal) with the
`yfinance` library as fallback.

Point-in-time discipline: adjusted closes only (splits/dividends), and we
drop rows with missing closes -- we never forward-fill across delistings or
corporate actions. Symbols with partial histories (recent listings like
ETERNAL.NS / JIOFIN.NS) are kept; the strategy layer excludes them from any
ranking that needs history they don't have.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _v8_chart(symbol: str, period1: int, period2: int) -> pd.DataFrame:
    """Raw daily OHLCV from Yahoo's v8 chart API. Raises on failure."""
    sym_q = urllib.parse.quote(symbol, safe="")
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{sym_q}"
           f"?interval=1d&period1={period1}&period2={period2}"
           f"&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                payload = json.loads(r.read().decode())
            last_err = None
            break
        except Exception as e:  # transient network/rate-limit blips
            last_err = e
            time.sleep(2 * (attempt + 1))
    if last_err is not None:
        raise ValueError(f"Yahoo v8 request failed for {symbol}: {last_err}")

    chart = payload.get("chart", {})
    result = chart.get("result")
    if not result:
        raise ValueError(f"No data for {symbol}: {chart.get('error')}")
    res = result[0]
    ts = res.get("timestamp") or []
    if not ts:
        raise ValueError(f"No data for {symbol}: empty timestamp series")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"),
        "volume": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s", utc=True))
    tz = res.get("meta", {}).get("exchangeTimezoneName") or "Asia/Kolkata"
    df.index = df.index.tz_convert(tz).tz_localize(None)

    # Adjust for splits/dividends on daily bars using adjclose factors.
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj:
        adj = pd.Series(adj, index=df.index).ffill()
        factor = (adj / df["close"]).replace([float("inf")], 1.0).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
    return df


def _via_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        df.columns = lvl0 if "Close" in lvl0 else df.columns.get_level_values(1)
    df = df.rename(columns=str.lower)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


class YFinanceProvider:
    """NSE OHLCV via Yahoo Finance (free). Symbols like 'RELIANCE.NS'."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache -----------------------------------------------------------
    def _key(self, symbol: str, start: str, end: str) -> str:
        tag = f"{symbol}_1d_{start}_{end}".replace(".", "_").replace("/", "_")
        return tag

    def _load(self, key: str):
        pq = self.cache_dir / f"{key}.parquet"
        cs = self.cache_dir / f"{key}.csv"
        if pq.exists():
            return pd.read_parquet(pq)
        if cs.exists():
            return pd.read_csv(cs, index_col=0, parse_dates=True)
        return None

    def _save(self, key: str, df: pd.DataFrame):
        try:
            df.to_parquet(self.cache_dir / f"{key}.parquet")
        except Exception:
            df.to_csv(self.cache_dir / f"{key}.csv")

    @staticmethod
    def _finalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol}")
        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]].dropna(subset=["close"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        return df[cols].sort_index()

    def _fetch(self, symbol, start, end):
        """Direct v8 API first, yfinance library as fallback."""
        try:
            p1 = int(pd.Timestamp(start).timestamp())
            p2 = int(pd.Timestamp(end).timestamp())
            return self._finalize(_v8_chart(symbol, p1, p2), symbol)
        except Exception as e1:
            try:
                return self._finalize(_via_yfinance(symbol, start, end), symbol)
            except Exception as e2:
                raise ValueError(f"All data sources failed for {symbol}: "
                                 f"v8: {e1}; yfinance: {e2}")

    def daily(self, symbol: str, start: str, end: str,
              use_cache: bool = True) -> pd.DataFrame:
        """Adjusted daily bars for one symbol. Raises if unavailable."""
        key = self._key(symbol, start, end)
        if use_cache:
            cached = self._load(key)
            if cached is not None:
                return cached
        df = self._fetch(symbol, start, end)
        self._save(key, df)
        return df


def load_universe(symbols: list, start: str, end: str,
                  data_dir: Path = None) -> dict:
    """Load cached daily bars written by pipeline/fetch.py.

    Returns {symbol: DataFrame}. Symbols whose file is missing or empty are
    skipped (graceful handling of delisted/failed names) -- the caller logs
    them from the fetch manifest.
    """
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data"
    out = {}
    for sym in symbols:
        fp = data_dir / f"daily_{sym.replace('.', '_')}.parquet"
        cp = data_dir / f"daily_{sym.replace('.', '_')}.csv"
        df = None
        try:
            if fp.exists():
                df = pd.read_parquet(fp)
            elif cp.exists():
                df = pd.read_csv(cp, index_col=0, parse_dates=True)
        except Exception:
            df = None
        if df is not None and not df.empty:
            df = df[(df.index >= start) & (df.index <= end)]
            if not df.empty:
                out[sym] = df
    return out


def closes_frame(bars: dict) -> pd.DataFrame:
    """Align per-symbol daily bars into one closes DataFrame.

    Point-in-time: each symbol keeps only its own real history; a symbol is
    simply absent (NaN) before its listing. The strategy layer drops names
    that lack the history a given ranking needs -- never forward-filled.
    """
    if not bars:
        raise ValueError("No symbols loaded -- cannot build closes frame.")
    frame = pd.DataFrame({s: df["close"] for s, df in bars.items()})
    return frame.sort_index()
