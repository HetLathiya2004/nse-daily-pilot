# NSE Daily Pilot

A free, always-on daily pipeline for Indian equities: it **backtests its own
strategies every day**, promotes a challenger to champion only when it proves
itself on unseen data, and publishes a next-session signal plan
(entries, stop-losses, targets, position sizes) to a public dashboard.

Paper-only. Research tooling, not investment advice.

## How it works

Every weekday at 17:00 IST (after the NSE close), GitHub Actions runs:

1. **`pipeline/fetch.py`** — downloads adjusted daily bars for the universe
   (`config/universe.yaml`, Nifty 50) from Yahoo Finance (free, no API key).
2. **`pipeline/tune.py`** — the self-improvement loop:
   - rolling window: grid-search on the last 4 years (in-sample), evaluate on
     the most recent 1 year (out-of-sample), net of the full Indian cost
     model (STT, stamp duty, exchange charges, GST, SEBI, brokerage,
     slippage);
   - every test slice is prepended with warmup history so indicators are
     never cold (cold indicators silently fake results);
   - a challenger replaces the champion **only if** its OOS Sharpe beats the
     champion's by ≥ 0.10 **and** its OOS net return is positive **and** its
     OOS max drawdown is within the cap — otherwise the champion stays;
   - every run is appended to `model_registry.json` (params, IS/OOS metrics,
     promoted yes/no, reason).
3. **`pipeline/signals.py`** — builds the next-session plan from the current
   champion: long-only picks, entry reference (previous close; the backtest
   fills at the next open), stop-loss = entry − 2×ATR(14),
   target = entry + 2R, position size capped at 1% risk per trade and 25% per
   name. Writes `site/data/signals.json`, `site/data/model_card.json`,
   `site/data/last_run.json`, which the workflow commits back to the repo.
4. **`deploy-pages`** — publishes `site/` to GitHub Pages.

The dashboard (`site/index.html`, no build step, no server) shows the last
run time, the champion model card with its OOS evidence and promotion
history, and the per-symbol plan.

Strategies live in `strategies/`: `xsmom` (cross-sectional momentum,
top-decile by trailing formation-month return skipping 1 month) and `ts_mom`
(time-series momentum, hold names with positive trailing return). Both are
monthly-rebalanced, long-only portfolios. Correctness rules inherited from
the proven v1 engine: signals execute at the next bar open (no lookahead),
full Indian cost model on every trade, point-in-time data handling (no
forward-fill across delistings; partial histories like ETERNAL.NS/JIOFIN.NS
are excluded from rankings they lack history for).

## Setup — about 10 minutes

### 1. Create the GitHub repo
1. Go to [github.com/new](https://github.com/new).
2. Name it e.g. `nse-daily-pilot`. **Public is recommended**: public repos get
   unlimited free Actions minutes and free Pages. (Private also works — a
   daily ~3-minute job uses ~90 min/month of the ~2000 free minutes.)
3. Do **not** initialize with a README (you're pushing your own files).

### 2. Push this folder
From this directory:

```bash
git init
git add .
git commit -m "nse daily pilot: initial scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/nse-daily-pilot.git
git push -u origin main
```

### 3. Enable Pages (one click)
1. In the repo: **Settings → Pages**.
2. Under **Build and deployment → Source**, select **GitHub Actions**.
3. Done. Your dashboard will be live at:

   `https://<your-username>.github.io/nse-daily-pilot/`

### 4. Trigger the first run
Go to **Actions → daily-pipeline → Run workflow** (the schedule only fires
on weekdays at 11:30 UTC). The first run takes a few minutes: it downloads
~7 years of history for ~50 symbols, runs the walk-forward tune, emits
signals, and commits the JSON. The `deploy-pages` workflow then publishes
the dashboard. Open your Pages link — if the run hasn't finished yet you'll
see a "waiting for first pipeline run" note.

That's it. From then on it runs itself every weekday at 17:00 IST.

## Changing things

| What | Where |
|---|---|
| Universe (add/remove symbols) | `config/universe.yaml` |
| Capital, risk per trade, position cap | `config/pipeline.yaml` |
| Promotion thresholds (Sharpe edge, DD cap) | `config/pipeline.yaml` → `tune:` |
| Schedule | `.github/workflows/daily.yml` → `cron` (UTC; 11:30 UTC = 17:00 IST) |
| New strategy | add a module in `strategies/` exposing `NAME`, `PARAM_GRID`, `WARMUP_MONTHS`, `run()`, `current_picks()`; register it in `strategies/__init__.py` |

To test a change before pushing: `pip install -r requirements.txt`, then
`python -m pipeline.fetch && python -m pipeline.tune && python -m pipeline.signals`,
and open `site/index.html` — it reads the local `site/data/*.json` files.

Smoke test (2–3 symbols, tiny grid, no long download):

```bash
python -m pipeline.fetch --symbols RELIANCE.NS,INFY.NS,HDFCBANK.NS --start 2021-01-01
python -m pipeline.tune --tiny
python -m pipeline.signals
```

## Reading the dashboard

- **Next-session plan**: one card per long pick — entry reference, stop,
  2R target, quantity, notional, and rupees at risk, plus trailing-12-month
  context per symbol (return, Sharpe, max drawdown).
- **Model card**: the champion strategy and its parameters, OOS Sharpe /
  return / drawdown / win rate / costs paid over the last 12 months, the OOS
  equity curve, and how many times it has been promoted.
- **Self-improvement history**: every daily run's challenger, its OOS
  Sharpe, and why it was or wasn't promoted. If the champion keeps surviving,
  that *is* the system working — it only changes when something proves
  itself on unseen data.

## Honest limits

- **End-of-day data only.** These are swing/positional signals computed after
  the close — not live intraday calls.
- **Yahoo Finance is free but unofficial**: data can be delayed, revised, or
  briefly unavailable; failed symbols are skipped and listed on the
  dashboard rather than guessed.
- **Paper only.** Nothing here places orders. The sizing math assumes you
  could trade at the next open with 5 bps slippage — real fills differ.
- **Survivorship bias**: the universe is today's Nifty 50; historical
  constituents that were removed (and their failures) are not in the test.
- **Past OOS performance ≠ future returns.** The promotion gate makes
  overfitting harder, not impossible.
- **Exchange holidays** are not modelled: the "next session" date is the
  next weekday — verify it's an NSE trading day.
- This is a research tool, not investment advice and not a recommendation
  on any specific security.
