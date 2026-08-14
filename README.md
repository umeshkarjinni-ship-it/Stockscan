# NiftyPulsePro

An NSE/BSE weekly & monthly trend-change **stock scanner** with an optional
**backtesting engine**, built around a Volatility Stop + ATR + KAMA trend
filter + RSI + Volume model (mirrors the companion TradingView Pine Script
in `pine/`).

> Research / decision-support tool — **not financial advice**. Signals are
> pure price/volume technicals with no fundamental or macro context.
> Validate before acting on anything this produces.

## What's in here

| File / folder | Purpose |
|---|---|
| `nse_scanner.py` | Main scanner. Loads the universe, pulls price history via `yfinance`, computes signals, writes CSVs to `signals/`, optionally emails/Telegrams a summary. |
| `signal_report.py` | Turns the raw scan into a ranked daily report (`signals/daily_signal_report.csv`, top-20 buy/sell watchlists). |
| `rank_buy.py` | Further ranks the day's BUY candidates into `signals/daily_top20_buy_ranked.csv`. |
| `build_universe.py` | Builds `stocks_universe_full.csv` from NSE's official equity + index-constituent lists (Large/Mid/Small cap tagging). |
| `daily_runner.py` | Optional always-on Python scheduler if you can't use cron / GitHub Actions. |
| `run_backtest.py` | Runs a historical replay of the scanner's signals over `backtest/`. |
| `test_simulator.py` | Minimal smoke test for `backtest.simulator`. |
| `ml/train_model.py` | Trains a real, validated classifier (gradient boosting) on `backtest_trades.csv` to predict win probability — see "ML win-probability model" below. |
| `ml/predict.py` | Scores today's BUY candidates with the trained model, producing `daily_top20_buy_ml_ranked.csv`. |
| `build_dashboard.py` | Reads the day's CSVs/reports and generates a single-page HTML dashboard at `docs/index.html` — so you can check results by opening one page instead of digging through CSVs. |
| `docs/index.html` | The generated dashboard. Publish it with GitHub Pages (see below) for a permanent URL, or just open the file locally in a browser. |
| `backtest/` | Backtesting package: replay engine, trade simulator, exit rules, performance metrics, charts, and text report. |
| `stocks_universe.csv` | Small starter universe (hand-picked). |
| `stocks_universe_full.csv` | Full ~1,900-symbol universe with Category tags, as produced by `build_universe.py`. |
| `pine/` | Companion TradingView Pine Script indicator + strategy (same logic, for chart/manual use). |
| `.github/workflows/daily-scan.yml` | Runs the scanner (+report +ranking) automatically on NSE trading days and uploads the CSVs as build artifacts. |

Generated/runtime folders (`signals/`, `reports/`, `cache/`, `logs/`) are
kept in the repo as empty placeholders (`.gitkeep`) — their contents are
git-ignored since they're recreated on every run.

## Setup

```bash
git clone <your-repo-url>
cd NiftyPulsePro
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Usage

```bash
# 1. (optional) rebuild the full NSE universe — otherwise the checked-in
#    stocks_universe_full.csv is used as-is
python build_universe.py

# 2. run the scanner
python nse_scanner.py

# 3. turn the scan into a ranked daily report
python signal_report.py
python rank_buy.py
```

Results land in `signals/` — most importantly `scan_<timestamp>.csv`,
`daily_signal_report.csv`, `daily_top20_buy_ranked.csv`, and
`daily_top20_sell_watchlist.csv`.

### Backtesting

```bash
python run_backtest.py
```

Tune capital, costs, holding periods, and exit rules (VSTOP reversal / ATR
stop / trailing stop / profit target) in `backtest/config.py`.

## ML win-probability model

`rank_buy.py`'s `BuyScore` is a hand-picked 100-point rubric (fixed
points for RSI bands, ADX strength, etc.) — never statistically
validated against what actually happened afterward. `ml/` adds a real
alternative: a gradient-boosting classifier trained on your own
`backtest_trades.csv`, evaluated on a held-out, chronologically later
slice of trades (not randomly shuffled, to avoid lookahead bias).

**Setup (one-time, then periodically):**

```bash
# 1. Generate a FULL backtest (not the 50-symbol quick test) — see
#    "Backtesting" above. You need a few hundred+ trades minimum.
python run_backtest.py
python -m backtest.simulator

# 2. Train the model
python -m ml.train_model
```

This prints validation accuracy, ROC-AUC, and feature importances so you
can judge honestly whether it's actually predictive — an ROC-AUC near
0.5 means "barely better than a coin flip," and the script tells you so
directly rather than hiding it. It saves `ml/model.pkl`.

**Daily use:** once `ml/model.pkl` exists, the GitHub Actions workflow
automatically runs `ml/predict.py` after `rank_buy.py` each day, adding
an `MLWinProbability` column (0-100%) to `daily_top20_buy_ml_ranked.csv`
and to the dashboard — shown right next to the original `BuyScore` so you
can compare them. If no model exists yet, this step just skips quietly.

**Re-train periodically** (e.g. monthly) using a freshly regenerated
`backtest_trades.csv`, so the model keeps learning from recent market
behavior instead of going stale.

## Dashboard

Instead of opening CSVs, run:

```bash
python build_dashboard.py
```

and open `docs/index.html` in a browser — a single page showing today's
BUY/SELL signals, the paper trade tracker, and the backtest summary.

The GitHub Actions workflow regenerates and commits this automatically
after every daily scan. To get a permanent URL for it instead of
downloading the file each time:

1. On GitHub, go to your repo's **Settings → Pages**.
2. Under "Build and deployment", set **Source** to "Deploy from a branch".
3. Set **Branch** to `main`, folder to `/docs`, then **Save**.
4. After the next scan runs, your dashboard is live at
   `https://<your-username>.github.io/<repo-name>/`.

### Scheduling

Two options, pick one:

- **GitHub Actions (recommended)** — already set up in
  `.github/workflows/daily-scan.yml`, runs weekdays at 10:00 UTC
  (3:30 PM IST, after NSE close) and uploads the CSVs as a downloadable
  artifact. Trigger it manually anytime from the **Actions** tab.
- **`daily_runner.py`** — if you'd rather run it yourself on a machine
  that's always on, `python daily_runner.py` keeps a process alive and
  fires the scan every weekday at a fixed local time.

### Email / Telegram alerts (optional)

`nse_scanner.py` can email or Telegram you when new BUY/SELL signals show
up. It's disabled unless these environment variables are set (locally via
your shell, or as **GitHub Actions secrets** for the workflow):

| Variable | Purpose |
|---|---|
| `SCANNER_EMAIL_FROM` | Sending Gmail address |
| `SCANNER_EMAIL_TO` | Recipient address |
| `SCANNER_EMAIL_APP_PASSWORD` | Gmail [app password](https://support.google.com/accounts/answer/185833) (not your normal password) |
| `SCANNER_TELEGRAM_TOKEN` | Telegram bot token |
| `SCANNER_TELEGRAM_CHAT_ID` | Telegram chat ID to post to |

Leave any/all unset to skip that channel — the script checks for them and
silently no-ops if they're missing.

## What was cleaned up from the original working folder

This repo is a consolidated version of a larger working directory. Removed
before publishing:

- Superseded scanner/report drafts (`nse_scanner_v1.0.py`,
  `nse_scanner_2.py`, `nse_scanner_before_*.py`, `signal_report_backup.py`,
  `signal_report_before_*.py`, `signal_report_working_backup.py`) — the
  current `nse_scanner.py` / `signal_report.py` are the up-to-date
  versions these evolved into. Recover any of them from your original
  folder if you specifically need old behavior.
- `.git/` history and `__pycache__/` (regenerate with `git init`).
- Raw NSE download caches (`EQUITY_L.csv`, `ind_nifty*.csv`) — these are
  only a manual-fallback input to `build_universe.py`, not something the
  scanner reads directly; git-ignored, not deleted from your original
  folder.
- Generated run artifacts: old `signals/scan_*.csv`, `daily_*.csv`, `.png`
  charts, `cache/checkpoint.json`, `logs/backtest.log` — all reproducible
  by rerunning the scripts.
- A personal email address that was hardcoded as the default
  `EMAIL_TO` — now sourced only from `SCANNER_EMAIL_TO`, so it isn't
  committed to a possibly-public repo.
- The two near-duplicate workflow files (`scanner.yml`, `daily-run.yml`)
  merged into one (`daily-scan.yml`) that runs scanner → report → ranking
  and uploads all resulting CSVs.

## Disclaimer

This tool is for research and education only. It does not constitute
investment advice. Markets carry risk of loss; past signal performance
(including anything shown by the backtester) does not guarantee future
results.
