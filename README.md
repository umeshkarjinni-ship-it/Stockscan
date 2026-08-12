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
"# Stockscan" 
