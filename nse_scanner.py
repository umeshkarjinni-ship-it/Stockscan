"""
NSE/BSE Weekly & Monthly Trend-Change Scanner
==============================================
Scans a universe of large/mid/small cap stocks and flags BUY / SELL
trend-change signals using the same logic as the companion TradingView
Pine Script (Volatility Stop + ATR + EMA trend filter + RSI + Volume).

This is a research / decision-support tool, NOT financial advice.
Signals are based purely on price/volume technicals with no fundamental
or macro context. Validate thoroughly before acting on any signal.

SETUP
-----
1. pip install -r requirements.txt
2. Edit stocks_universe.csv to include the stocks you want to track
   (a starter list is provided — replace/expand it with the full NSE
   list from https://www.nseindia.com/market-data/securities-available-for-trading
   or an index constituent list such as Nifty 500).
3. Run once manually to test:  python nse_scanner.py
4. Schedule it to run daily (see "SCHEDULING" section at the bottom of
   this file, or the README notes below).

OUTPUT
------
- Prints a summary table to the console.
- Writes a timestamped CSV to the "signals/" folder with every stock's
  latest Weekly and Monthly trend + signal.
- Optionally emails / sends a Telegram message when new BUY or SELL
  signals are found (disabled by default — see NOTIFY_* settings).
"""

import os
import json
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed on a server/scheduled task
import matplotlib.pyplot as plt

# =========================================================================
# CONFIG — tweak these to match the Pine Script inputs
# =========================================================================
UNIVERSE_CSV = "stocks_universe_full.csv"     # Symbol, Name, Category columns
# For the full ~1900-symbol NSE universe, run build_universe.py first, then
# point this at its output instead:
#   UNIVERSE_CSV = "stocks_universe_full.csv"

INCLUDE_OTHER_CATEGORY = True             # "Other" = micro-caps/recent listings/
                                          # thin liquidity from build_universe.py.
                                          # Set False to scan only the ~500
                                          # Large/Mid/Small cap names.
                                          #
                                          # With True the universe goes from
                                          # ~500 to ~2,075. That extra tier is
                                          # where corrupt price data and
                                          # untradeable illiquidity live: an
                                          # early full-universe backtest gave
                                          # PF 0.94 vs 1.92 on the clean 500,
                                          # including a "-86% trade" that was
                                          # really an unadjusted 7.35x split.
                                          #
                                          # The filters below exist to make this
                                          # setting survivable. Don't disable
                                          # them while this is True.

# ---------------------------------------------------------------------
# Liquidity floor
# ---------------------------------------------------------------------
#
# Applied to every symbol regardless of category. Screens out names you
# couldn't actually trade at size, and where a single small order moves
# price enough to make technical signals meaningless.
# Set MIN_AVG_TURNOVER to 0 to disable.

MIN_PRICE = 10.0                  # rupees; sub-10 names are tick-dominated
MIN_AVG_TURNOVER = 10_000_000     # rupees/day (~1 crore), 20-day average

# ---------------------------------------------------------------------
# Split / bad-tick guard
# ---------------------------------------------------------------------
#
# yfinance sometimes serves unadjusted history across a corporate action,
# producing an overnight move no real stock made. Any symbol showing a
# single daily move beyond this is skipped with a warning rather than
# silently generating signals from broken data.
#
# CALIBRATED against a real 2,075-symbol run: at 35% this caught 74
# symbols, but among them were ADANIENT and CANBK — both at exactly 38.7%,
# from a genuine 2015 demerger and a 2017 bonus issue in liquid large caps.
# Only 15 of the 74 catches sat in the 35-45% band while 45 exceeded 60%
# (median 66%, max 4,350%), so 50% keeps essentially all the genuinely
# corrupt data while no longer discarding real large caps.
MAX_PLAUSIBLE_DAILY_MOVE_PCT = 50.0

OUTPUT_DIR = "signals"

# Volatility Stop / ATR
ATR_LEN = 14
ATR_MULT = 3.0

# Trend Filter — KAMA (Kaufman's Adaptive MA) replaces fixed EMA. KAMA
# automatically slows down in choppy conditions and speeds up when trending,
# instead of using one fixed smoothing speed for every market regime.
KAMA_FAST_LEN = 20                # replaces old EMA_FAST
KAMA_MID_LEN = 50                 # replaces old EMA_MID
KAMA_SLOW_LEN = 100                # replaces old EMA_SLOW
KAMA_FASTEST_SC = 2                # KAMA's internal fastest smoothing period
KAMA_SLOWEST_SC = 30               # KAMA's internal slowest smoothing period
REQUIRE_MA_STACK = False           # KAMA_fast > KAMA_mid > KAMA_slow

# RSI momentum filter
RSI_LEN = 14
RSI_BUY_LEVEL = 50
RSI_OVERBOUGHT = 78

# Volume confirmation
VOL_MA_LEN = 20
VOL_MULT_REQ = 0.7

# Signal staleness warning threshold (%).
#
# Signals are computed on COMPLETED bars only, so a Monthly signal can be
# up to ~30 days old by the time you see it (Weekly, up to ~7). If price
# has already moved this far from the signal-bar close, the entry you'd
# actually get differs materially from the one the signal identified, and
# the scan output marks StalePrice = True.
PRICE_DRIFT_WARN_PCT = 7.0

# ---------------------------------------------------------------------
# Pullback-entry reference thresholds
# ---------------------------------------------------------------------
#
# Derived from out-of-sample testing, not chosen arbitrarily. Entering on
# a pullback beat entering on strength by +7 to +12% across three hold
# periods in 2021+ and 2023+ data, with controls verified unbiased. The
# strength-based entry the scanner itself uses was significantly WORSE
# than a random nearby date over the same period.
#
# These are REFERENCE markers only — they do not filter or gate signals.
# They exist so you can see whether a stock is currently at a
# historically better entry point than its own signal implies.
PULLBACK_RSI_MAX = 45.0            # RSI below this = pulled back
PULLBACK_MIN_OFF_HIGH_PCT = -8.0   # at least 8% below the 52-week high

# On-Balance Volume (OBV) trend confirmation — checks that volume is
# actually flowing IN alongside the price rise (not just spiking on isolated
# days), catching cases where volume is quietly leaking out on down days.
USE_OBV_FILTER = True
OBV_MA_LEN = 10

# Relative Strength vs NIFTY 50 — requires the stock to be outperforming the
# index, not just drifting up with the broader market. Catches genuine
# leaders rather than passive followers.
USE_RELATIVE_STRENGTH_FILTER = True
RS_MA_LEN = 10

# Trend-strength filter (ADX) — avoids taking signals in choppy/sideways
# conditions, which is where most false VStop flips happen.
ADX_LEN = 14
ADX_THRESHOLD = 20                # require ADX above this to accept a BUY

# Market regime filter — only take a BUY on a stock if the NIFTY 50 index
# itself is in an uptrend on the same timeframe. Cuts down on buying
# individual stocks against the broader market current.
USE_MARKET_REGIME_FILTER = False
REGIME_INDEX_TICKER = "^NSEI"     # NIFTY 50 on Yahoo Finance
# IMPORTANT: When NIFTY Weekly VStop is DOWN, all Weekly confirmed BUYs
# are intentionally blocked by the market-regime filter. The new BUY
# diagnostic shows exactly how many candidates are stopped at this stage.

# Multi-timeframe confluence — a Weekly BUY only counts as high-quality if
# the Monthly trend (the "bigger picture") is also UP. Monthly signals are
# already the highest timeframe here, so they don't need a further filter.
REQUIRE_MONTHLY_CONFLUENCE_FOR_WEEKLY = True

# Anti-whipsaw buffer — requires price to close beyond the Volatility Stop
# by this percentage before a trend flip is registered, instead of any
# close beyond it. Cuts down on flip-then-immediately-flip-back whipsaws.
VSTOP_WHIPSAW_BUFFER_PCT = 0.5    # 0.5 = require a 0.5% buffer past the stop

# Earnings-week avoidance — many "false" technical breakouts are really
# pre-earnings volatility the chart can't distinguish from a real breakout.
# OFF by default: this adds one extra network call PER STOCK (via yfinance's
# earnings calendar), which meaningfully slows down a 1500-stock scan and is
# itself an unofficial/sometimes-missing data source. Turn on for smaller
# watchlists where the extra time is acceptable.
USE_EARNINGS_AVOIDANCE = False
EARNINGS_AVOID_DAYS = 5           # skip BUY if earnings due within N days

# Data
HISTORY_PERIOD = "15y"            # yfinance period to download.
                                  #
                                  # MUST exceed the MONTHLY warm-up:
                                  # compute_signals() needs KAMA_SLOW_LEN + 2
                                  # = 102 bars before returning anything, and
                                  # 102 MONTHLY bars is 8.5 years. The old
                                  # "10y" left only ~18 usable monthly bars,
                                  # so monthly backtests covered barely a year.
                                  #
                                  # 15y gives ~180 monthly bars (~78 usable,
                                  # i.e. 6.5 years of testable signals) while
                                  # staying much faster than "max": the
                                  # backtest re-runs the whole indicator stack
                                  # on a growing window for every bar, so
                                  # runtime scales roughly with the SQUARE of
                                  # history length. "max" made a full run take
                                  # ~6 hours and blow past the job timeout.
BATCH_SIZE = 40                   # tickers per yfinance batch download
BATCH_SLEEP_SEC = 3               # pause between batches (avoid rate limits)
EXCHANGE_SUFFIX = ".NS"           # ".NS" = NSE, ".BO" = BSE
CHECKPOINT_EVERY_N_BATCHES = 5    # autosave partial results periodically —
                                   # matters at 1500+ stocks since a full run
                                   # can take well over an hour and you don't
                                   # want to lose everything to one crash/
                                   # network drop

# Notifications
NOTIFY_EMAIL = True               # set to False to disable
NOTIFY_TELEGRAM = False

EMAIL_FROM = os.environ.get("SCANNER_EMAIL_FROM", "")
EMAIL_TO = os.environ.get("SCANNER_EMAIL_TO", "")
EMAIL_APP_PASSWORD = os.environ.get("SCANNER_EMAIL_APP_PASSWORD", "")  # Gmail app password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

TELEGRAM_BOT_TOKEN = os.environ.get("SCANNER_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("SCANNER_TELEGRAM_CHAT_ID", "")

# Email chart images — small inline PNG per BUY signal, kept deliberately
# tiny to keep the total email size small.
EMAIL_INCLUDE_CHARTS = True
EMAIL_MAX_CHARTS = 10              # cap embedded charts regardless of how
                                    # many BUY signals fire, so the email
                                    # can't balloon on a big signal day
EMAIL_CHART_WIDTH_IN = 3.6
EMAIL_CHART_HEIGHT_IN = 2.0
EMAIL_CHART_DPI = 80               # ~15-30 KB per chart at this size/DPI
EMAIL_CHART_LOOKBACK_WEEKLY = 60   # bars of history to show
EMAIL_CHART_LOOKBACK_MONTHLY = 36


# =========================================================================
# INDICATOR LOGIC (mirrors the Pine Script f_signal() function)
# =========================================================================
def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA smoothing — matches Pine Script's ta.atr / ta.rsi internals."""
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder_smooth(tr, length)


def compute_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_kama(series: pd.Series, er_length: int, fastest: int = 2, slowest: int = 30) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average. Unlike EMA (fixed smoothing speed),
    KAMA speeds up when the market is trending efficiently and slows down
    in choppy/sideways conditions, based on the Efficiency Ratio (ER).
    """
    change = series.diff(er_length).abs()
    volatility = series.diff().abs().rolling(er_length).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0)

    fast_sc = 2 / (fastest + 1)
    slow_sc = 2 / (slowest + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(np.nan, index=series.index)
    values = series.values
    sc_values = sc.values
    n = len(series)

    first_valid = er_length
    if first_valid >= n:
        return kama

    kama_vals = np.full(n, np.nan)
    kama_vals[first_valid] = values[first_valid]
    for i in range(first_valid + 1, n):
        prev = kama_vals[i - 1]
        if np.isnan(prev):
            kama_vals[i] = values[i]
        else:
            s = sc_values[i] if not np.isnan(sc_values[i]) else 0
            kama_vals[i] = prev + s * (values[i] - prev)

    return pd.Series(kama_vals, index=series.index)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume signed by the direction of price change."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()

def compute_adx(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Wilder's ADX — measures trend STRENGTH regardless of direction.
    Used as a filter: only accept a trend-following signal when ADX shows
    the stock is actually trending, not chopping sideways (VStop's most
    common source of false flips).
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = wilder_smooth(tr, length)
    plus_di = 100 * wilder_smooth(plus_dm, length) / atr.replace(0, np.nan)
    minus_di = 100 * wilder_smooth(minus_dm, length) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder_smooth(dx.fillna(0), length)
    return adx.fillna(0)


def compute_vstop(df: pd.DataFrame, atr_len: int, atr_mult: float, whipsaw_buffer_pct: float = 0.0) -> pd.DataFrame:
    """
    Recursive Volatility Stop — identical logic to the Pine Script version:
    trailing stop that only tightens with the trend and flips when price
    crosses it.

    whipsaw_buffer_pct: require price to close beyond the stop by this
    percentage before registering a flip, instead of any close beyond it.
    Reduces flip-then-flip-back whipsaws in choppy conditions. 0 = off
    (original behavior).
    """
    atr_val = atr_mult * compute_atr(df, atr_len)
    close = df["Close"].values
    atr_arr = atr_val.values
    buf = whipsaw_buffer_pct / 100.0

    n = len(df)
    stop = np.full(n, np.nan)
    uptrend = np.full(n, True)
    extreme = np.full(n, np.nan)
    flip_up = np.full(n, False)
    flip_down = np.full(n, False)

    trend = True
    ext = np.nan
    stp = np.nan

    for i in range(n):
        src = close[i]
        a = atr_arr[i]
        if np.isnan(a):
            stop[i], uptrend[i], extreme[i] = stp, trend, ext
            continue

        ext = src if np.isnan(ext) else (max(ext, src) if trend else min(ext, src))
        new_stop = (ext - a) if trend else (ext + a)

        stp = new_stop if np.isnan(stp) else (max(stp, new_stop) if trend else min(stp, new_stop))

        flip = (src < stp * (1 - buf)) if trend else (src > stp * (1 + buf))
        if flip:
            trend = not trend
            ext = src
            stp = (src - a) if trend else (src + a)
            flip_up[i] = trend
            flip_down[i] = not trend

        stop[i], uptrend[i], extreme[i] = stp, trend, ext

    out = df.copy()
    out["ATR_STOP"] = atr_val
    out["VSTOP"] = stop
    out["UPTREND"] = uptrend
    out["FLIP_UP"] = flip_up
    out["FLIP_DOWN"] = flip_down
    return out


def compute_signals(df: pd.DataFrame, nifty_close: pd.Series = None) -> pd.DataFrame:
    """
    Full signal stack: VStop (with anti-whipsaw buffer) + KAMA trend filter
    + RSI + Volume + OBV + ADX + Relative Strength vs NIFTY (if provided).
    """
    min_len = max(KAMA_SLOW_LEN, ATR_LEN, RSI_LEN, VOL_MA_LEN, ADX_LEN, KAMA_SLOWEST_SC) + 2
    if df.empty or len(df) < min_len:
        return pd.DataFrame()

    out = compute_vstop(df, ATR_LEN, ATR_MULT, VSTOP_WHIPSAW_BUFFER_PCT)

    out["KAMA_FAST"] = compute_kama(out["Close"], KAMA_FAST_LEN, KAMA_FASTEST_SC, KAMA_SLOWEST_SC)
    out["KAMA_MID"] = compute_kama(out["Close"], KAMA_MID_LEN, KAMA_FASTEST_SC, KAMA_SLOWEST_SC)
    out["KAMA_SLOW"] = compute_kama(out["Close"], KAMA_SLOW_LEN, KAMA_FASTEST_SC, KAMA_SLOWEST_SC)

    out["RSI"] = compute_rsi(out["Close"], RSI_LEN)
    out["VOL_MA"] = out["Volume"].rolling(VOL_MA_LEN).mean()
    out["ADX"] = compute_adx(out, ADX_LEN)

    out["OBV"] = compute_obv(out)
    out["OBV_MA"] = out["OBV"].rolling(OBV_MA_LEN).mean()

    if REQUIRE_MA_STACK:
        ma_trend_ok = (out["KAMA_FAST"] > out["KAMA_MID"]) & (out["KAMA_MID"] > out["KAMA_SLOW"])
    else:
        ma_trend_ok = out["Close"] > out["KAMA_MID"]

    rsi_ok = (out["RSI"] > RSI_BUY_LEVEL) & (out["RSI"] < RSI_OVERBOUGHT)
    vol_ok = out["Volume"] > (out["VOL_MA"] * VOL_MULT_REQ)
    adx_ok = out["ADX"] > ADX_THRESHOLD
    obv_ok = (out["OBV"] > out["OBV_MA"]) if USE_OBV_FILTER else pd.Series(True, index=out.index)

    if USE_RELATIVE_STRENGTH_FILTER and nifty_close is not None and not nifty_close.empty:
        # Align the daily NIFTY series onto this timeframe's bar dates.
        #
        # IMPORTANT: reindex(...).ffill() alone is NOT sufficient. Weekly
        # resampling produces Sunday-stamped bars, but NIFTY only has
        # weekday data, so a direct reindex matches ZERO dates and yields
        # all-NaN — which made rs_ok False on every bar and silently
        # prevented ANY weekly BUY signal from ever firing.
        #
        # Reindexing onto the union of both date sets first, forward-
        # filling there, and only then selecting this timeframe's dates
        # means each bar picks up the most recent prior trading day's
        # close, which is what the comparison actually intends.
        combined_index = nifty_close.index.union(out.index)
        nifty_aligned = nifty_close.reindex(combined_index).ffill().reindex(out.index)
        rs_line = out["Close"] / nifty_aligned.replace(0, np.nan)
        rs_ma = rs_line.rolling(RS_MA_LEN).mean()
        rs_ok = rs_line > rs_ma
        out["RS_LINE"] = rs_line
    else:
        rs_ok = pd.Series(True, index=out.index)
        out["RS_LINE"] = np.nan

    # Expose every BUY component so the scan can explain exactly why a
    # candidate was rejected. These are diagnostic columns only; the
    # actual BUY logic remains unchanged.
    out["BUY_FLIP_OK"] = out["FLIP_UP"]
    out["BUY_MA_OK"] = ma_trend_ok
    out["BUY_RSI_OK"] = rsi_ok
    out["BUY_VOL_OK"] = vol_ok
    out["BUY_ADX_OK"] = adx_ok
    out["BUY_OBV_OK"] = obv_ok
    out["BUY_RS_OK"] = rs_ok

    out["OBV_OK"] = obv_ok
    out["RS_OK"] = rs_ok

    # Core signal on THIS timeframe alone — regime and multi-timeframe
    # confluence are layered on afterward in scan(), since those need
    # data from other symbols/timeframes this function doesn't see.
    out["BUY_SIGNAL_CORE"] = (
        out["FLIP_UP"]
        & ma_trend_ok
        & rsi_ok
        & vol_ok
        & adx_ok
        & obv_ok
        & rs_ok
    )
    out["SELL_SIGNAL"] = out["FLIP_DOWN"]
    return out


# =========================================================================
# DATA FETCHING
# =========================================================================
def load_universe(path: str) -> pd.DataFrame:
    uni = pd.read_csv(path)
    uni.columns = [c.strip() for c in uni.columns]
    return uni


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a flat OHLCV dataframe from yfinance output.

    Recent yfinance versions can return a MultiIndex even for a single
    ticker, e.g. (Price, Ticker).  Pandas resample/agg expects ordinary
    column names such as Open/High/Low/Close/Volume, so flatten that output
    before any indicator calculation.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    required = {"Open", "High", "Low", "Close", "Volume"}

    if isinstance(out.columns, pd.MultiIndex):
        # Find the MultiIndex level that contains the OHLCV field names.
        chosen = None
        for level in range(out.columns.nlevels):
            vals = {str(v) for v in out.columns.get_level_values(level)}
            if required.issubset(vals):
                chosen = level
                break

        if chosen is not None:
            out.columns = out.columns.get_level_values(chosen)
        else:
            # Defensive fallback: retain the first level and let the caller
            # report a useful missing-column error if the provider changes.
            out.columns = out.columns.get_level_values(0)

    out.columns = [str(c).strip() for c in out.columns]
    return out


def resample(df_daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample daily OHLCV data into Weekly/Monthly bars.

    Only completed Weekly and Monthly candles are returned.
    """

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    missing = [c for c in agg if c not in df_daily.columns]

    if missing:
        raise KeyError(
            f"Missing required OHLCV columns: {missing}; "
            f"available={list(df_daily.columns)}"
        )

    df_daily = df_daily.sort_index()

    # Pandas 3.x compatibility
    if rule == "M":
        rule = "ME"

    r = (
        df_daily
        .resample(rule)
        .agg(agg)
        .dropna(how="any")
    )

    
    if r.empty:
        return r

    # ------------------------------------------------------------
    # COMPLETED CANDLE PROTECTION
    # ------------------------------------------------------------

    last_daily_date = pd.Timestamp(df_daily.index.max()).normalize()

    if rule == "W":
        # Current week runs Monday-Sunday.
        # If today is Tuesday 11-Aug-2026, the current
        # week ends on Sunday 16-Aug-2026 and is incomplete.
        current_week_start = last_daily_date.to_period("W-SUN").start_time

        completed_cutoff = (
            current_week_start - pd.Timedelta(days=1)
        )

        r = r[r.index <= completed_cutoff]

    elif rule in ("M", "ME"):
        # Current month is incomplete until month-end.
        # On 11-Aug-2026 the current month ends 31-Aug-2026.
        current_month_start = last_daily_date.to_period("M").start_time

        completed_cutoff = (
            current_month_start - pd.Timedelta(days=1)
        )

        r = r[r.index <= completed_cutoff]

    return r


def fetch_batch(tickers: list) -> dict:
    """Download daily OHLCV for a batch of tickers, return dict[ticker] -> DataFrame."""
    data = yf.download(
        tickers=tickers,
        period=HISTORY_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    result = {}
    if len(tickers) == 1:
        t = tickers[0]
        if not data.empty:
            result[t] = normalize_ohlcv_columns(data).dropna(how="all")
        return result

    for t in tickers:
        try:
            sub = data[t].dropna(how="all")
            if not sub.empty:
                result[t] = sub
        except (KeyError, Exception):
            continue
    return result


def fetch_single(ticker: str, retries: int = 2, sleep_between: float = 3.0):
    """
    Fetch one ticker on its own, with retries. Used to recover symbols that
    failed inside a batch download (often a transient rate-limit/network
    blip rather than a genuinely delisted symbol).
    """
    ticker = str(ticker).strip().upper()

    # Add NSE suffix if missing
    if "." not in ticker and ticker != REGIME_INDEX_TICKER:
        ticker = ticker + ".NS"
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=True,
                threads=False,
                progress=False,
            )
            df = normalize_ohlcv_columns(df)
            df = df.dropna(how="all")
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(sleep_between)
    return None


def fetch_nifty_daily():
    """Single download of NIFTY 50 daily data, reused for both the market
    regime filter and the relative-strength filter (avoids fetching twice)."""
    if not (USE_MARKET_REGIME_FILTER or USE_RELATIVE_STRENGTH_FILTER):
        return None
    print(f"Fetching {REGIME_INDEX_TICKER} (used for market regime + relative strength)...")
    try:
        df = yf.download(REGIME_INDEX_TICKER, period=HISTORY_PERIOD, interval="1d",
                          auto_adjust=True, progress=False)
        df = normalize_ohlcv_columns(df)
        df = df.dropna(how="all")
        if df.empty:
            print("  Could not fetch index data — regime/RS filters disabled for this run.")
            return None
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"NIFTY OHLCV columns missing: {missing}; available={list(df.columns)}")
        return df[required]
    except Exception as e:
        print(f"  Index fetch failed ({e}) — regime/RS filters disabled for this run.")
        return None


def compute_market_regime(nifty_daily) -> dict:
    """
    Determine whether NIFTY 50 is currently in an uptrend on Weekly and
    Monthly timeframes, using the exact same VStop logic applied to stocks.
    Returns {'Weekly': bool, 'Monthly': bool}, defaulting to True (i.e. no
    filtering) if index data is unavailable, so a data hiccup doesn't
    silently suppress every signal.
    """
    regime = {"Weekly": True, "Monthly": True}
    if not USE_MARKET_REGIME_FILTER or nifty_daily is None:
        return regime

    for tf_name, rule in (("Weekly", "W"), ("Monthly", "ME")):
        df_tf = resample(nifty_daily, rule)
        vstop_df = compute_vstop(df_tf, ATR_LEN, ATR_MULT, VSTOP_WHIPSAW_BUFFER_PCT)
        if not vstop_df.empty:
            regime[tf_name] = bool(vstop_df.iloc[-1]["UPTREND"])
    print(f"  NIFTY 50 regime — Weekly: {'UP' if regime['Weekly'] else 'DOWN'}, "
          f"Monthly: {'UP' if regime['Monthly'] else 'DOWN'}")
    return regime


def get_next_earnings_days(ticker_symbol: str):
    """
    Days until this stock's next earnings report, or None if unavailable.
    Uses yfinance's earnings calendar — an unofficial/best-effort source,
    so failures are treated as 'unknown' (doesn't block the BUY) rather
    than as a reason to skip.
    """
    try:
        t = yf.Ticker(ticker_symbol)
        edf = t.get_earnings_dates(limit=4)
        if edf is None or edf.empty:
            return None
        now = pd.Timestamp.now(tz=edf.index.tz) if edf.index.tz else pd.Timestamp.now()
        future = edf[edf.index >= now]
        if future.empty:
            return None
        next_date = future.index.min()
        return (next_date - now).days
    except Exception:
        return None


def make_chart_png(sig: pd.DataFrame, symbol: str, timeframe: str, lookback: int) -> bytes:
    """
    Small inline chart for a BUY signal: price, VStop, and the KAMA trend
    stack over the last `lookback` bars. Deliberately tiny (small figsize +
    low DPI) so embedding several of these in one email stays lightweight.
    """
    df = sig.tail(lookback)
    fig, ax = plt.subplots(figsize=(EMAIL_CHART_WIDTH_IN, EMAIL_CHART_HEIGHT_IN), dpi=EMAIL_CHART_DPI)

    ax.plot(df.index, df["Close"], color="#1a73e8", linewidth=1.1, label="Close")
    ax.plot(df.index, df["VSTOP"], color="#e37400", linewidth=0.9, linestyle="--", label="VStop")
    if "KAMA_MID" in df.columns:
        ax.plot(df.index, df["KAMA_MID"], color="#888888", linewidth=0.8, label="KAMA")

    # Mark the buy bar
    ax.scatter([df.index[-1]], [df["Close"].iloc[-1]], color="#188038", marker="^", s=35, zorder=5)

    ax.set_title(f"{symbol} — {timeframe}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="upper left", frameon=False)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout(pad=0.4)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=EMAIL_CHART_DPI)
    plt.close(fig)
    return buf.getvalue()


# =========================================================================
# BUY DIAGNOSTICS
# =========================================================================
def new_buy_diagnostic():
    """
    Sequential BUY funnel counters.

    The funnel is intentionally sequential:
      Universe -> VStop -> MA -> RSI -> Volume -> ADX -> OBV -> RS
      -> Core BUY -> NIFTY regime -> Monthly confluence -> Earnings -> Final BUY

    This makes it possible to see exactly where BUY candidates are being
    eliminated without changing the trading strategy.
    """
    return {
        "Universe": 0,
        "VStopFlipUp": 0,
        "MAStack": 0,
        "RSI": 0,
        "Volume": 0,
        "ADX": 0,
        "OBV": 0,
        "RelativeStrength": 0,
        "CoreBUY": 0,
        "NIFTYRegime": 0,
        "MonthlyConfluence": 0,
        "EarningsOK": 0,
        "FinalBUY": 0,
    }


def print_buy_diagnostics(diagnostics):
    print("\n" + "=" * 78)
    print("BUY REJECTION DIAGNOSTIC")
    print("=" * 78)
    print("Sequential funnel: each row is the number surviving that filter.")
    print("-" * 78)
    print(f"{'Filter':<28} {'Weekly':>12} {'Monthly':>12}")
    print("-" * 78)

    labels = [
        ("Universe", "Universe"),
        ("Fresh VStop FLIP_UP", "VStopFlipUp"),
        ("KAMA MA Stack", "MAStack"),
        ("RSI", "RSI"),
        ("Volume", "Volume"),
        ("ADX", "ADX"),
        ("OBV", "OBV"),
        ("Relative Strength", "RelativeStrength"),
        ("Core BUY", "CoreBUY"),
        ("NIFTY Regime", "NIFTYRegime"),
        ("Monthly Confluence", "MonthlyConfluence"),
        ("Earnings OK", "EarningsOK"),
        ("FINAL BUY", "FinalBUY"),
    ]

    for label, key in labels:
        print(
            f"{label:<28} "
            f"{diagnostics['Weekly'][key]:>12} "
            f"{diagnostics['Monthly'][key]:>12}"
        )

    print("-" * 78)
    print(
        "NIFTY regime: "
        f"Weekly={'UP' if diagnostics['_market_regime']['Weekly'] else 'DOWN'}, "
        f"Monthly={'UP' if diagnostics['_market_regime']['Monthly'] else 'DOWN'}"
    )
    print("=" * 78)


# =========================================================================
# SCAN
# =========================================================================
def scan() -> pd.DataFrame:
    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        before = len(uni)
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
        skipped = before - len(uni)
        if skipped:
            print(f"Skipping {skipped} 'Other' category symbols (set INCLUDE_OTHER_CATEGORY=True to include).")

    tickers = [f"{s.strip()}{EXCHANGE_SUFFIX}" for s in uni["Symbol"]]
    symbol_map = dict(zip(tickers, uni["Symbol"]))
    category_map = dict(zip(uni["Symbol"], uni["Category"]))
    name_map = dict(zip(uni["Symbol"], uni["Name"]))

    rows = []
    charts = {}  # (symbol, timeframe) -> PNG bytes, only for BUY signals
    failures = []  # (symbol, reason) for the end-of-run report
    total = len(tickers)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Scanning {total} symbols in {n_batches} batches of up to {BATCH_SIZE}...")
    run_start = time.time()

    nifty_daily = fetch_nifty_daily()
    market_regime = compute_market_regime(nifty_daily)

    diagnostics = {
        "Weekly": new_buy_diagnostic(),
        "Monthly": new_buy_diagnostic(),
        "_market_regime": market_regime.copy(),
    }

    bar_counts = {}

    for batch_num, i in enumerate(range(0, total, BATCH_SIZE), start=1):
        batch = tickers[i:i + BATCH_SIZE]
        elapsed = time.time() - run_start
        avg_per_batch = elapsed / (batch_num - 1) if batch_num > 1 else None
        eta_str = ""
        if avg_per_batch:
            remaining = avg_per_batch * (n_batches - batch_num + 1)
            eta_str = f" | ETA ~{remaining / 60:.1f} min"
        pct = (batch_num - 1) / n_batches * 100
        print(f"\n[Batch {batch_num}/{n_batches} | {pct:.0f}% done{eta_str}] "
              f"Fetching {i + 1}-{min(i + BATCH_SIZE, total)} of {total}")
        try:
            batch_data = fetch_batch(batch)
        except Exception as e:
            print(f"  Batch failed: {e}")
            batch_data = {}

        # Anything missing from the batch result gets a solo retry —
        # batch failures are frequently transient rate-limit blips, not
        # real delistings.
        missing = [t for t in batch if t not in batch_data]
        for tkr in missing:
            print(f"  Retrying {tkr} individually...")
            solo = fetch_single(tkr)
            if solo is not None and not solo.empty:
                batch_data[tkr] = solo
            else:
                symbol = symbol_map[tkr]
                failures.append((symbol, "No data after retry — check if ticker was renamed/delisted, "
                                          "or try the .BO (BSE) suffix instead of .NS"))

        for tkr, df_daily in batch_data.items():
            symbol = symbol_map[tkr]
            try:
                df_daily = df_daily[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if df_daily.empty:
                    continue

                # Remember how much usable history this symbol had, so the
                # failure report can say WHY a symbol was skipped and when it
                # will qualify, rather than just "insufficient history".
                try:
                    bar_counts[symbol] = len(resample(df_daily, "W"))
                except Exception:
                    pass

                # ----------------------------------------------------------
                # Data-quality and liquidity gates
                # ----------------------------------------------------------
                # These matter most when INCLUDE_OTHER_CATEGORY is True and
                # the universe expands to ~2,075 names, but they're applied
                # universally so behaviour is consistent either way.

                last_close = float(df_daily["Close"].iloc[-1])

                if last_close < MIN_PRICE:
                    failures.append((symbol, f"Below MIN_PRICE ({last_close:.2f} < {MIN_PRICE})"))
                    continue

                if MIN_AVG_TURNOVER > 0:
                    recent = df_daily.tail(20)
                    avg_turnover = float((recent["Close"] * recent["Volume"]).mean())
                    if avg_turnover < MIN_AVG_TURNOVER:
                        failures.append((
                            symbol,
                            f"Illiquid: avg turnover Rs {avg_turnover:,.0f}/day "
                            f"< Rs {MIN_AVG_TURNOVER:,.0f}"
                        ))
                        continue

                # Unadjusted-split / bad-tick detection. A genuine NSE
                # equity does not move 35%+ in one session outside of a
                # corporate action the data feed failed to adjust for.
                daily_moves = df_daily["Close"].pct_change().abs() * 100
                worst_move = float(daily_moves.max()) if len(daily_moves) else 0.0
                if worst_move >= MAX_PLAUSIBLE_DAILY_MOVE_PCT:
                    when = daily_moves.idxmax()
                    failures.append((
                        symbol,
                        f"Suspect price data: {worst_move:.1f}% single-day move on "
                        f"{pd.Timestamp(when).date()} — likely an unadjusted split/bonus"
                    ))
                    continue

                # Earnings avoidance check — one call per stock (not per
                # timeframe), only if enabled (adds real time at scale).
                earnings_days = None
                if USE_EARNINGS_AVOIDANCE:
                    earnings_days = get_next_earnings_days(tkr)
                earnings_soon = (earnings_days is not None
                                  and 0 <= earnings_days <= EARNINGS_AVOID_DAYS)

                # Compute Monthly FIRST — Weekly needs it for confluence.
                tf_signals = {}
                for tf_name, rule in (("Monthly", "ME"), ("Weekly", "W")):
                    df_tf = resample(df_daily, rule)
                    nifty_tf_close = None
                    if nifty_daily is not None and USE_RELATIVE_STRENGTH_FILTER:
                        nifty_tf = resample(nifty_daily, rule)
                        nifty_tf_close = nifty_tf["Close"]
                    sig = compute_signals(df_tf, nifty_close=nifty_tf_close)
                    tf_signals[tf_name] = sig

                monthly_sig = tf_signals["Monthly"]
                monthly_uptrend = bool(monthly_sig.iloc[-1]["UPTREND"]) if not monthly_sig.empty else True

                for tf_name in ("Weekly", "Monthly"):
                    sig = tf_signals[tf_name]
                    if sig.empty:
                        continue
                    last = sig.iloc[-1]

                    core_buy = bool(last["BUY_SIGNAL_CORE"])
                    regime_ok = market_regime.get(tf_name, True)

                    if tf_name == "Weekly" and REQUIRE_MONTHLY_CONFLUENCE_FOR_WEEKLY:
                        confluence_ok = monthly_uptrend
                    else:
                        confluence_ok = True

                    final_buy = core_buy and regime_ok and confluence_ok and not earnings_soon

                    # ---------------------------------------------------------
                    # BUY REJECTION DIAGNOSTIC
                    # ---------------------------------------------------------
                    # Count the current/latest bar only. The funnel is
                    # sequential, so later filters count only candidates that
                    # survived all previous filters.
                    d = diagnostics[tf_name]
                    d["Universe"] += 1

                    if bool(last["BUY_FLIP_OK"]):
                        d["VStopFlipUp"] += 1

                        if bool(last["BUY_MA_OK"]):
                            d["MAStack"] += 1

                            if bool(last["BUY_RSI_OK"]):
                                d["RSI"] += 1

                                if bool(last["BUY_VOL_OK"]):
                                    d["Volume"] += 1

                                    if bool(last["BUY_ADX_OK"]):
                                        d["ADX"] += 1

                                        if bool(last["BUY_OBV_OK"]):
                                            d["OBV"] += 1

                                            if bool(last["BUY_RS_OK"]):
                                                d["RelativeStrength"] += 1

                                                if core_buy:
                                                    d["CoreBUY"] += 1

                                                    if regime_ok:
                                                        d["NIFTYRegime"] += 1

                                                        if confluence_ok:
                                                            d["MonthlyConfluence"] += 1

                                                            if not earnings_soon:
                                                                d["EarningsOK"] += 1

                                                                if final_buy:
                                                                    d["FinalBUY"] += 1

                    if final_buy and EMAIL_INCLUDE_CHARTS:
                        try:
                            lookback = EMAIL_CHART_LOOKBACK_WEEKLY if tf_name == "Weekly" else EMAIL_CHART_LOOKBACK_MONTHLY
                            charts[(symbol, tf_name)] = make_chart_png(sig, symbol, tf_name, lookback)
                        except Exception as e:
                            print(f"  Chart generation failed for {symbol} ({tf_name}): {e}")

                    # Current price vs the signal-bar price.
                    #
                    # resample() deliberately drops the in-progress candle,
                    # so "Price" below is the close of the last COMPLETED
                    # bar — for a Monthly scan run mid-month that can be up
                    # to 30 days old. In live use that gap matters: on one
                    # real scan, 4 of 13 Monthly picks had already moved
                    # 10%+ since their signal bar (one +24.7%, another
                    # -18.4%), so the "recommendation" was priced off a
                    # thesis that had already changed.
                    #
                    # The signal itself is still computed from completed
                    # bars (correct — you shouldn't act on an unfinished
                    # candle). This just surfaces how stale it is.
                    try:
                        current_price = round(float(df_daily["Close"].iloc[-1]), 2)
                        signal_price = round(float(last["Close"]), 2)
                        price_drift = (
                            round((current_price - signal_price) / signal_price * 100, 2)
                            if signal_price else np.nan
                        )
                        stale = (
                            bool(abs(price_drift) >= PRICE_DRIFT_WARN_PCT)
                            if not np.isnan(price_drift) else False
                        )
                    except Exception:
                        current_price, price_drift, stale = np.nan, np.nan, False

                    # Pullback-entry reference values. Lookback is 52 bars
                    # on Weekly (one year) and 12 on Monthly, so both mean
                    # "52-week high" in calendar terms.
                    try:
                        lookback = 52 if tf_name.lower() == "weekly" else 12
                        recent_high = float(sig["Close"].iloc[-lookback:].max())
                        cur = float(last["Close"])
                        pct_from_high = (
                            round((cur - recent_high) / recent_high * 100, 2)
                            if recent_high else np.nan
                        )
                        kama_mid = float(last["KAMA_MID"]) if "KAMA_MID" in last.index else np.nan
                        rsi_val = float(last["RSI"]) if "RSI" in last.index else np.nan
                        pullback_zone = bool(
                            not np.isnan(pct_from_high)
                            and not np.isnan(kama_mid)
                            and not np.isnan(rsi_val)
                            and cur > kama_mid
                            and rsi_val < PULLBACK_RSI_MAX
                            and pct_from_high <= PULLBACK_MIN_OFF_HIGH_PCT
                        )
                    except Exception:
                        pct_from_high, pullback_zone = np.nan, False

                    rows.append({
                        "Symbol": symbol,
                        "Name": name_map.get(symbol, ""),
                        "Category": category_map.get(symbol, ""),
                        "Timeframe": tf_name,
                        "Date": sig.index[-1].date(),
                        "Price": round(float(last["Close"]), 2),
                        "CurrentPrice": current_price,
                        "PriceDriftPct": price_drift,
                        "StalePrice": stale,
                        "Volume": int(last["Volume"]),
                        "VolumeMA": float(last["VOL_MA"]) if not np.isnan(last["VOL_MA"]) else np.nan,
                        "VolumeRatio": (float(last["Volume"] / last["VOL_MA"])
                                       if not np.isnan(last["VOL_MA"]) and last["VOL_MA"] != 0
                                       else np.nan
                        ),
                        "VolAboveAvg": bool(last["Volume"] > last["VOL_MA"] * VOL_MULT_REQ) if not np.isnan(last["VOL_MA"]) else False,
                        "RSI": round(float(last["RSI"]), 1),
                        "ADX": round(float(last["ADX"]), 1),
                        "TrendStrengthOK": bool(last["ADX"] > ADX_THRESHOLD),
                        "OBV_OK": bool(last["OBV_OK"]) if USE_OBV_FILTER else "N/A",
                        "RelStrengthOK": bool(last["RS_OK"]) if USE_RELATIVE_STRENGTH_FILTER else "N/A",
                        "MarketRegimeOK": regime_ok,
                        "MonthlyConfluence": confluence_ok if tf_name == "Weekly" else "N/A",
                        "EarningsDaysAway": earnings_days if earnings_days is not None else "",
                        "EarningsSoon": earnings_soon,
                        "Trend": "UP" if last["UPTREND"] else "DOWN",
                        "VStop": round(float(last["VSTOP"]), 2),
                        "FLIP_UP": bool(last["FLIP_UP"]),
                        # Use the per-bar diagnostic columns populated by
                        # compute_signals().  The local filter variables are
                        # not in scope here (they are created in that
                        # function), which previously caused every symbol to
                        # be skipped while building the result row.
                        "MA_Trend_OK": bool(last["BUY_MA_OK"]),
                        "RSI_OK": bool(last["BUY_RSI_OK"]),
                        "VOL_OK": bool(last["BUY_VOL_OK"]),
                        "ADX_OK": bool(last["BUY_ADX_OK"]),
                        "OBV_Filter_OK": bool(last["BUY_OBV_OK"]),
                        "RelStrength_Filter_OK": bool(last["BUY_RS_OK"]),
                        "BUY_SIGNAL_CORE": bool(core_buy),

                        # BUY diagnostic fields
                        "BUY_FlipOK": bool(last["BUY_FLIP_OK"]),
                        "BUY_MA_OK": bool(last["BUY_MA_OK"]),
                        "BUY_RSI_OK": bool(last["BUY_RSI_OK"]),
                        "BUY_VolumeOK": bool(last["BUY_VOL_OK"]),
                        "BUY_ADX_OK": bool(last["BUY_ADX_OK"]),
                        "BUY_OBV_OK": bool(last["BUY_OBV_OK"]),
                        "BUY_RS_OK": bool(last["BUY_RS_OK"]),
                        "BUY_CoreOK": core_buy,

                        # Numeric relative-strength value (stock/NIFTY ratio
                        # line) — kept alongside the existing RelStrengthOK
                        # boolean so the live scan exposes the same feature
                        # the backtester trains on.
                        "RS_LINE": (round(float(last["RS_LINE"]), 4)
                                    if "RS_LINE" in last.index and not pd.isna(last["RS_LINE"])
                                    else ""),

                        # ------------------------------------------------
                        # Pullback-entry reference
                        # ------------------------------------------------
                        # Out-of-sample validation (2021+ and 2023+, with
                        # verified controls) found that entering a stock on
                        # a PULLBACK beat entering on strength by roughly
                        # +7 to +12%, while the strength-based entry was
                        # significantly WORSE than a random nearby date.
                        #
                        # These columns show, at a glance, whether a stock
                        # is currently in that pullback zone. They do NOT
                        # gate the signal — they're reference information
                        # for timing an entry you've already decided on.
                        #
                        # Zone = trend intact (above KAMA_MID)
                        #        AND RSI < 45
                        #        AND at least 8% below the 52-week high.
                        "PctFrom52WHigh": pct_from_high,
                        "PullbackZone": pullback_zone,

                        "Signal": "BUY" if final_buy else ("SELL" if last["SELL_SIGNAL"] else "-"),
                    })
            except Exception as e:
                print(f"  Skipping {tkr}: {e}")
                failures.append((symbol_map.get(tkr, tkr), str(e)))
                continue

        if batch_num % CHECKPOINT_EVERY_N_BATCHES == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "_checkpoint_partial.csv"), index=False)
            print(f"  (checkpoint saved: {len(rows)} rows so far)")

        time.sleep(BATCH_SLEEP_SEC)

    # Symbols that returned data but had too little history for the
    # indicators (e.g. recent IPOs) never reach compute_signals successfully
    # and won't appear in `rows` OR `failures` — catch those too.
    seen_symbols = {r["Symbol"] for r in rows}
    attempted_symbols = {symbol_map[t] for t in tickers}
    failed_symbols = {f[0] for f in failures}
    silent_gaps = attempted_symbols - seen_symbols - failed_symbols

    # These are almost always recent IPOs, not errors. compute_signals()
    # needs KAMA_SLOW_LEN + 2 bars before it returns anything — 102 bars,
    # which is 2 years of Weekly or 8.5 years of Monthly data. A stock
    # listed in 2025 simply cannot supply that yet.
    #
    # Lowering KAMA_SLOW_LEN to include them would mean computing a
    # "100-period trend" from 40 bars: a number that looks valid and means
    # nothing. Better to exclude them and say when they'll qualify.
    min_bars = KAMA_SLOW_LEN + 2
    for sym in silent_gaps:
        detail = ""
        try:
            wk = bar_counts.get(sym)
            if wk:
                weeks_short = max(0, min_bars - wk)
                eta = (f", qualifies in ~{weeks_short} weeks"
                       if 0 < weeks_short <= 260 else "")
                detail = f" (has {wk} weekly bars, needs {min_bars}{eta})"
        except Exception:
            pass
        failures.append((sym,
            f"Too little history for a {KAMA_SLOW_LEN}-period trend filter — "
            f"typically a recent listing{detail}"))

    if failures:
        print(f"\n{'=' * 70}\n{len(failures)} SYMBOL(S) COULD NOT BE SCANNED\n{'=' * 70}")
        for sym, reason in failures:
            print(f"  {sym}: {reason}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pd.DataFrame(failures, columns=["Symbol", "Reason"]).to_csv(
            os.path.join(OUTPUT_DIR, "failed_symbols.csv"), index=False)
        print(f"  (also saved to {OUTPUT_DIR}/failed_symbols.csv)")

    print_buy_diagnostics(diagnostics)

    # Persist the market regime so downstream tools (notably the
    # dashboard) can explain WHY a timeframe produced no BUY signals.
    #
    # When NIFTY's Weekly VStop is DOWN, every Weekly BUY is deliberately
    # blocked by the regime filter. Without this, a dashboard showing 13
    # Monthly BUYs and zero Weekly ones looks like a broken scanner
    # rather than a filter working exactly as intended.
    try:
        regime = diagnostics.get("_market_regime", {})
        with open(os.path.join(OUTPUT_DIR, "market_regime.json"), "w", encoding="utf-8") as f:
            json.dump({
                "Weekly": "UP" if regime.get("Weekly") else "DOWN",
                "Monthly": "UP" if regime.get("Monthly") else "DOWN",
                "weekly_buys_blocked": (not regime.get("Weekly")) and USE_MARKET_REGIME_FILTER,
                "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }, f, indent=2)
    except Exception as e:
        print(f"  (could not save market_regime.json: {e})")

    return pd.DataFrame(rows), charts


# =========================================================================
# NOTIFICATIONS
# =========================================================================
def send_email(subject: str, body: str):
    """Plain-text fallback (used if HTML/chart email isn't wanted)."""
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_APP_PASSWORD):
        print("Email not configured — skipping (set SCANNER_EMAIL_* env vars).")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
    print("Email sent (plain text).")


def send_email_with_charts(subject: str, html_body: str, charts: dict):
    """
    HTML email with small inline chart images. `charts` is
    {cid_string: png_bytes}. Reports the final message size so you can see
    exactly how "small" it ended up.
    """
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_APP_PASSWORD):
        print("Email not configured — skipping (set SCANNER_EMAIL_* env vars).")
        return

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html_body, "html"))

    for cid, png_bytes in charts.items():
        img = MIMEImage(png_bytes, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)

    raw = msg.as_string()
    size_kb = len(raw.encode("utf-8")) / 1024
    print(f"Email size: {size_kb:.0f} KB ({len(charts)} chart(s) embedded)")

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], raw)
    print("Email sent (HTML with charts).")


def send_telegram(text: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        print("Telegram not configured — skipping (set SCANNER_TELEGRAM_* env vars).")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})
    print("Telegram message sent.")


def build_html_email(buys: pd.DataFrame, sells: pd.DataFrame, charts: dict, failures_count: int):
    """
    Builds a compact HTML email: a small results table plus up to
    EMAIL_MAX_CHARTS inline chart images for the highest-ADX BUY signals.
    Returns (html_string, {cid: png_bytes}) — the cid dict is only the
    charts actually being embedded (capped), not all charts generated.
    """
    # Prioritize by ADX (strongest trend) when there are more BUYs than
    # the embed cap allows.
    buys_sorted = buys.sort_values("ADX", ascending=False)
    to_embed = []
    for _, r in buys_sorted.iterrows():
        key = (r["Symbol"], r["Timeframe"])
        if key in charts and len(to_embed) < EMAIL_MAX_CHARTS:
            to_embed.append(r)

    embedded_charts = {}

    style = ("font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#202124;"
             "border-collapse:collapse;width:100%;")
    th = "text-align:left;padding:4px 8px;background:#f1f3f4;font-size:12px;"
    td = "padding:4px 8px;border-top:1px solid #eee;font-size:12px;"

    html = ["<div style='font-family:Arial,Helvetica,sans-serif;'>"]
    html.append(f"<h3 style='margin:0 0 8px 0;'>NSE Scanner — {datetime.now().strftime('%d-%b-%Y %H:%M')}</h3>")

    # --- BUY table ---
    html.append(f"<p style='margin:12px 0 4px 0;font-weight:bold;color:#188038;'>BUY signals ({len(buys)})</p>")
    if buys.empty:
        html.append("<p style='margin:0;'>None today.</p>")
    else:
        html.append(f"<table style='{style}'><tr>"
                     f"<th style='{th}'>Symbol</th><th style='{th}'>Cat</th>"
                     f"<th style='{th}'>TF</th><th style='{th}'>Price</th>"
                     f"<th style='{th}'>RSI</th><th style='{th}'>ADX</th></tr>")
        for _, r in buys_sorted.iterrows():
            html.append(f"<tr><td style='{td}'>{r['Symbol']}</td><td style='{td}'>{r['Category']}</td>"
                         f"<td style='{td}'>{r['Timeframe']}</td><td style='{td}'>{r['Price']}</td>"
                         f"<td style='{td}'>{r['RSI']}</td><td style='{td}'>{r['ADX']}</td></tr>")
        html.append("</table>")

    # --- Inline charts for the top signals ---
    if to_embed:
        html.append(f"<p style='margin:16px 0 4px 0;font-weight:bold;'>Charts (top {len(to_embed)} by ADX)</p>")
        html.append("<div>")
        for i, r in enumerate(to_embed):
            cid = f"chart{i}"
            embedded_charts[cid] = charts[(r["Symbol"], r["Timeframe"])]
            html.append(f"<img src='cid:{cid}' alt='{r['Symbol']} {r['Timeframe']}' "
                        f"style='display:block;margin:4px 0;max-width:100%;'/>")
        html.append("</div>")
        skipped = len(buys) - len(to_embed)
        if skipped > 0:
            html.append(f"<p style='font-size:11px;color:#666;margin:4px 0;'>"
                        f"+{skipped} more BUY signal(s) not charted here — see the full CSV.</p>")

    # --- SELL table ---
    html.append(f"<p style='margin:16px 0 4px 0;font-weight:bold;color:#c5221f;'>SELL / trend-break ({len(sells)})</p>")
    if sells.empty:
        html.append("<p style='margin:0;'>None today.</p>")
    else:
        html.append(f"<table style='{style}'><tr>"
                     f"<th style='{th}'>Symbol</th><th style='{th}'>Cat</th>"
                     f"<th style='{th}'>TF</th><th style='{th}'>Price</th>"
                     f"<th style='{th}'>RSI</th></tr>")
        for _, r in sells.iterrows():
            html.append(f"<tr><td style='{td}'>{r['Symbol']}</td><td style='{td}'>{r['Category']}</td>"
                         f"<td style='{td}'>{r['Timeframe']}</td><td style='{td}'>{r['Price']}</td>"
                         f"<td style='{td}'>{r['RSI']}</td></tr>")
        html.append("</table>")

    if failures_count:
        html.append(f"<p style='font-size:11px;color:#666;margin-top:12px;'>"
                    f"{failures_count} symbol(s) could not be scanned — see failed_symbols.csv</p>")

    html.append("<p style='font-size:11px;color:#999;margin-top:16px;'>"
                "Not financial advice. Verify against your own chart before acting.</p>")
    html.append("</div>")

    return "".join(html), embedded_charts


# =========================================================================
# MAIN
# =========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== NSE Scanner run started: {datetime.now()} ===")

    results, charts = scan()
    if results.empty:
        print("No results — check your universe CSV and internet connection.")
        return

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(OUTPUT_DIR, f"scan_{ts}.csv")
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")

    buys = results[results["Signal"] == "BUY"].sort_values(["Timeframe", "Category"])
    sells = results[results["Signal"] == "SELL"].sort_values(["Timeframe", "Category"])

    failures_path = os.path.join(OUTPUT_DIR, "failed_symbols.csv")
    failures_count = 0
    if os.path.exists(failures_path):
        try:
            failures_count = len(pd.read_csv(failures_path))
        except Exception:
            pass

    print(f"\n{'=' * 70}\nBUY SIGNALS ({len(buys)})\n{'=' * 70}")
    if not buys.empty:
        print(buys[["Symbol", "Category", "Timeframe", "Price", "RSI", "ADX", "VolAboveAvg"]].to_string(index=False))
    else:
        print("None today.")

    print(f"\n{'=' * 70}\nSELL / TREND-BREAK SIGNALS ({len(sells)})\n{'=' * 70}")
    if not sells.empty:
        print(sells[["Symbol", "Category", "Timeframe", "Price", "RSI"]].to_string(index=False))
    else:
        print("None today.")

    if NOTIFY_EMAIL or NOTIFY_TELEGRAM:
        # Plain-text version, used for Telegram and as the email fallback
        # if charts are disabled/unavailable.
        body_lines = [f"NSE Scanner — {datetime.now().strftime('%d-%b-%Y %H:%M')}", ""]
        if not buys.empty:
            body_lines.append(f"BUY ({len(buys)}):")
            for _, r in buys.iterrows():
                body_lines.append(f"  {r['Symbol']} [{r['Category']}/{r['Timeframe']}] @ {r['Price']} RSI {r['RSI']}")
        else:
            body_lines.append("BUY: none today.")

        if not sells.empty:
            body_lines.append(f"\nSELL ({len(sells)}):")
            for _, r in sells.iterrows():
                body_lines.append(f"  {r['Symbol']} [{r['Category']}/{r['Timeframe']}] @ {r['Price']} RSI {r['RSI']}")
        else:
            body_lines.append("\nSELL: none today.")

        if failures_count:
            body_lines.append(f"\n({failures_count} symbol(s) could not be scanned — see failed_symbols.csv)")

        body_lines.append("\n— Not financial advice. Verify against your own chart before acting.")
        body = "\n".join(body_lines)

        if NOTIFY_EMAIL:
            subject = f"NSE Scanner: {len(buys)} BUY / {len(sells)} SELL — {datetime.now().strftime('%d-%b-%Y')}"
            if EMAIL_INCLUDE_CHARTS and charts:
                html, embedded_charts = build_html_email(buys, sells, charts, failures_count)
                send_email_with_charts(subject, html, embedded_charts)
            else:
                send_email(subject, body)
        if NOTIFY_TELEGRAM:
            send_telegram(body)

    print(f"\n=== Run finished: {datetime.now()} ===")


if __name__ == "__main__":
    main()


# =========================================================================
# SCHEDULING — how to run this automatically every day
# =========================================================================
# OPTION A (recommended): OS-level scheduler — runs once a day, no process
# needs to stay alive in the background.
#
#   Linux / macOS (cron), run daily at 18:00 (after market close, IST):
#     1. crontab -e
#     2. Add this line (edit paths to match your setup):
#        0 18 * * 1-5 cd /path/to/nse_scanner && /usr/bin/python3 nse_scanner.py >> run.log 2>&1
#        (the "1-5" restricts it to Mon–Fri)
#
#   Windows (Task Scheduler):
#     1. Open Task Scheduler -> Create Basic Task
#     2. Trigger: Daily, 18:00
#     3. Action: Start a program
#        Program: python.exe
#        Arguments: nse_scanner.py
#        Start in: C:\path\to\nse_scanner
#
# OPTION B: keep a Python process running continuously and let it sleep
# until the scheduled time each day. Use this only if you can't set up
# cron/Task Scheduler (e.g. sharing one long-running server). See
# daily_runner.py in this same folder for a ready-made version of this.
