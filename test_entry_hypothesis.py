"""
test_entry_hypothesis.py
==========================
Generates entry signals from ALTERNATIVE rules and benchmarks each one
against the same date-matched controls and bootstrap significance test
used by benchmark_signals.py.

BACKGROUND — why this exists
-----------------------------
The original strategy (VStop flip + KAMA + RSI>50 + volume surge + ADX +
OBV + relative strength) was tested against proper controls and found to
have NO detectable edge. At a 20-bar hold it was significantly WORSE than
entering the same stock on a nearby random date (-4.78%, CI [-9.15,
-0.42], excludes zero).

There is a plausible mechanism: every one of those conditions describes a
stock that has ALREADY run. Monthly signals fired an average of just
-0.2% from the 52-week high. That is a late-stage entry.

So the hypotheses below mostly test the OPPOSITE idea — buying temporary
weakness inside an intact uptrend — plus a couple of controls to make
sure any result isn't just an artifact.

HOW TO AVOID FOOLING YOURSELF
------------------------------
Testing many hypotheses guarantees some will look good by chance. Before
reading the output, fix the bar for success:

    An entry rule is only interesting if its confidence interval
    EXCLUDES ZERO on at least two of the three hold periods, against
    BOTH controls.

One good-looking cell out of many is noise, not a discovery. Resist
tweaking parameters until something passes — that is how backtests get
overfitted.

USAGE
-----
    python test_entry_hypothesis.py

Uses the same 500-stock universe as the scanner. Takes a while on the
first run (downloads price history), then reuses it across hypotheses.
"""

import os

import numpy as np
import pandas as pd

from backtest import config as cfg
from backtest.utils import pct_return
from nse_scanner import (
    fetch_single,
    fetch_nifty_daily,
    compute_signals,
    load_universe,
    resample,
    UNIVERSE_CSV,
    INCLUDE_OTHER_CATEGORY,
)

SIGNALS_DIR = "signals"
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "entry_hypothesis_results.csv")

HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 3
DATE_WINDOW_BARS = 26
BOOTSTRAP_ITERS = 2000
TIMEFRAME = "Weekly"          # Weekly has the most signals to work with
RESAMPLE_RULE = "W"

# Cap symbols for a faster first pass; set to None for the full universe.
MAX_SYMBOLS = None

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------
# ENTRY HYPOTHESES
#
# Each takes the computed signal frame and returns a boolean Series:
# True on bars where that rule would enter.
# ---------------------------------------------------------------------

def h_original_style(df):
    """Roughly the current strategy: buy confirmed strength."""
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (df["RSI"] > 50) & (df["RSI"] < 78)
        & (df["ADX"] > 20)
        & (df["Volume"] > df["VOL_MA"])
    )


def h_pullback_in_uptrend(df):
    """
    THE MAIN HYPOTHESIS.

    Long-term trend intact (price above the 50-period KAMA), but
    currently pulled back: RSI below 45 and price at least 8% off its
    52-week high. The direct opposite of buying extension.

    Thresholds are deliberately looser than a "textbook" pullback —
    tighter versions (RSI<35, -20% off high, price above the 100-period
    KAMA) produced almost no entries, and a rule that fires 5 times is
    untestable regardless of whether it works.
    """
    high_52 = df["Close"].rolling(52, min_periods=20).max()
    off_high = (df["Close"] - high_52) / high_52 * 100
    return (
        (df["Close"] > df["KAMA_MID"])
        & (df["RSI"] < 45)
        & (off_high < -8)
    )


def h_deep_pullback(df):
    """Same idea, deeper and slower-trend: RSI below 40, 15%+ off high."""
    high_52 = df["Close"].rolling(52, min_periods=20).max()
    off_high = (df["Close"] - high_52) / high_52 * 100
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (df["RSI"] < 40)
        & (off_high < -15)
    )


def h_pullback_with_reversal(df):
    """Pullback plus a first sign of turning back up."""
    high_52 = df["Close"].rolling(52, min_periods=20).max()
    off_high = (df["Close"] - high_52) / high_52 * 100
    rsi_turning = (df["RSI"] > df["RSI"].shift(1)) & (df["RSI"].shift(1) < 45)
    return (
        (df["Close"] > df["KAMA_MID"])
        & (off_high < -8)
        & rsi_turning
    )


def h_low_volatility_base(df):
    """
    Different angle entirely: buy quiet consolidation in an uptrend,
    on the theory that low volatility often precedes expansion.
    """
    atr_pct = df["ATR_STOP"] / df["Close"] * 100
    atr_rank = atr_pct.rolling(52, min_periods=20).rank(pct=True)
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (atr_rank < 0.25)
    )


def h_random_control(df):
    """
    Sanity check: fires at random on ~2% of bars. Should show NO edge.
    If this ever looks significant, the test harness is broken.
    """
    return pd.Series(
        RNG.random(len(df)) < 0.02,
        index=df.index,
    )


HYPOTHESES = [
    ("original_style_strength", h_original_style),
    ("pullback_in_uptrend", h_pullback_in_uptrend),
    ("deep_pullback", h_deep_pullback),
    ("pullback_with_reversal", h_pullback_with_reversal),
    ("low_volatility_base", h_low_volatility_base),
    ("random_control", h_random_control),
]


# ---------------------------------------------------------------------

def bootstrap_diff(a, b, iters=BOOTSTRAP_ITERS):
    a = np.asarray(pd.Series(a).dropna())
    b = np.asarray(pd.Series(b).dropna())
    if len(a) < 30 or len(b) < 30:
        return None
    diffs = np.empty(iters)
    for i in range(iters):
        diffs[i] = (
            RNG.choice(a, len(a), replace=True).mean()
            - RNG.choice(b, len(b), replace=True).mean()
        )
    return {
        "mean_diff": round(float(a.mean() - b.mean()), 2),
        "ci_low": round(float(np.percentile(diffs, 2.5)), 2),
        "ci_high": round(float(np.percentile(diffs, 97.5)), 2),
    }


def window_return(bars, pos, hold):
    if pos < 0 or pos + hold >= len(bars):
        return None
    entry = float(bars.iloc[pos]["Open"])
    exit_ = float(bars.iloc[pos + hold]["Close"])
    if entry <= 0:
        return None
    return pct_return(entry, exit_) - cfg.ROUND_TRIP_COST_PCT


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
    symbols = uni["Symbol"].dropna().astype(str).str.strip().tolist()
    if MAX_SYMBOLS:
        symbols = symbols[:MAX_SYMBOLS]
    print(f"Universe: {len(symbols)} symbols")

    print("Fetching NIFTY...")
    nd = fetch_nifty_daily()
    nifty_close = nd["Close"] if nd is not None else None

    print("Building signal frames (downloaded once, reused for every hypothesis)...")
    frames = {}
    for n, sym in enumerate(symbols, start=1):
        if n % 50 == 0 or n == 1:
            print(f"  [{n}/{len(symbols)}]", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is None or len(raw) == 0:
                continue
            bars = resample(raw.sort_index(), RESAMPLE_RULE)
            sig = compute_signals(bars, nifty_close)
            if sig is not None and not sig.empty:
                frames[sym] = sig
        except Exception:
            continue
    print(f"Ready: {len(frames)} symbols\n")

    cached = sorted(frames.keys())
    rows = []

    for name, rule in HYPOTHESES:
        print(f"=== {name} ===")

        # Collect entry positions per symbol, keeping only NEW entries
        # (a fresh transition into the condition, not every qualifying bar).
        entries = {}
        total = 0
        for sym, df in frames.items():
            try:
                mask = rule(df).fillna(False)
            except Exception:
                continue
            new = mask & ~mask.shift(1, fill_value=False)
            pos = np.flatnonzero(new.values)
            if len(pos):
                entries[sym] = pos
                total += len(pos)

        print(f"  entries: {total}")
        if total < 100:
            print("  too few entries to evaluate — skipping\n")
            continue

        for hold in HOLD_PERIODS:
            sig_r, same_date_r, same_stock_r = [], [], []

            for sym, positions in entries.items():
                bars = frames[sym]
                for p in positions:
                    r = window_return(bars, p + 1, hold)
                    if r is None:
                        continue
                    sig_r.append(r)
                    entry_date = bars.index[p]

                    for _ in range(CONTROLS_PER_SIGNAL):
                        other = cached[int(RNG.integers(0, len(cached)))]
                        ob = frames[other]
                        opos = ob.index.searchsorted(entry_date, side="right")
                        rr = window_return(ob, opos, hold)
                        if rr is not None:
                            same_date_r.append(rr)

                    for _ in range(CONTROLS_PER_SIGNAL):
                        off = int(RNG.integers(-DATE_WINDOW_BARS, DATE_WINDOW_BARS + 1))
                        rr = window_return(bars, p + 1 + off, hold)
                        if rr is not None:
                            same_stock_r.append(rr)

            if len(sig_r) < 30:
                continue

            row = {
                "hypothesis": name,
                "hold_bars": hold,
                "n": len(sig_r),
                "win_rate": round(float((pd.Series(sig_r) > 0).mean() * 100), 1),
                "mean_return": round(float(np.mean(sig_r)), 2),
                "median_return": round(float(np.median(sig_r)), 2),
            }
            for label, ctrl in [("vs_stock_same_date", same_date_r),
                                ("vs_same_stock_nearby", same_stock_r)]:
                bs = bootstrap_diff(sig_r, ctrl)
                if bs:
                    row[f"{label}_diff"] = bs["mean_diff"]
                    row[f"{label}_ci"] = f"[{bs['ci_low']}, {bs['ci_high']}]"
                    row[f"{label}_sig"] = (
                        "BETTER" if bs["ci_low"] > 0
                        else "WORSE" if bs["ci_high"] < 0
                        else "none"
                    )
            rows.append(row)
            print(f"  hold={hold:>3d}  n={row['n']:>5d}  mean={row['mean_return']:>+7.2f}%  "
                  f"vs_date={row.get('vs_stock_same_date_sig','-'):<7s} "
                  f"vs_stock={row.get('vs_same_stock_nearby_sig','-')}")
        print()

    if not rows:
        print("No hypothesis produced enough entries to evaluate.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print("=" * 100)
    print("  ENTRY HYPOTHESIS RESULTS")
    print("=" * 100)
    print(out.to_string(index=False))
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()
    print("SUCCESS BAR, set before you looked: a rule is only interesting if")
    print("it shows BETTER on at least two of the three hold periods against")
    print("BOTH controls. Anything less is consistent with chance.")
    print()
    print("Check random_control first — it SHOULD show 'none' everywhere.")
    print("If it doesn't, distrust the whole table rather than the control.")


if __name__ == "__main__":
    main()
