"""
validate_out_of_sample.py
===========================
The decisive test. Splits history into two eras, and checks whether the
pullback edge found in the first era still holds in the second — data
that played no part in generating the hypothesis.

WHY THIS IS THE TEST THAT MATTERS
----------------------------------
test_entry_hypothesis.py found that `pullback_in_uptrend` beat its
control by +5 to +6% across three hold periods, all confidence intervals
excluding zero. Encouraging — but that hypothesis was formulated AFTER
looking at why the original strategy failed, and it was one of 18 tests
run on the same data. Both of those are ways to find patterns that exist
only in the sample you looked at.

An out-of-sample test cannot be gamed the same way. Fit on 2012-2020,
verify on 2021-2026. If the edge survives untouched data, it is probably
real. If it evaporates, it was noise dressed up as a finding.

CONTROL BUG FIXED HERE
-----------------------
The previous same-date control was broken, and the random_control row
proved it: random entries came out significantly "WORSE" than random,
which is impossible. Cause — when a control stock had not listed yet at
the signal date, pandas' searchsorted() returned position 0, so instead
of skipping that stock the test entered at its FIRST EVER bar. That
systematically over-sampled newly listed stocks at their IPO period and
produced a constant ~-3.4% bias.

Fix: require the control symbol to have real data both BEFORE and far
enough AFTER the entry date, and skip it otherwise. The random_control
row below re-checks this: it must read "none" everywhere for the rest of
the table to be trustworthy.

USAGE
-----
    python validate_out_of_sample.py
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
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "out_of_sample_results.csv")

SPLIT_DATE = pd.Timestamp("2021-01-01")   # in-sample before, out-of-sample after

HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 4
DATE_WINDOW_BARS = 26
BOOTSTRAP_ITERS = 2000
RESAMPLE_RULE = "W"

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------
# Hypotheses carried over. Parameters are FROZEN — deliberately not
# retuned, since retuning on the out-of-sample era would destroy the
# entire point of holding it back.
# ---------------------------------------------------------------------

def h_original_style(df):
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (df["RSI"] > 50) & (df["RSI"] < 78)
        & (df["ADX"] > 20)
        & (df["Volume"] > df["VOL_MA"])
    )


def h_pullback_in_uptrend(df):
    high_52 = df["Close"].rolling(52, min_periods=20).max()
    off_high = (df["Close"] - high_52) / high_52 * 100
    return (
        (df["Close"] > df["KAMA_MID"])
        & (df["RSI"] < 45)
        & (off_high < -8)
    )


def h_deep_pullback(df):
    high_52 = df["Close"].rolling(52, min_periods=20).max()
    off_high = (df["Close"] - high_52) / high_52 * 100
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (df["RSI"] < 40)
        & (off_high < -15)
    )


def h_random_control(df):
    """Must show 'none' everywhere, or the harness is still broken."""
    return pd.Series(RNG.random(len(df)) < 0.02, index=df.index)


HYPOTHESES = [
    ("original_style_strength", h_original_style),
    ("pullback_in_uptrend", h_pullback_in_uptrend),
    ("deep_pullback", h_deep_pullback),
    ("random_control", h_random_control),
]


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
        "diff": round(float(a.mean() - b.mean()), 2),
        "lo": round(float(np.percentile(diffs, 2.5)), 2),
        "hi": round(float(np.percentile(diffs, 97.5)), 2),
    }


def window_return(bars, pos, hold):
    if pos < 0 or pos + hold >= len(bars):
        return None
    entry = float(bars.iloc[pos]["Open"])
    exit_ = float(bars.iloc[pos + hold]["Close"])
    if entry <= 0:
        return None
    return pct_return(entry, exit_) - cfg.ROUND_TRIP_COST_PCT


def same_date_control(ob, entry_date, hold):
    """
    Enter a DIFFERENT stock on the same date.

    Returns None unless that stock genuinely traded at the time. The old
    version skipped this check, so a not-yet-listed stock silently
    entered at its first ever bar — biasing the control toward IPO
    periods.
    """
    if len(ob) == 0:
        return None
    if entry_date < ob.index[0] or entry_date > ob.index[-1]:
        return None
    pos = ob.index.searchsorted(entry_date, side="right")
    if pos <= 0 or pos + hold >= len(ob):
        return None
    return window_return(ob, pos, hold)


def evaluate(frames, cached, rule, era_mask_fn, hold):
    """Collect signal + control returns for one hypothesis, one era."""
    sig_r, date_r, stock_r = [], [], []

    for sym, df in frames.items():
        try:
            mask = rule(df).fillna(False)
        except Exception:
            continue
        new = mask & ~mask.shift(1, fill_value=False)
        pos_arr = np.flatnonzero(new.values)

        for p in pos_arr:
            entry_date = df.index[p]
            if not era_mask_fn(entry_date):
                continue

            r = window_return(df, p + 1, hold)
            if r is None:
                continue
            sig_r.append(r)

            for _ in range(CONTROLS_PER_SIGNAL):
                other = cached[int(RNG.integers(0, len(cached)))]
                rr = same_date_control(frames[other], entry_date, hold)
                if rr is not None:
                    date_r.append(rr)

            for _ in range(CONTROLS_PER_SIGNAL):
                off = int(RNG.integers(-DATE_WINDOW_BARS, DATE_WINDOW_BARS + 1))
                rr = window_return(df, p + 1 + off, hold)
                if rr is not None:
                    stock_r.append(rr)

    return sig_r, date_r, stock_r


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
    symbols = uni["Symbol"].dropna().astype(str).str.strip().tolist()
    print(f"Universe: {len(symbols)} symbols")

    print("Fetching NIFTY...")
    nd = fetch_nifty_daily()
    nifty_close = nd["Close"] if nd is not None else None

    print("Building signal frames...")
    frames = {}
    for n, sym in enumerate(symbols, start=1):
        if n % 50 == 0 or n == 1:
            print(f"  [{n}/{len(symbols)}]", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is None or len(raw) == 0:
                continue
            sig = compute_signals(resample(raw.sort_index(), RESAMPLE_RULE), nifty_close)
            if sig is not None and not sig.empty:
                frames[sym] = sig
        except Exception:
            continue
    print(f"Ready: {len(frames)} symbols\n")

    cached = sorted(frames.keys())
    eras = [
        ("IN_SAMPLE(pre-2021)", lambda d: d < SPLIT_DATE),
        ("OUT_OF_SAMPLE(2021+)", lambda d: d >= SPLIT_DATE),
    ]

    rows = []
    for name, rule in HYPOTHESES:
        print(f"=== {name} ===")
        for era_name, era_fn in eras:
            for hold in HOLD_PERIODS:
                sig_r, date_r, stock_r = evaluate(frames, cached, rule, era_fn, hold)
                if len(sig_r) < 30:
                    continue

                row = {
                    "hypothesis": name,
                    "era": era_name,
                    "hold": hold,
                    "n": len(sig_r),
                    "win_rate": round(float((pd.Series(sig_r) > 0).mean() * 100), 1),
                    "mean_return": round(float(np.mean(sig_r)), 2),
                }
                for label, ctrl in [("vs_date", date_r), ("vs_stock", stock_r)]:
                    bs = bootstrap_diff(sig_r, ctrl)
                    if bs:
                        row[f"{label}_diff"] = bs["diff"]
                        row[f"{label}_ci"] = f"[{bs['lo']}, {bs['hi']}]"
                        row[f"{label}_sig"] = (
                            "BETTER" if bs["lo"] > 0
                            else "WORSE" if bs["hi"] < 0
                            else "none"
                        )
                rows.append(row)
                print(f"  {era_name:<22s} hold={hold:>3d} n={row['n']:>5d} "
                      f"mean={row['mean_return']:>+7.2f}%  "
                      f"vs_date={row.get('vs_date_sig','-'):<7s} "
                      f"vs_stock={row.get('vs_stock_sig','-')}")
        print()

    if not rows:
        print("Not enough data to evaluate.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print("=" * 100)
    print("  OUT-OF-SAMPLE VALIDATION")
    print("=" * 100)
    print(out.to_string(index=False))
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()
    print("HOW TO READ THIS")
    print("-" * 100)
    print("1. Check random_control FIRST. It must read 'none' in both eras")
    print("   against both controls. If it doesn't, the harness is still")
    print("   biased and nothing else in the table can be trusted.")
    print()
    print("2. Then look at pullback_in_uptrend in the OUT_OF_SAMPLE rows.")
    print("   The in-sample result is already known to look good — that is")
    print("   where the idea came from, so it proves nothing on its own.")
    print("   Only the out-of-sample era is real evidence.")
    print()
    print("3. Edge holding out-of-sample -> worth pursuing further.")
    print("   Edge vanishing out-of-sample -> it was noise, and the honest")
    print("   move is to stop rather than retune until it reappears.")


if __name__ == "__main__":
    main()
