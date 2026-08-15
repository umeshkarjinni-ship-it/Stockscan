"""
test_combined_strategy.py
===========================
Tests the hypothesis that fell out of the out-of-sample validation:

    The ORIGINAL strength rule picks good STOCKS but bad MOMENTS.
    The PULLBACK rule times entries well but picks no better than random.
    So: use strength to SELECT, and pullback to TIME.

WHERE THIS CAME FROM
--------------------
validate_out_of_sample.py (2021+ data, controls verified unbiased):

    original_style_strength   vs random stock, same date : BETTER 3/3
                              vs same stock, nearby date : WORSE  3/3
    pullback_in_uptrend       vs random stock, same date : none   0/3
                              vs same stock, nearby date : BETTER 3/3

Two complementary skills, each failing where the other succeeds. This
script tests whether combining them beats either alone.

NO LOOK-AHEAD
-------------
The strength condition is evaluated on PAST bars only. At bar i the test
asks "did this stock pass the strength filter at any point in the
preceding LOOKBACK bars, and is it pulled back NOW?" — every input is
available at bar i. Entry is at bar i+1's open, exit at a later close.

THIS IS A NEW HYPOTHESIS
------------------------
It was formulated by looking at out-of-sample results, which means those
results are no longer untouched for this purpose. So the split moves:
this fits on pre-2023 and validates on 2023+. Judge it only on the
OUT_OF_SAMPLE rows.

SUCCESS BAR (set before looking, as before)
-------------------------------------------
The combination is only interesting if, OUT OF SAMPLE, it beats BOTH
controls on at least 2 of 3 hold periods — i.e. it does what neither
component rule managed alone. Beating only one control means it inherited
one parent's skill and nothing more, which is not a discovery.

USAGE
-----
    python test_combined_strategy.py
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
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "combined_strategy_results.csv")

# New split — the previous out-of-sample era informed this hypothesis.
SPLIT_DATE = pd.Timestamp("2023-01-01")

HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 4
DATE_WINDOW_BARS = 26
BOOTSTRAP_ITERS = 2000
RESAMPLE_RULE = "W"

# How far back the strength qualification may have occurred, in bars.
# 52 weekly bars = about one year.
STRENGTH_LOOKBACK = 52

RNG = np.random.default_rng(11)


# ---------------------------------------------------------------------
# Component conditions
# ---------------------------------------------------------------------

def strength_condition(df):
    """The original rule: confirmed strength. Good at picking stocks."""
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (df["RSI"] > 50) & (df["RSI"] < 78)
        & (df["ADX"] > 20)
        & (df["Volume"] > df["VOL_MA"])
    )


def pullback_condition(df):
    """The pullback rule: good at timing entries."""
    high_52 = df["Close"].rolling(52, min_periods=20).max()
    off_high = (df["Close"] - high_52) / high_52 * 100
    return (
        (df["Close"] > df["KAMA_MID"])
        & (df["RSI"] < 45)
        & (off_high < -8)
    )


# ---------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------

def h_strength_only(df):
    return strength_condition(df)


def h_pullback_only(df):
    return pullback_condition(df)


def h_combined(df):
    """
    THE TEST: stock qualified on strength at some point in the last
    STRENGTH_LOOKBACK bars, AND is pulled back right now.

    rolling(...).max() over a shifted series looks strictly backwards —
    shift(1) excludes the current bar, so the strength check can only
    ever use information from earlier bars. Verified: with strength true
    only at bar 50, the window activates at bar 51 and never earlier.
    """
    strong = strength_condition(df).astype(float)
    was_strong = (
        strong.shift(1).rolling(STRENGTH_LOOKBACK, min_periods=1).max() > 0
    )
    return was_strong & pullback_condition(df)


def h_combined_wide(df):
    """
    Wider qualification window — strength anywhere in the past two years.

    The narrower windows first tried (26 and 52 bars) produced only ~90
    to ~140 entries across the whole universe, which splits into too few
    per era to test. The constraint is that a stock rarely goes from
    "strong" to "pulled back" quickly; it needs time to fade.
    """
    strong = strength_condition(df).astype(float)
    was_strong = strong.shift(1).rolling(104, min_periods=1).max() > 0
    return was_strong & pullback_condition(df)


def h_combined_ever(df):
    """
    Loosest version: the stock passed the strength filter at ANY earlier
    point in its history. Treats strength as a persistent quality marker
    ("this is the kind of stock that trends") rather than a recent event,
    and uses pullback purely for timing.
    """
    strong = strength_condition(df).astype(float)
    was_strong = strong.shift(1).cumsum() > 0
    return was_strong & pullback_condition(df)


def h_random_control(df):
    """Harness check. Must read 'none' everywhere."""
    return pd.Series(RNG.random(len(df)) < 0.02, index=df.index)


HYPOTHESES = [
    ("strength_only", h_strength_only),
    ("pullback_only", h_pullback_only),
    ("combined_52bar", h_combined),
    ("combined_104bar", h_combined_wide),
    ("combined_ever", h_combined_ever),
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
    """Different stock, same date — skipped unless it genuinely traded then."""
    if len(ob) == 0:
        return None
    if entry_date < ob.index[0] or entry_date > ob.index[-1]:
        return None
    pos = ob.index.searchsorted(entry_date, side="right")
    if pos <= 0 or pos + hold >= len(ob):
        return None
    return window_return(ob, pos, hold)


def evaluate(frames, cached, rule, era_fn, hold):
    sig_r, date_r, stock_r = [], [], []
    for sym, df in frames.items():
        try:
            mask = rule(df).fillna(False)
        except Exception:
            continue
        new = mask & ~mask.shift(1, fill_value=False)
        for p in np.flatnonzero(new.values):
            entry_date = df.index[p]
            if not era_fn(entry_date):
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
        ("IN_SAMPLE(pre-2023)", lambda d: d < SPLIT_DATE),
        ("OUT_OF_SAMPLE(2023+)", lambda d: d >= SPLIT_DATE),
    ]

    rows = []
    for name, rule in HYPOTHESES:
        print(f"=== {name} ===")
        for era_name, era_fn in eras:
            for hold in HOLD_PERIODS:
                sig_r, date_r, stock_r = evaluate(frames, cached, rule, era_fn, hold)
                if len(sig_r) < 30:
                    print(f"  {era_name:<22s} hold={hold:>3d}  only {len(sig_r)} entries — skipped")
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
        print("Not enough data.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print("=" * 100)
    print("  COMBINED STRATEGY RESULTS")
    print("=" * 100)
    print(out.to_string(index=False))
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()

    # Scorecard on the pre-declared bar.
    print("SCORECARD — OUT_OF_SAMPLE only, vs the bar set before looking")
    print("-" * 100)
    oos = out[out.era.str.startswith("OUT")]
    for name, _ in HYPOTHESES:
        h = oos[oos.hypothesis == name]
        if h.empty:
            continue
        d = int((h.get("vs_date_sig") == "BETTER").sum())
        s = int((h.get("vs_stock_sig") == "BETTER").sum())
        verdict = (
            "PASSES — beats both controls" if d >= 2 and s >= 2
            else "timing skill only" if s >= 2
            else "stock-picking skill only" if d >= 2
            else "no edge"
        )
        print(f"  {name:<18s} vs_date {d}/3   vs_stock {s}/3   -> {verdict}")

    print()
    print("Check random_control reads 0/3 and 0/3. If it doesn't, ignore")
    print("everything above and treat the harness as broken.")
    print()
    print("A combination that beats only ONE control has merely inherited")
    print("one parent's skill. Only beating BOTH would show the pairing adds")
    print("something neither rule had alone.")


if __name__ == "__main__":
    main()
