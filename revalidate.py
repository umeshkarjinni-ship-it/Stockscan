"""
revalidate.py
=============
Re-tests every hypothesis this project has relied on, using controls that
pass the self-test in controls.py.

WHY THIS EXISTS
---------------
The +7% to +12% pullback finding came from the same-stock +/-W control.
That control flags on 12 of 12 synthetic runs where no edge exists, when
the signal fires at local minima — which pullback and FLIP_DOWN both do.
The clean same-date control read +0.28 / -0.71 / -2.75 out-of-sample for
that same hypothesis, with every CI excluding the published effect size.

So the finding needs re-testing, not patching. This runs all of it in one
pass so the pullback rule and the SELL rule are judged by the same yardstick.

CONTROLS USED
-------------
  same_date     Different stock, same date.              Selection.
  forward_only  Same stock, offsets +1..+W only.         Timing.
  peer          Same date, matched on % off 52w high.    Selection among
                                                         equally beaten-down
                                                         stocks. Strictest.

The legacy +/-W control is deliberately NOT computed. It is broken for
this class of signal and reporting it invites reading it.

READING THE RESULT
------------------
These are BUY-side hypotheses, so the sign is NOT inverted here (unlike
validate_sell_signals.py). Positive diff = the signal beat its control.

`peer` is the one that matters for a pullback claim. If a rule beats
same_date but not peer, it is picking beaten-down stocks rather than
picking well among them — and "buy things that have fallen" is not a
finding, it is a description of the era.

USAGE
-----
    python revalidate.py --max-symbols 60      # smoke run
    python revalidate.py                       # full

Writes signals/revalidation_weekly.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from nse_scanner import (
    fetch_single,
    fetch_nifty_daily,
    compute_signals,
    load_universe,
    resample,
    UNIVERSE_CSV,
    INCLUDE_OTHER_CATEGORY,
)

from controls import (
    window_return,
    signal_admissible,
    same_date_control,
    forward_only_control,
    pct_off_high_series,
    bootstrap_diff,
)

SIGNALS_DIR = "signals"
OUT_FILE = os.path.join(SIGNALS_DIR, "revalidation_weekly.csv")

HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 4
DATE_WINDOW_BARS = 26
HI_LOOKBACK = 52
PEER_TOLERANCE = 5.0
RESAMPLE_RULE = "W"

SPLIT_2021 = pd.Timestamp("2021-01-01")
SPLIT_2023 = pd.Timestamp("2023-01-01")

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------
# Hypotheses — parameters FROZEN at their published values. Nothing here
# is retuned. If a rule fails, that is the result.
# ---------------------------------------------------------------------

def h_pullback_in_uptrend(df):
    high = df["Close"].rolling(52, min_periods=20).max()
    off = (df["Close"] - high) / high * 100
    return (df["Close"] > df["KAMA_MID"]) & (df["RSI"] < 45) & (off < -8)


def h_deep_pullback(df):
    high = df["Close"].rolling(52, min_periods=20).max()
    off = (df["Close"] - high) / high * 100
    return (df["Close"] > df["KAMA_SLOW"]) & (df["RSI"] < 40) & (off < -15)


def h_original_style(df):
    return (
        (df["Close"] > df["KAMA_SLOW"])
        & (df["RSI"] > 50) & (df["RSI"] < 78)
        & (df["ADX"] > 20)
        & (df["Volume"] > df["VOL_MA"])
    )


def h_sell_flip_down(df):
    return df["FLIP_DOWN"].fillna(False)


def h_random_control(df):
    return pd.Series(RNG.random(len(df)) < 0.02, index=df.index)


HYPOTHESES = [
    ("pullback_in_uptrend", h_pullback_in_uptrend),
    ("deep_pullback", h_deep_pullback),
    ("original_style_strength", h_original_style),
    ("sell_flip_down", h_sell_flip_down),
    ("random_control", h_random_control),
]


def build_peer_panel(drawdowns):
    """Date-indexed DataFrame of % off 52w high, one column per symbol."""
    return pd.DataFrame(drawdowns).sort_index()


def peer_control_fast(peer_panel, frames, entry_date, signal_dd, hold, sym):
    """
    Same-date control drawn from stocks within PEER_TOLERANCE points of the
    signal stock's drawdown. Uses a precomputed cross-section instead of
    rejection sampling — with tens of thousands of signals the retry loop
    in controls.peer_control is too slow.
    """
    if signal_dd is None or not np.isfinite(signal_dd):
        return None
    if entry_date not in peer_panel.index:
        return None
    row = peer_panel.loc[entry_date]
    eligible = row[(row - signal_dd).abs() <= PEER_TOLERANCE].index
    eligible = [s for s in eligible if s != sym]
    if not eligible:
        return None
    other = eligible[int(RNG.integers(0, len(eligible)))]
    return same_date_control(frames[other], entry_date, hold)


def evaluate(frames, drawdowns, peer_panel, symbols, rule, era_fn, hold):
    sig, c_date, c_fwd, c_peer = [], [], [], []

    for sym in symbols:
        df = frames[sym]
        try:
            mask = rule(df).fillna(False)
        except Exception:
            continue
        new = mask & ~mask.shift(1, fill_value=False)
        dd = drawdowns[sym]

        for p in np.flatnonzero(new.values):
            sig_pos = p + 1
            # Matched validity — every forward offset must be evaluable,
            # or the control silently skews toward earlier periods.
            if not signal_admissible(df, sig_pos, hold, DATE_WINDOW_BARS):
                continue
            entry_date = df.index[sig_pos]
            if not era_fn(entry_date):
                continue

            r = window_return(df, sig_pos, hold)
            if r is None:
                continue
            sig.append(r)
            signal_dd = dd.iloc[sig_pos] if sig_pos < len(dd) else np.nan

            for _ in range(CONTROLS_PER_SIGNAL):
                other = symbols[int(RNG.integers(0, len(symbols)))]
                rr = same_date_control(frames[other], entry_date, hold)
                if rr is not None:
                    c_date.append(rr)

                rr = forward_only_control(df, sig_pos, hold, DATE_WINDOW_BARS, RNG)
                if rr is not None:
                    c_fwd.append(rr)

                rr = peer_control_fast(peer_panel, frames, entry_date,
                                       signal_dd, hold, sym)
                if rr is not None:
                    c_peer.append(rr)

    return sig, c_date, c_fwd, c_peer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-symbols", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(SIGNALS_DIR, exist_ok=True)

    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
    symbols = uni["Symbol"].dropna().astype(str).str.strip().tolist()
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    print(f"Universe: {len(symbols)} symbols")

    print("Fetching NIFTY...")
    nd = fetch_nifty_daily()
    nifty_close = nd["Close"] if nd is not None else None

    print("Building signal frames...")
    frames, drawdowns = {}, {}
    for n, sym in enumerate(symbols, start=1):
        if n % 50 == 0 or n == 1:
            print(f"  [{n}/{len(symbols)}]", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is None or len(raw) == 0:
                continue
            sig = compute_signals(resample(raw.sort_index(), RESAMPLE_RULE), nifty_close)
            if sig is None or sig.empty or "FLIP_DOWN" not in sig.columns:
                continue
            frames[sym] = sig
            drawdowns[sym] = pct_off_high_series(sig, HI_LOOKBACK)
        except Exception:
            continue
    print(f"Ready: {len(frames)} symbols\n")

    if len(frames) < 20:
        raise SystemExit("Too few symbols to run a controlled test.")

    syms = sorted(frames.keys())
    peer_panel = build_peer_panel(drawdowns)

    eras = [
        ("IN_SAMPLE(pre-2021)", lambda d: d < SPLIT_2021),
        ("OUT_OF_SAMPLE(2021+)", lambda d: d >= SPLIT_2021),
        ("OUT_OF_SAMPLE(2023+)", lambda d: d >= SPLIT_2023),
    ]

    rows = []
    for name, rule in HYPOTHESES:
        print(f"=== {name} ===")
        for era_name, era_fn in eras:
            for hold in HOLD_PERIODS:
                sig, cd, cf, cp = evaluate(
                    frames, drawdowns, peer_panel, syms, rule, era_fn, hold
                )
                if len(sig) < 30:
                    print(f"  {era_name:<22s} hold={hold:>3d} n={len(sig):>5d}  (skipped)")
                    continue

                row = {
                    "hypothesis": name,
                    "era": era_name,
                    "hold": hold,
                    "n": len(sig),
                    "mean_return": round(float(np.mean(sig)), 2),
                }
                for label, ctrl in [("same_date", cd), ("forward_only", cf),
                                    ("peer", cp)]:
                    bs = bootstrap_diff(sig, ctrl, RNG)
                    if bs:
                        row[f"{label}_diff"] = round(bs["diff"], 2)
                        row[f"{label}_ci"] = f"[{bs['lo']:.2f}, {bs['hi']:.2f}]"
                        row[f"{label}_verdict"] = (
                            "BETTER" if bs["lo"] > 0
                            else "WORSE" if bs["hi"] < 0
                            else "no edge"
                        )
                    else:
                        row[f"{label}_verdict"] = "insufficient n"
                rows.append(row)
                print(f"  {era_name:<22s} hold={hold:>3d} n={row['n']:>6d} "
                      f"date={row.get('same_date_verdict','-'):<8s} "
                      f"fwd={row.get('forward_only_verdict','-'):<8s} "
                      f"peer={row.get('peer_verdict','-')}")
        print()

    if not rows:
        print("Not enough data.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False)
    print("=" * 110)
    print("  REVALIDATION — controls that pass controls.py --self-test")
    print("=" * 110)
    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_FILE}\n")
    print("READ IN THIS ORDER")
    print("-" * 110)
    print("1. random_control must read 'no edge' on all three controls. It is")
    print("   the gate, and ~5% of cells flag by construction — judge the")
    print("   pattern, not single cells.")
    print("2. pullback_in_uptrend, OUT_OF_SAMPLE rows, `peer` column. This is")
    print("   the finding the whole project rests on. peer asks whether the")
    print("   rule picks well AMONG beaten-down stocks, which is what it claims.")
    print("3. If pullback beats same_date but not peer, the effect is 'buy")
    print("   things that have fallen', not a signal — and 2021-2026 in Indian")
    print("   mid-caps is exactly the era where that flatters itself.")
    print("4. Nothing here is tradeable on its own. These are overlapping")
    print("   windows with no position sizing and no capital constraint.")


if __name__ == "__main__":
    main()
