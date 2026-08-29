"""
validate_sell_signals.py
========================
The SELL side has never been benchmarked.

Every test in the project so far interrogates the ENTRY rule. The exit /
SELL rule is a bare VStop flip:

    nse_scanner.py:494    out["SELL_SIGNAL"] = out["FLIP_DOWN"]

One condition, no confirmation, against seven conditions on the BUY side.
It has never been compared to a control, so "SELL" on the dashboard is an
unvalidated label attached to a raw indicator flip.

There is also a live contradiction to resolve. Today's SELL list is
mostly stocks 14-55% off their 52-week highs with RSI in the 36-48 band —
close to the pullback definition. On 2026-08-28 the dashboard rendered a
PULLBACK badge on a SELL row (BSE, RSI 43.3, -22.5% off high). The one
validated finding in this project says that condition is a BETTER entry
than buying strength. The scanner says sell it. Both cannot be right.

WHAT THIS TESTS
---------------
  sell_live_rule           FLIP_DOWN — exactly what nse_scanner.py:494 fires on
  sell_confirmed           FLIP_DOWN + RSI < 45 + Close < KAMA_MID
  sell_excluding_pullback  FLIP_DOWN, minus rows in the pullback zone
  sell_pullback_only       FLIP_DOWN rows that ARE in the pullback zone
                           -> this is the BSE case. Resolves the contradiction.
  random_control           Mandatory harness check. Must read 'none' everywhere.

HOW TO READ IT (the sign is inverted vs the BUY tests)
-----------------------------------------------------
This measures forward LONG return after the signal — "what happened to
the stock if you had held". So for a SELL:

    signal return SIGNIFICANTLY BELOW control  -> SELL EDGE (it works)
    CI straddles zero                          -> NO EDGE (it is noise)
    signal return SIGNIFICANTLY ABOVE control  -> BACKWARDS (it sells lows)

The verdict column already applies that inversion. Do not read the raw
vs_date_sig / vs_stock_sig columns as if this were a BUY test.

Controls, cost model, bootstrap and era-split logic are IMPORTED from
validate_out_of_sample.py rather than reimplemented, so the SELL rule
faces exactly the same harness the pullback finding survived.

USAGE
-----
    python validate_sell_signals.py --timeframe W          # weekly
    python validate_sell_signals.py --timeframe ME         # monthly
    python validate_sell_signals.py --timeframe W --max-symbols 60   # smoke test

Writes signals/sell_validation_<timeframe>.csv
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

# Import the harness itself, not a copy of it. If these functions ever
# change, this test changes with them — which is the point.
import validate_out_of_sample as voos
from validate_out_of_sample import (
    bootstrap_diff,
    window_return,
    same_date_control,
    CONTROLS_PER_SIGNAL,
)

SIGNALS_DIR = "signals"

# Three eras. 2023+ is deliberately a SUBSET of 2021+, matching how the
# pullback finding was reported ("replicated across 2021+ and 2023+").
# Agreement between the two nested out-of-sample windows is weak evidence
# on its own, but disagreement is a red flag worth seeing.
SPLIT_2021 = pd.Timestamp("2021-01-01")
SPLIT_2023 = pd.Timestamp("2023-01-01")

# Timeframe-dependent settings. The weekly numbers match
# validate_out_of_sample.py exactly. Monthly holds and the control window
# are scaled by ~1/4 so they cover comparable CALENDAR spans — reusing 20
# / 40 / 60 on monthly bars would test 1.7 to 5 year holds, and a +/-26
# bar control window would let the "nearby date" control wander over four
# years, which is not a control at all.
TF_SETTINGS = {
    "W":  {"name": "Weekly",  "holds": [20, 40, 60], "date_window": 26, "hi_lookback": 52},
    "ME": {"name": "Monthly", "holds": [3, 6, 12],   "date_window": 6,  "hi_lookback": 12},
}


# ---------------------------------------------------------------------
# Hypotheses
#
# Parameters are FROZEN at the values the live scanner and the existing
# pullback finding already use. Nothing here is tuned. If a rule fails,
# the honest response is to stop calling it SELL, not to search for
# thresholds that make it pass.
# ---------------------------------------------------------------------

def _pullback_mask(df, hi_lookback):
    """Same definition as PULLBACK on the dashboard and in nse_scanner."""
    high = df["Close"].rolling(hi_lookback, min_periods=max(4, hi_lookback // 4)).max()
    off_high = (df["Close"] - high) / high * 100
    return (
        (df["Close"] > df["KAMA_MID"])
        & (df["RSI"] < 45)
        & (off_high <= -8)
    )


def build_hypotheses(hi_lookback):
    def sell_live_rule(df):
        return df["FLIP_DOWN"].fillna(False)

    def sell_confirmed(df):
        return (
            df["FLIP_DOWN"].fillna(False)
            & (df["RSI"] < 45)
            & (df["Close"] < df["KAMA_MID"])
        )

    def sell_excluding_pullback(df):
        return df["FLIP_DOWN"].fillna(False) & ~_pullback_mask(df, hi_lookback).fillna(False)

    def sell_pullback_only(df):
        return df["FLIP_DOWN"].fillna(False) & _pullback_mask(df, hi_lookback).fillna(False)

    def random_control(df):
        return pd.Series(voos.RNG.random(len(df)) < 0.02, index=df.index)

    return [
        ("sell_live_rule", sell_live_rule, "sell"),
        ("sell_confirmed", sell_confirmed, "sell"),
        ("sell_excluding_pullback", sell_excluding_pullback, "sell"),
        ("sell_pullback_only", sell_pullback_only, "sell"),
        ("random_control", random_control, "sell"),
    ]


# ---------------------------------------------------------------------

def evaluate(frames, cached, rule, era_mask_fn, hold, date_window):
    """
    Same structure as validate_out_of_sample.evaluate, with the control
    window passed in rather than read from a module constant, so Monthly
    can use a sane +/-6 bar window instead of +/-26.

    window_return and same_date_control are the IMPORTED originals.
    """
    sig_r, date_r, stock_r = [], [], []

    for sym, df in frames.items():
        try:
            mask = rule(df).fillna(False)
        except Exception:
            continue

        # FLIP_DOWN is already a single-bar event, so this is a no-op for
        # the SELL rules. Kept so state-style rules behave identically to
        # the BUY harness.
        new = mask & ~mask.shift(1, fill_value=False)
        pos_arr = np.flatnonzero(new.values)

        for p in pos_arr:
            entry_date = df.index[p]
            if not era_mask_fn(entry_date):
                continue

            # Enter the bar AFTER the signal — no look-ahead.
            r = window_return(df, p + 1, hold)
            if r is None:
                continue
            sig_r.append(r)

            for _ in range(CONTROLS_PER_SIGNAL):
                other = cached[int(voos.RNG.integers(0, len(cached)))]
                rr = same_date_control(frames[other], entry_date, hold)
                if rr is not None:
                    date_r.append(rr)

            for _ in range(CONTROLS_PER_SIGNAL):
                off = int(voos.RNG.integers(-date_window, date_window + 1))
                rr = window_return(df, p + 1 + off, hold)
                if rr is not None:
                    stock_r.append(rr)

    return sig_r, date_r, stock_r


def verdict_for_sell(bs):
    """
    Invert the BUY reading. bs is the CI on (signal - control) forward
    LONG return, so a working SELL is a NEGATIVE difference.
    """
    if bs is None:
        return "insufficient n"
    if bs["hi"] < 0:
        return "SELL EDGE"
    if bs["lo"] > 0:
        return "BACKWARDS"
    return "no edge"


def summarise_consistency(out):
    """
    Aggregate the out-of-sample cells per hypothesis.

    This run evaluates 5 hypotheses x 3 eras x 3 holds x 2 controls = 90
    cells at a 95% CI, so roughly 4-5 will flag on noise alone. Reading a
    single flagged cell as a finding is the mistake this table exists to
    prevent. A real effect shows up in MOST out-of-sample cells and points
    the SAME way; noise shows up in one or two, often in both directions.

    Note the two out-of-sample eras are nested — 2023+ is a subset of
    2021+ — so agreement between them is weaker evidence than it looks.
    """
    oos = out[out["era"].str.startswith("OUT_OF_SAMPLE")]
    if oos.empty:
        return

    verdict_cols = [c for c in ["vs_date_verdict", "vs_stock_verdict"] if c in oos.columns]
    if not verdict_cols:
        return

    print()
    print("=" * 104)
    print("  CONSISTENCY ACROSS OUT-OF-SAMPLE CELLS  (do not read single cells)")
    print("=" * 104)
    print(f"  {'hypothesis':<26s} {'cells':>6s} {'SELL EDGE':>10s} {'no edge':>9s} "
          f"{'BACKWARDS':>10s}   read")

    for hyp in out["hypothesis"].unique():
        sub = oos[oos["hypothesis"] == hyp]
        verdicts = [str(v) for c in verdict_cols for v in sub[c].dropna()]
        if not verdicts:
            continue
        total = len(verdicts)
        n_edge = verdicts.count("SELL EDGE")
        n_none = verdicts.count("no edge")
        n_back = verdicts.count("BACKWARDS")

        if n_edge >= 0.6 * total:
            read = "consistent SELL EDGE"
        elif n_back >= 0.6 * total:
            read = "consistent BACKWARDS"
        elif n_none >= 0.6 * total:
            read = "no edge"
        else:
            read = "mixed — treat as no edge"

        marker = " <-- gate" if hyp == "random_control" else ""
        print(f"  {hyp:<26s} {total:>6d} {n_edge:>10d} {n_none:>9d} {n_back:>10d}   "
              f"{read}{marker}")

    print()
    print("  A finding needs the same direction in MOST cells. One or two flags")
    print("  scattered across 90 cells is what a 95% CI produces by construction.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", choices=["W", "ME"], default="W",
                    help="W = weekly bars, ME = monthly bars")
    ap.add_argument("--max-symbols", type=int, default=0,
                    help="Cap the universe for a fast smoke run (0 = all)")
    args = ap.parse_args()

    tf = TF_SETTINGS[args.timeframe]
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    out_file = os.path.join(SIGNALS_DIR, f"sell_validation_{tf['name'].lower()}.csv")

    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
    symbols = uni["Symbol"].dropna().astype(str).str.strip().tolist()
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    print(f"Timeframe: {tf['name']}   Universe: {len(symbols)} symbols")
    print(f"Holds: {tf['holds']} bars   Control window: +/-{tf['date_window']} bars\n")

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
            sig = compute_signals(resample(raw.sort_index(), args.timeframe), nifty_close)
            if sig is not None and not sig.empty and "FLIP_DOWN" in sig.columns:
                frames[sym] = sig
        except Exception:
            continue
    print(f"Ready: {len(frames)} symbols\n")

    if len(frames) < 20:
        raise SystemExit("Too few symbols with usable history to run a controlled test.")

    cached = sorted(frames.keys())
    eras = [
        ("IN_SAMPLE(pre-2021)",   lambda d: d < SPLIT_2021),
        ("OUT_OF_SAMPLE(2021+)",  lambda d: d >= SPLIT_2021),
        ("OUT_OF_SAMPLE(2023+)",  lambda d: d >= SPLIT_2023),
    ]

    rows = []
    for name, rule, direction in build_hypotheses(tf["hi_lookback"]):
        print(f"=== {name} ===")
        for era_name, era_fn in eras:
            for hold in tf["holds"]:
                sig_r, date_r, stock_r = evaluate(
                    frames, cached, rule, era_fn, hold, tf["date_window"]
                )
                if len(sig_r) < 30:
                    print(f"  {era_name:<22s} hold={hold:>3d} n={len(sig_r):>5d}  (skipped, n<30)")
                    continue

                row = {
                    "timeframe": tf["name"],
                    "hypothesis": name,
                    "era": era_name,
                    "hold": hold,
                    "n": len(sig_r),
                    "mean_return": round(float(np.mean(sig_r)), 2),
                    "median_return": round(float(np.median(sig_r)), 2),
                }
                for label, ctrl in [("vs_date", date_r), ("vs_stock", stock_r)]:
                    bs = bootstrap_diff(sig_r, ctrl)
                    if bs:
                        row[f"{label}_diff"] = bs["diff"]
                        row[f"{label}_ci"] = f"[{bs['lo']}, {bs['hi']}]"
                        row[f"{label}_verdict"] = verdict_for_sell(bs)
                    else:
                        row[f"{label}_verdict"] = "insufficient n"
                rows.append(row)
                print(f"  {era_name:<22s} hold={hold:>3d} n={row['n']:>5d} "
                      f"mean={row['mean_return']:>+7.2f}%  "
                      f"vs_date={row.get('vs_date_verdict','-'):<14s} "
                      f"vs_stock={row.get('vs_stock_verdict','-')}")
        print()

    if not rows:
        print("Not enough data to evaluate.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(out_file, index=False)

    print("=" * 104)
    print(f"  SELL SIGNAL VALIDATION — {tf['name']} bars, net of costs")
    print("=" * 104)
    print(out.to_string(index=False))

    summarise_consistency(out)

    print()
    print(f"Saved to {out_file}")
    print()
    print("HOW TO READ THIS")
    print("-" * 104)
    print("0. Verdicts are ALREADY sign-inverted for SELL. 'SELL EDGE' means the")
    print("   stock underperformed its control after the signal, which is what a")
    print("   working SELL should do.")
    print()
    print("1. Check random_control FIRST. It must read 'no edge' in every era")
    print("   against both controls. If it does not, the harness is biased and")
    print("   nothing else in the table means anything.")
    print()
    print("2. Then read sell_live_rule in the OUT_OF_SAMPLE rows only. That is")
    print("   the rule nse_scanner.py:494 actually fires on today.")
    print()
    print("3. sell_pullback_only settles the BSE case. 'BACKWARDS' there means")
    print("   the scanner is flagging SELL at the exact condition the validated")
    print("   pullback finding calls a better-than-average entry, and those rows")
    print("   should be suppressed from the SELL panel regardless of how the")
    print("   rest of the rule scores.")
    print()
    print("4. If sell_live_rule reads 'no edge' out-of-sample, the honest move is")
    print("   to relabel the panel 'Trend break' and stop calling it SELL — not")
    print("   to search for a confirmation filter that makes it pass. Adding")
    print("   filters until one works is how the PF 2.42 result happened.")
    print()
    print("Costs are subtracted from signal and control alike, so they cancel in")
    print("the difference. These are overlapping windows, not an equity curve.")


if __name__ == "__main__":
    main()
