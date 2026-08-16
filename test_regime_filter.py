"""
test_regime_filter.py
=======================
Does the market-regime filter help or hurt?

THE QUESTION
------------
nse_scanner.py blocks EVERY Weekly BUY whenever NIFTY's own Weekly
volatility stop is in a downtrend. That is why the dashboard currently
shows zero Weekly signals — it is deliberate, not a fault.

The rationale is reasonable on its face: don't buy breakouts into a
falling market. But it has a cost that is easy to overlook. It silences
the Weekly timeframe exactly when a falling market is producing the
PULLBACK setups — the one thing that survived out-of-sample validation
(+7 to +12% vs date-matched controls). And it leaves only Monthly
signals, which carry up to 30 days of staleness and, on ATR x 3 monthly
bars, a structural 3-4 month lag behind reversals.

So the filter might be costing more than it saves. This measures it
instead of guessing.

WHAT IT COMPARES
----------------
The identical signal rule, run three ways:

    regime_on          BUY only when NIFTY weekly trend is UP  (current)
    regime_off         BUY regardless of index trend
    regime_inverted    BUY only when NIFTY weekly trend is DOWN

The inverted case is the interesting one. If buying INTO index weakness
outperforms, that supports the pullback finding and suggests the filter
is backwards.

NO LOOK-AHEAD
-------------
The NIFTY regime is computed once over full history with a causal
recursive stop, then each signal reads the regime value AT ITS OWN BAR.
Nothing consults a future index value.

SUCCESS BAR — set before looking
--------------------------------
A variant is only preferable if it beats BOTH controls on at least 2 of 3
hold periods where the current setting does not. Simply producing more
signals is not an improvement.

USAGE
-----
    python test_regime_filter.py
"""

import os

import numpy as np
import pandas as pd

import nse_scanner as ns
from backtest import config as cfg
from backtest.utils import pct_return
from nse_scanner import (
    fetch_single,
    fetch_nifty_daily,
    compute_signals,
    compute_vstop,
    load_universe,
    resample,
    UNIVERSE_CSV,
    INCLUDE_OTHER_CATEGORY,
)

SIGNALS_DIR = "signals"
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "regime_filter_results.csv")

RESAMPLE_RULE = "W"
TIMEFRAME_LABEL = "Weekly"     # the timeframe the filter actually silences
HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 4
DATE_WINDOW_BARS = 26
BOOTSTRAP_ITERS = 2000
SPLIT_DATE = pd.Timestamp("2023-01-01")   # report out-of-sample only

RNG = np.random.default_rng(31)


def bootstrap_diff(a, b, iters=BOOTSTRAP_ITERS):
    a = np.asarray(pd.Series(a).dropna())
    b = np.asarray(pd.Series(b).dropna())
    if len(a) < 30 or len(b) < 30:
        return None
    d = np.empty(iters)
    for i in range(iters):
        d[i] = RNG.choice(a, len(a), replace=True).mean() - RNG.choice(b, len(b), replace=True).mean()
    return {"diff": round(float(a.mean() - b.mean()), 2),
            "lo": round(float(np.percentile(d, 2.5)), 2),
            "hi": round(float(np.percentile(d, 97.5)), 2)}


def window_return(bars, pos, hold):
    if pos < 0 or pos + hold >= len(bars):
        return None
    entry = float(bars.iloc[pos]["Open"])
    exit_ = float(bars.iloc[pos + hold]["Close"])
    return pct_return(entry, exit_) - cfg.ROUND_TRIP_COST_PCT if entry > 0 else None


def same_date_control(ob, entry_date, hold):
    if len(ob) == 0 or entry_date < ob.index[0] or entry_date > ob.index[-1]:
        return None
    pos = ob.index.searchsorted(entry_date, side="right")
    if pos <= 0 or pos + hold >= len(ob):
        return None
    return window_return(ob, pos, hold)


def build_regime_series(nifty_daily):
    """
    NIFTY weekly uptrend as a boolean series indexed by weekly bar date.

    compute_vstop is a causal recursive stop — the value at bar i depends
    only on bars up to i — so computing it once over full history and then
    reading the value at each signal's own bar introduces no look-ahead.
    """
    tf = resample(nifty_daily, RESAMPLE_RULE)
    v = compute_vstop(tf, ns.ATR_LEN, ns.ATR_MULT, ns.VSTOP_WHIPSAW_BUFFER_PCT)
    return v["UPTREND"].astype(bool) if not v.empty else pd.Series(dtype=bool)


def regime_at(regime, when):
    """Regime value as of `when`, using the last bar at or before it."""
    if regime.empty:
        return True
    pos = regime.index.searchsorted(when, side="right") - 1
    if pos < 0:
        return True
    return bool(regime.iloc[pos])


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
    symbols = uni["Symbol"].dropna().astype(str).str.strip().tolist()
    print(f"Universe: {len(symbols)} symbols  |  timeframe: {TIMEFRAME_LABEL}")

    print("Fetching NIFTY...")
    nd = fetch_nifty_daily()
    if nd is None or nd.empty:
        raise SystemExit("Could not fetch NIFTY — the regime filter cannot be tested without it.")
    nifty_close = nd["Close"]
    regime = build_regime_series(nd)
    up_share = float(regime.mean()) * 100 if len(regime) else float("nan")
    print(f"NIFTY weekly regime UP on {up_share:.1f}% of weeks in history\n")

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
            if sig is not None and not sig.empty and "BUY_SIGNAL_CORE" in sig.columns:
                frames[sym] = sig
        except Exception:
            continue
    print(f"Ready: {len(frames)} symbols\n")

    cached = sorted(frames.keys())

    # Core signal positions per symbol, computed once and reused.
    core = {}
    for sym, df in frames.items():
        m = df["BUY_SIGNAL_CORE"].fillna(False).astype(bool)
        new = m & ~m.shift(1, fill_value=False)
        core[sym] = np.flatnonzero(new.values)

    VARIANTS = [
        ("regime_on  (current)", lambda up: up),
        ("regime_off",           lambda up: True),
        ("regime_inverted",      lambda up: not up),
    ]

    rows = []
    for label, gate in VARIANTS:
        print(f"=== {label} ===")
        total = 0
        for sym, positions in core.items():
            df = frames[sym]
            total += sum(1 for p in positions if gate(regime_at(regime, df.index[p])))
        print(f"  entries (all history): {total}")
        if total < 100:
            print("  too few to evaluate — skipping\n")
            continue

        for hold in HOLD_PERIODS:
            sig_r, date_r, stock_r = [], [], []
            for sym, positions in core.items():
                df = frames[sym]
                for p in positions:
                    when = df.index[p]
                    if when < SPLIT_DATE:
                        continue
                    if not gate(regime_at(regime, when)):
                        continue
                    r = window_return(df, p + 1, hold)
                    if r is None:
                        continue
                    sig_r.append(r)

                    for _ in range(CONTROLS_PER_SIGNAL):
                        other = cached[int(RNG.integers(0, len(cached)))]
                        rr = same_date_control(frames[other], when, hold)
                        if rr is not None:
                            date_r.append(rr)
                    for _ in range(CONTROLS_PER_SIGNAL):
                        off = int(RNG.integers(-DATE_WINDOW_BARS, DATE_WINDOW_BARS + 1))
                        rr = window_return(df, p + 1 + off, hold)
                        if rr is not None:
                            stock_r.append(rr)

            if len(sig_r) < 30:
                print(f"  hold={hold:>3d}  only {len(sig_r)} out-of-sample entries — skipped")
                continue

            row = {"variant": label, "hold": hold, "n": len(sig_r),
                   "win_rate": round(float((pd.Series(sig_r) > 0).mean() * 100), 1),
                   "mean_return": round(float(np.mean(sig_r)), 2)}
            for name, ctrl in [("vs_date", date_r), ("vs_stock", stock_r)]:
                bs = bootstrap_diff(sig_r, ctrl)
                if bs:
                    row[f"{name}_diff"] = bs["diff"]
                    row[f"{name}_ci"] = f"[{bs['lo']}, {bs['hi']}]"
                    row[f"{name}_sig"] = ("BETTER" if bs["lo"] > 0
                                          else "WORSE" if bs["hi"] < 0 else "none")
            rows.append(row)
            print(f"  hold={hold:>3d}  n={row['n']:>5d}  mean={row['mean_return']:>+7.2f}%  "
                  f"vs_date={row.get('vs_date_sig','-'):<7s} vs_stock={row.get('vs_stock_sig','-')}")
        print()

    if not rows:
        print("Not enough data to evaluate.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print("=" * 100)
    print("  MARKET REGIME FILTER — OUT-OF-SAMPLE (2023+)")
    print("=" * 100)
    print(out.to_string(index=False))
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()
    print("SCORECARD")
    print("-" * 100)
    for label, _ in VARIANTS:
        h = out[out.variant == label]
        if h.empty:
            continue
        d = int((h.get("vs_date_sig") == "BETTER").sum())
        s = int((h.get("vs_stock_sig") == "BETTER").sum())
        print(f"  {label:<22s} n={int(h.n.mean()):>5d}  mean={h.mean_return.mean():>+6.2f}%  "
              f"vs_date {d}/3   vs_stock {s}/3")
    print()
    print("HOW TO READ IT")
    print("  regime_off with MORE signals and NO worse edge -> the filter is")
    print("    costing you opportunities for no measurable protection.")
    print("  regime_on clearly ahead -> the filter earns its place; keep it.")
    print("  regime_inverted ahead -> buying into index weakness works better,")
    print("    which would fit the pullback finding and mean the filter is")
    print("    pointed the wrong way.")
    print("  All three showing 'none' -> the filter is irrelevant either way,")
    print("    consistent with every other test: entry timing on price data")
    print("    alone does not predict returns in this universe.")


if __name__ == "__main__":
    main()
