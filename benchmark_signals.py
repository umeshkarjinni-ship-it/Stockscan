"""
benchmark_signals.py
======================
Answers the single most important question about this strategy:

    Do the BUY signals actually have edge, or do they just measure the
    fact that the Indian market went up a lot over the test period?

WHY THIS MATTERS
----------------
compare_exits.py showed profit factor rising without limit as the hold
period grew — 40 bars gave PF 1.92, 60 gave 2.42, 130 gave 4.14, and 200
gave 6.05. It never plateaued. That is the classic signature of a result
driven by market exposure rather than signal quality: hold Indian
equities for ~4 years anywhere in 2012-2026 and the numbers look great
regardless of when you bought.

A profit factor is only meaningful RELATIVE to what you'd get without the
signal. This script builds that comparison.

WHAT IT DOES
------------
For every real signal, it constructs control groups on identical terms:

  1. RANDOM_SAME_DATE — enter a DIFFERENT randomly chosen stock on the
                        SAME date, held the same number of bars.
                        Isolates STOCK SELECTION: did this signal pick a
                        better stock than a dart throw on that same day?

  2. RANDOM_SAME_STOCK — enter the SAME stock on a random date drawn from
                        the SAME CALENDAR MONTH ±6 months as the signal.
                        Isolates TIMING while holding the market regime
                        roughly constant.

  3. NIFTY_HOLD       — buy the index over the signal's exact window.

DATE MATCHING (important)
-------------------------
An earlier version drew random entries from anywhere in a stock's
history. That was NOT a fair control: real signals cluster in particular
periods (83% fire while NIFTY is above its 50-bar MA), so the controls
were sampling different market conditions than the signals. Both control
groups are now date-matched, which is what makes the comparison
meaningful.

STATISTICAL SIGNIFICANCE
------------------------
Raw mean differences are not enough when returns range from -75% to
+200%. This script bootstraps a 95% confidence interval on the
difference, so you can see whether an apparent edge is distinguishable
from noise.

HOW TO READ THE RESULT
----------------------
  CI excludes zero and is positive -> real, measurable edge
  CI straddles zero                -> no detectable edge; the apparent
                                      difference is sampling noise

USAGE
-----
    python benchmark_signals.py

Requires signals/backtest_signals.csv (from run_backtest.py).
"""

import os

import numpy as np
import pandas as pd

from backtest import config as cfg
from backtest.utils import pct_return
from nse_scanner import fetch_single, fetch_nifty_daily, resample

SIGNALS_DIR = "signals"
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "benchmark_comparison.csv")

# Hold lengths to test, in BARS of the signal's own timeframe.
HOLD_PERIODS = [20, 40, 60]

# Random control entries generated per real signal.
CONTROLS_PER_SIGNAL = 3

# How far the same-stock control may wander from the signal date, in
# bars. Keeps the control inside a comparable market regime instead of
# sampling a different decade.
DATE_WINDOW_BARS = 26

BOOTSTRAP_ITERS = 2000

RNG = np.random.default_rng(42)


def summarize(returns, label, hold):
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return {"group": label, "hold_bars": hold, "n": 0}
    wins = r[r > 0]
    losses = r[r <= 0]
    gl = abs(losses.sum())
    return {
        "group": label,
        "hold_bars": hold,
        "n": len(r),
        "win_rate": round(len(wins) / len(r) * 100, 1),
        "mean_return": round(r.mean(), 2),
        "median_return": round(r.median(), 2),
        "profit_factor": round(wins.sum() / gl, 2) if gl else None,
    }


def bootstrap_diff(a, b, iters=BOOTSTRAP_ITERS):
    """95% CI on mean(a) - mean(b), and the share of resamples favouring a."""
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
        "pct_positive": round(float((diffs > 0).mean() * 100), 1),
    }


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    if not os.path.exists(cfg.SIGNALS_FILE):
        raise SystemExit(f"\n{cfg.SIGNALS_FILE} not found — run run_backtest.py first.\n")

    signals = pd.read_csv(cfg.SIGNALS_FILE, parse_dates=["signal_date"])
    print(f"Loaded {len(signals)} signals")

    print("Fetching NIFTY...")
    nifty_daily = fetch_nifty_daily()
    nifty_close = nifty_daily["Close"] if nifty_daily is not None else None

    symbols = sorted(signals["symbol"].dropna().unique())
    print(f"Fetching price history for {len(symbols)} symbols (once, reused)...")

    cache = {}
    for n, sym in enumerate(symbols, start=1):
        if n % 50 == 0 or n == 1:
            print(f"  [{n}/{len(symbols)}]", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is not None and len(raw) > 0:
                raw = raw.sort_index()
                cache[sym] = {
                    "Weekly": resample(raw, "W"),
                    "Monthly": resample(raw, "ME"),
                }
        except Exception:
            continue
    print(f"Cached {len(cache)} symbols\n")

    cached_symbols = sorted(cache.keys())
    rows = []
    tests = []

    def window_return(bars, start_pos, hold):
        if start_pos < 0 or start_pos + hold >= len(bars):
            return None
        entry = float(bars.iloc[start_pos]["Open"])
        exit_ = float(bars.iloc[start_pos + hold]["Close"])
        if entry <= 0:
            return None
        return pct_return(entry, exit_) - cfg.ROUND_TRIP_COST_PCT

    for hold in HOLD_PERIODS:
        sig_r, same_date_r, same_stock_r, nifty_r = [], [], [], []

        for _, s in signals.iterrows():
            tf = str(s.get("timeframe", "Weekly"))
            bars = cache.get(s["symbol"], {}).get(tf)
            if bars is None or len(bars) < hold + 2:
                continue

            sig_date = pd.Timestamp(s["signal_date"])
            future_idx = bars.index.searchsorted(sig_date, side="right")

            ret = window_return(bars, future_idx, hold)
            if ret is None:
                continue
            sig_r.append(ret)

            # --- NIFTY over the same calendar window ---
            if nifty_close is not None:
                try:
                    seg = nifty_close.loc[bars.index[future_idx]: bars.index[future_idx + hold]]
                    if len(seg) >= 2:
                        nifty_r.append(
                            pct_return(float(seg.iloc[0]), float(seg.iloc[-1]))
                            - cfg.ROUND_TRIP_COST_PCT
                        )
                except Exception:
                    pass

            # --- Control 1: different stock, SAME date ---
            for _ in range(CONTROLS_PER_SIGNAL):
                other = cached_symbols[int(RNG.integers(0, len(cached_symbols)))]
                ob = cache.get(other, {}).get(tf)
                if ob is None or len(ob) < hold + 2:
                    continue
                opos = ob.index.searchsorted(sig_date, side="right")
                r = window_return(ob, opos, hold)
                if r is not None:
                    same_date_r.append(r)

            # --- Control 2: same stock, nearby date ---
            for _ in range(CONTROLS_PER_SIGNAL):
                offset = int(RNG.integers(-DATE_WINDOW_BARS, DATE_WINDOW_BARS + 1))
                r = window_return(bars, future_idx + offset, hold)
                if r is not None:
                    same_stock_r.append(r)

        rows.append(summarize(sig_r, "SIGNAL", hold))
        rows.append(summarize(same_date_r, "RANDOM_SAME_DATE", hold))
        rows.append(summarize(same_stock_r, "RANDOM_SAME_STOCK", hold))
        rows.append(summarize(nifty_r, "NIFTY_HOLD", hold))

        for label, ctrl in [("vs random stock, same date", same_date_r),
                            ("vs same stock, nearby date", same_stock_r)]:
            bs = bootstrap_diff(sig_r, ctrl)
            if bs:
                bs.update({"hold_bars": hold, "comparison": label})
                tests.append(bs)

        print(f"hold={hold} bars done")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 80)
    print("  SIGNAL vs DATE-MATCHED CONTROLS  (net of costs)")
    print("=" * 80)
    print(out.to_string(index=False))

    if tests:
        t = pd.DataFrame(tests)[
            ["hold_bars", "comparison", "mean_diff", "ci_low", "ci_high", "pct_positive"]
        ]
        print()
        print("=" * 80)
        print("  IS THE EDGE REAL?  95% bootstrap CI on the difference")
        print("=" * 80)
        print(t.to_string(index=False))
        print()
        for _, r in t.iterrows():
            verdict = (
                "REAL EDGE (CI excludes zero)" if r.ci_low > 0
                else "SIGNAL WORSE (CI excludes zero)" if r.ci_high < 0
                else "NO DETECTABLE EDGE (CI straddles zero)"
            )
            print(f"  {int(r.hold_bars):>3d} bars {r.comparison:<28s} "
                  f"{r.mean_diff:>+7.2f}%  [{r.ci_low:>+7.2f}, {r.ci_high:>+7.2f}]  -> {verdict}")

    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()
    print("Controls are now date-matched, so signals and controls face the")
    print("same market conditions. If the confidence interval straddles zero,")
    print("the entry rule has no edge that this data can distinguish from")
    print("chance. These are overlapping windows, not a tradeable equity")
    print("curve, and they ignore position sizing and capital limits.")


if __name__ == "__main__":
    main()
