"""
test_conflict_filter.py
=========================
Should a BUY be skipped when the OTHER timeframe is flagging a SELL on
the same stock?

WHERE THIS CAME FROM
--------------------
On 15 Aug 2026 GREAVESCOT appeared as a Monthly BUY at Rs 238.05 while
simultaneously showing a Weekly SELL at Rs 198.85. It then fell to
Rs 194.37 — the worst performer of the day's 13 picks at -18.35%, while
the other seven visible BUYs averaged +8.3%.

The Weekly SELL was right and the Monthly BUY was wrong.

That is n=1 and proves nothing on its own, but the mechanism is
plausible: a shorter timeframe rolling over while a longer one still
reads "up" is the classic early-deterioration pattern, and Monthly
signals carry up to 30 days of staleness so the Weekly view is fresher
by construction.

conflicting_signals.csv already records these cases but nothing acts on
them. This measures whether acting on them would have helped.

WHAT IT COMPARES
----------------
    all_buys            every Weekly BUY  (current behaviour)
    no_conflict         skip BUYs where the Monthly timeframe is bearish
    only_conflict       ONLY the conflicted BUYs

If the filter works, no_conflict should beat all_buys and only_conflict
should be visibly worse. If only_conflict is no worse, GREAVESCOT was
simply an unlucky draw.

NO LOOK-AHEAD
-------------
Both timeframes are computed over full history with causal indicators,
then the Monthly state is read AT the Weekly signal's own bar using
searchsorted on the preceding Monthly bar. Nothing consults a future
Monthly close.

SUCCESS BAR — set before looking
--------------------------------
The filter is worth adopting only if no_conflict beats BOTH controls on
at least 2 of 3 hold periods where all_buys does not. Removing signals
is not free: fewer trades means less diversification across signals.

USAGE
-----
    python test_conflict_filter.py
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
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "conflict_filter_results.csv")

HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 4
DATE_WINDOW_BARS = 26
BOOTSTRAP_ITERS = 2000
SPLIT_DATE = pd.Timestamp("2023-01-01")     # out-of-sample only

RNG = np.random.default_rng(47)


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


def monthly_bearish_at(monthly, when):
    """
    Is the Monthly timeframe bearish as of `when`?

    Reads the last COMPLETED monthly bar at or before the weekly signal
    date, so no future monthly close is consulted. Returns None when
    there is no monthly bar yet (recent listings).
    """
    if monthly is None or monthly.empty or "UPTREND" not in monthly.columns:
        return None
    pos = monthly.index.searchsorted(when, side="right") - 1
    if pos < 0:
        return None
    val = monthly["UPTREND"].iloc[pos]
    if pd.isna(val):
        return None
    return not bool(val)


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

    print("Building Weekly + Monthly frames (downloaded once per symbol)...")
    weekly, monthly = {}, {}
    for n, sym in enumerate(symbols, start=1):
        if n % 50 == 0 or n == 1:
            print(f"  [{n}/{len(symbols)}]", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is None or len(raw) == 0:
                continue
            raw = raw.sort_index()
            w = compute_signals(resample(raw, "W"), nifty_close)
            if w is None or w.empty or "BUY_SIGNAL_CORE" not in w.columns:
                continue
            weekly[sym] = w
            m = compute_signals(resample(raw, "ME"), nifty_close)
            monthly[sym] = m if (m is not None and not m.empty) else None
        except Exception:
            continue
    print(f"Ready: {len(weekly)} symbols "
          f"({sum(1 for v in monthly.values() if v is not None)} with monthly data)\n")

    cached = sorted(weekly.keys())

    # Weekly BUY positions, split by whether Monthly was bearish at the time.
    plain, conflicted = {}, {}
    n_plain = n_conf = n_unknown = 0
    for sym, w in weekly.items():
        mask = w["BUY_SIGNAL_CORE"].fillna(False).astype(bool)
        new = mask & ~mask.shift(1, fill_value=False)
        pl, cf = [], []
        for p in np.flatnonzero(new.values):
            bear = monthly_bearish_at(monthly.get(sym), w.index[p])
            if bear is None:
                n_unknown += 1
                continue
            (cf if bear else pl).append(p)
        plain[sym], conflicted[sym] = pl, cf
        n_plain += len(pl); n_conf += len(cf)

    print(f"Weekly BUYs: {n_plain} with Monthly bullish, {n_conf} conflicted "
          f"(Monthly bearish), {n_unknown} undetermined\n")
    if n_conf < 50:
        print("Too few conflicted signals to measure — the comparison would be noise.")
        return

    VARIANTS = [
        ("all_buys      (current)", lambda s: plain[s] + conflicted[s]),
        ("no_conflict   (filtered)", lambda s: plain[s]),
        ("only_conflict (the skipped ones)", lambda s: conflicted[s]),
    ]

    rows = []
    for label, pick in VARIANTS:
        print(f"=== {label} ===")
        for hold in HOLD_PERIODS:
            sig, ctl_date, ctl_stock = [], [], []
            for sym in weekly:
                w = weekly[sym]
                for p in pick(sym):
                    when = w.index[p]
                    if when < SPLIT_DATE:
                        continue
                    r = window_return(w, p + 1, hold)
                    if r is None:
                        continue
                    sig.append(r)
                    for _ in range(CONTROLS_PER_SIGNAL):
                        other = cached[int(RNG.integers(0, len(cached)))]
                        rr = same_date_control(weekly[other], when, hold)
                        if rr is not None:
                            ctl_date.append(rr)
                    for _ in range(CONTROLS_PER_SIGNAL):
                        off = int(RNG.integers(-DATE_WINDOW_BARS, DATE_WINDOW_BARS + 1))
                        rr = window_return(w, p + 1 + off, hold)
                        if rr is not None:
                            ctl_stock.append(rr)

            if len(sig) < 30:
                print(f"  hold={hold:>3d}  only {len(sig)} out-of-sample entries — skipped")
                continue

            row = {"variant": label, "hold": hold, "n": len(sig),
                   "win_rate": round(float((pd.Series(sig) > 0).mean() * 100), 1),
                   "mean_return": round(float(np.mean(sig)), 2)}
            for name, ctrl in [("vs_date", ctl_date), ("vs_stock", ctl_stock)]:
                bs = bootstrap_diff(sig, ctrl)
                if bs:
                    row[f"{name}_diff"] = bs["diff"]
                    row[f"{name}_ci"] = f"[{bs['lo']}, {bs['hi']}]"
                    row[f"{name}_sig"] = ("BETTER" if bs["lo"] > 0
                                          else "WORSE" if bs["hi"] < 0 else "none")
            rows.append(row)
            print(f"  hold={hold:>3d}  n={row['n']:>5d}  win={row['win_rate']:>5.1f}%  "
                  f"mean={row['mean_return']:>+7.2f}%  "
                  f"vs_date={row.get('vs_date_sig','-'):<7s} vs_stock={row.get('vs_stock_sig','-')}")
        print()

    if not rows:
        print("Not enough data to evaluate.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print("=" * 100)
    print("  CONFLICT FILTER — OUT-OF-SAMPLE (2023+)")
    print("=" * 100)
    print(out.to_string(index=False))
    print(f"\nSaved to {OUTPUT_FILE}\n")

    print("SCORECARD")
    print("-" * 100)
    for label, _ in VARIANTS:
        h = out[out.variant == label]
        if h.empty:
            continue
        d = int((h.get("vs_date_sig") == "BETTER").sum())
        s = int((h.get("vs_stock_sig") == "BETTER").sum())
        print(f"  {label:<34s} n={int(h.n.mean()):>5d}  mean={h.mean_return.mean():>+6.2f}%  "
              f"vs_date {d}/3   vs_stock {s}/3")

    print()
    print("HOW TO READ IT")
    print("  The direct comparison is all_buys vs no_conflict. If filtering")
    print("  helps, no_conflict should show a higher mean and better control")
    print("  scores, and only_conflict should be visibly worse.")
    print()
    print("  If only_conflict is NOT worse than the others, GREAVESCOT was an")
    print("  unlucky draw rather than evidence of a pattern, and the filter")
    print("  would remove signals for nothing.")
    print()
    print("  Removing signals has a cost even when neutral: fewer trades means")
    print("  more concentration in whatever remains.")


if __name__ == "__main__":
    main()
