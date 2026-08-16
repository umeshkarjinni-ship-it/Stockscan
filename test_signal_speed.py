"""
test_signal_speed.py
======================
Answers, with evidence rather than chart-reading: do FASTER signal
settings actually perform better, or do they just look better on the one
chart you happened to examine?

WHERE THIS CAME FROM
--------------------
On GABRIEL India (Monthly) the current settings sold at Rs 824.60 on
02 Mar 26 — essentially at the low — then bought back at Rs 1,438.80 on
01 Jul 26, after a +74% run. Four months of a ~90% rally captured by
nobody. That is the whipsaw the benchmark testing predicted when it found
the entry rule performed WORSE than a random nearby date.

The obvious response is to make signals faster. The trap is tuning
parameters until GABRIEL looks right, which is overfitting to one stock.
This script instead runs each configuration across the whole universe and
benchmarks it against date-matched controls, the same way
test_entry_hypothesis.py does.

WHAT IT TESTS
-------------
Each configuration varies the volatility stop and which confirmations are
required. Speed is measured directly (signals per symbol, and how far
price has already moved from the recent low at entry), alongside whether
the signals beat their controls.

SUCCESS BAR — set before looking, as always
-------------------------------------------
A faster configuration is only worth adopting if it BEATS BOTH CONTROLS
on at least 2 of 3 hold periods. Firing earlier is not itself a win: more
signals with no edge just means more trades, more cost, more slippage.

Watch random_control — it must show no edge, or the harness is broken.

USAGE
-----
    python test_signal_speed.py
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
    load_universe,
    resample,
    compute_vstop,
    compute_kama,
    compute_rsi,
    compute_adx,
    compute_obv,
    UNIVERSE_CSV,
    INCLUDE_OTHER_CATEGORY,
)

SIGNALS_DIR = "signals"
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "signal_speed_results.csv")

TIMEFRAME = "Weekly"      # Weekly gives the most signals to measure
RESAMPLE_RULE = "W"
HOLD_PERIODS = [20, 40, 60]
CONTROLS_PER_SIGNAL = 4
DATE_WINDOW_BARS = 26
BOOTSTRAP_ITERS = 2000
SPLIT_DATE = pd.Timestamp("2023-01-01")

RNG = np.random.default_rng(23)

# label, atr_mult, whipsaw_buffer_pct, required confirmations
CONFIGS = [
    ("current_3.0x_all",      3.0, 0.5, {"rsi", "vol", "adx", "obv"}),
    ("faster_2.0x_all",       2.0, 0.0, {"rsi", "vol", "adx", "obv"}),
    ("current_3.0x_no_adxobv",3.0, 0.5, {"rsi", "vol"}),
    ("fast_2.0x_no_adxobv",   2.0, 0.0, {"rsi", "vol"}),
    ("fastest_1.5x_rsi_only", 1.5, 0.0, {"rsi"}),
    ("flip_only_2.0x",        2.0, 0.0, set()),
    ("random_control",        None, None, None),   # harness sanity check
]


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


def build_frame(bars, nifty_close, atr_mult, buffer_pct):
    """
    Recompute indicators for one configuration.

    Only the volatility stop depends on atr_mult / buffer, so the moving
    averages and oscillators are computed once per call rather than being
    shared — clearer, and the cost is small next to the downloads.
    """
    out = compute_vstop(bars, ns.ATR_LEN, atr_mult, buffer_pct)
    out["KAMA_MID"] = compute_kama(out["Close"], ns.KAMA_MID_LEN,
                                   ns.KAMA_FASTEST_SC, ns.KAMA_SLOWEST_SC)
    out["RSI"] = compute_rsi(out["Close"], ns.RSI_LEN)
    out["VOL_MA"] = out["Volume"].rolling(ns.VOL_MA_LEN).mean()
    out["ADX"] = compute_adx(out, ns.ADX_LEN)
    obv = compute_obv(out)
    out["OBV_OK"] = obv > obv.rolling(ns.OBV_MA_LEN).mean()
    return out


def entries_for(out, required):
    """Boolean series: bars where this configuration would enter."""
    flip_up = out["FLIP_UP"] if "FLIP_UP" in out.columns else pd.Series(False, index=out.index)
    cond = flip_up & (out["Close"] > out["KAMA_MID"])
    if "rsi" in required:
        cond &= (out["RSI"] > ns.RSI_BUY_LEVEL) & (out["RSI"] < ns.RSI_OVERBOUGHT)
    if "vol" in required:
        cond &= out["Volume"] > out["VOL_MA"] * ns.VOL_MULT_REQ
    if "adx" in required:
        cond &= out["ADX"] > ns.ADX_THRESHOLD
    if "obv" in required:
        cond &= out["OBV_OK"]
    return cond.fillna(False)


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    uni = load_universe(UNIVERSE_CSV)
    if not INCLUDE_OTHER_CATEGORY and "Category" in uni.columns:
        uni = uni[uni["Category"].str.strip().str.lower() != "other"]
    symbols = uni["Symbol"].dropna().astype(str).str.strip().tolist()
    print(f"Universe: {len(symbols)} symbols  |  timeframe: {TIMEFRAME}")

    print("Fetching NIFTY...")
    nd = fetch_nifty_daily()
    nifty_close = nd["Close"] if nd is not None else None

    print("Downloading price history (once, reused across all configs)...")
    raw_bars = {}
    for n, sym in enumerate(symbols, start=1):
        if n % 50 == 0 or n == 1:
            print(f"  [{n}/{len(symbols)}]", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is not None and len(raw) > 0:
                b = resample(raw.sort_index(), RESAMPLE_RULE)
                if len(b) > 120:
                    raw_bars[sym] = b
        except Exception:
            continue
    print(f"Ready: {len(raw_bars)} symbols\n")

    cached = sorted(raw_bars.keys())
    rows = []

    for label, atr_mult, buffer_pct, required in CONFIGS:
        print(f"=== {label} ===")

        # Build entry masks for this configuration
        frames, masks = {}, {}
        for sym, bars in raw_bars.items():
            try:
                if required is None:      # random control
                    out = build_frame(bars, nifty_close, 3.0, 0.5)
                    mask = pd.Series(RNG.random(len(out)) < 0.02, index=out.index)
                else:
                    out = build_frame(bars, nifty_close, atr_mult, buffer_pct)
                    mask = entries_for(out, required)
                frames[sym] = out
                masks[sym] = mask & ~mask.shift(1, fill_value=False)
            except Exception:
                continue

        total = int(sum(m.sum() for m in masks.values()))
        per_symbol = round(total / max(len(masks), 1), 2)
        print(f"  entries: {total}  ({per_symbol} per symbol)")

        if total < 100:
            print("  too few entries to evaluate — skipping\n")
            continue

        for hold in HOLD_PERIODS:
            sig, ctl_date, ctl_stock, run_up = [], [], [], []

            for sym, mask in masks.items():
                bars = frames[sym]
                for p in np.flatnonzero(mask.values):
                    entry_date = bars.index[p]
                    if entry_date < SPLIT_DATE:
                        continue                      # out-of-sample only
                    r = window_return(bars, p + 1, hold)
                    if r is None:
                        continue
                    sig.append(r)

                    # How far price had ALREADY run from its recent low at
                    # entry — a direct measure of how late the signal is.
                    lo = float(bars["Close"].iloc[max(0, p - 26):p + 1].min())
                    if lo > 0:
                        run_up.append((float(bars["Close"].iloc[p]) - lo) / lo * 100)

                    for _ in range(CONTROLS_PER_SIGNAL):
                        other = cached[int(RNG.integers(0, len(cached)))]
                        ob = frames.get(other)
                        if ob is not None:
                            rr = same_date_control(ob, entry_date, hold)
                            if rr is not None:
                                ctl_date.append(rr)

                    for _ in range(CONTROLS_PER_SIGNAL):
                        off = int(RNG.integers(-DATE_WINDOW_BARS, DATE_WINDOW_BARS + 1))
                        rr = window_return(bars, p + 1 + off, hold)
                        if rr is not None:
                            ctl_stock.append(rr)

            if len(sig) < 30:
                continue

            row = {
                "config": label,
                "hold": hold,
                "n": len(sig),
                "entries_per_symbol": per_symbol,
                "run_up_at_entry": round(float(np.mean(run_up)), 1) if run_up else None,
                "win_rate": round(float((pd.Series(sig) > 0).mean() * 100), 1),
                "mean_return": round(float(np.mean(sig)), 2),
            }
            for name, ctrl in [("vs_date", ctl_date), ("vs_stock", ctl_stock)]:
                bs = bootstrap_diff(sig, ctrl)
                if bs:
                    row[f"{name}_diff"] = bs["diff"]
                    row[f"{name}_ci"] = f"[{bs['lo']}, {bs['hi']}]"
                    row[f"{name}_sig"] = ("BETTER" if bs["lo"] > 0
                                          else "WORSE" if bs["hi"] < 0 else "none")
            rows.append(row)
            print(f"  hold={hold:>3d}  n={row['n']:>5d}  runup={row['run_up_at_entry']}%  "
                  f"mean={row['mean_return']:>+7.2f}%  "
                  f"vs_date={row.get('vs_date_sig','-'):<7s} vs_stock={row.get('vs_stock_sig','-')}")
        print()

    if not rows:
        print("No configuration produced enough entries.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print("=" * 105)
    print("  SIGNAL SPEED COMPARISON  (out-of-sample: 2023+ only)")
    print("=" * 105)
    print(out.to_string(index=False))
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()

    print("SPEED vs EDGE")
    print("-" * 105)
    print(f"{'config':<26s} {'entries/sym':>11s} {'run-up at entry':>16s} {'vs_date':>9s} {'vs_stock':>9s}")
    for label, *_ in CONFIGS:
        h = out[out.config == label]
        if h.empty:
            continue
        d = int((h.get("vs_date_sig") == "BETTER").sum())
        s = int((h.get("vs_stock_sig") == "BETTER").sum())
        print(f"{label:<26s} {h.entries_per_symbol.iloc[0]:>11.2f} "
              f"{str(h.run_up_at_entry.iloc[0]) + '%':>16s} {str(d) + '/3':>9s} {str(s) + '/3':>9s}")

    print()
    print("HOW TO READ THIS")
    print("  run-up at entry = how far price had ALREADY moved off its recent")
    print("    low when the signal fired. Lower is earlier. This is the number")
    print("    that captures the GABRIEL problem directly.")
    print()
    print("  A faster config is only worth adopting if it beats BOTH controls")
    print("  on 2 of 3 holds. Firing earlier while showing no edge just means")
    print("  more trades, more brokerage, more slippage — for nothing.")
    print()
    print("  If NO config beats the controls, that is the honest answer: the")
    print("  problem is not the speed setting, it is that entry timing on")
    print("  price data alone does not predict returns here.")


if __name__ == "__main__":
    main()
