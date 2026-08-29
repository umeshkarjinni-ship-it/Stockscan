"""
controls.py
===========
Control constructions for the signal validation harness, plus the
self-test that should have existed from the start.

WHAT WENT WRONG
---------------
The same-stock nearby-date control drew entries from offsets -W..+W around
the signal bar. For a signal that fires at a local price minimum — which
is what a VStop FLIP_DOWN and an RSI<45 pullback both do by construction —
the NEGATIVE half of that window sits on the decline leading into the
signal. Those control entries buy in higher and exit earlier, so they are
handicapped by the signal's own definition rather than beaten by it.

On a pure geometric random walk with no edge whatsoever, that control
reports +6.04% [+5.68, +6.42] for local-minimum signals. It is not
measuring edge. It is measuring the arithmetic of conditioning.

WHY THE RANDOM_CONTROL GATE MISSED IT
-------------------------------------
random_control fires on uniformly random bars. It has no conditioning, so
it has no artifact — it reads clean on every control, in both eras. A
uniform random control can verify that a control is unbiased FOR UNIFORM
SIGNALS. It structurally cannot verify it for signals that fire at
extremes. That is the hole this file closes.

THE THREE CONTROLS
------------------
  same_date       Different stock, same date. Isolates SELECTION.
                  Clean in the self-test. Unchanged from the original.

  forward_only    Same stock, offsets +1..+W only. Isolates TIMING with
                  the contaminated pre-decline half removed. Requires
                  matched validity (see below) or end-of-sample truncation
                  reintroduces a bias of its own.

  peer            Different stock on the same date, matched on how far it
                  sits below its own 52-week high. Isolates whether the
                  signal adds anything BEYOND "this stock is beaten down".
                  The strongest of the three, and the one that most
                  directly tests what a pullback rule claims.

MATCHED VALIDITY
----------------
A signal is only admitted if sig_pos + window + hold < len(bars), so every
forward offset is available for it. Without this the forward control
silently rejects more entries near the end of the series than the signal
does, biasing controls toward earlier — and in a rising market, better —
periods. That is a smaller version of the same class of bug.

SELF-TEST
---------
    python controls.py --self-test

Generates a synthetic panel under a known null (geometric random walk, no
edge, no mean reversion), fires local-minimum signals, and reports what
each control says. Anything that flags is measuring its own construction.
Then repeats with a known edge injected, to confirm the surviving controls
can still detect one.

Run this after ANY change to control logic.
"""

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------

def window_return(bars, pos, hold, cost_pct=0.0):
    """Enter at Open[pos], exit at Close[pos+hold]. None if out of range."""
    if pos < 0 or pos + hold >= len(bars):
        return None
    entry = float(bars.iloc[pos]["Open"])
    if entry <= 0:
        return None
    exit_ = float(bars.iloc[pos + hold]["Close"])
    return (exit_ / entry - 1.0) * 100.0 - cost_pct


def signal_admissible(bars, sig_pos, hold, window):
    """
    Matched validity: admit the signal only if every forward control
    offset is also evaluable. Keeps signal and control facing identical
    end-of-sample truncation.
    """
    return sig_pos + window + hold < len(bars)


def same_date_control(other_bars, entry_date, hold, cost_pct=0.0):
    """Different stock, same date. Returns None if it wasn't trading then."""
    if len(other_bars) == 0:
        return None
    if entry_date < other_bars.index[0] or entry_date > other_bars.index[-1]:
        return None
    pos = other_bars.index.searchsorted(entry_date, side="right")
    if pos <= 0:
        return None
    return window_return(other_bars, pos, hold, cost_pct)


def forward_only_control(bars, sig_pos, hold, window, rng, cost_pct=0.0):
    """
    Same stock, a random date AFTER the signal (+1..+window).

    The negative half is dropped deliberately — it is the contaminated
    half. See module docstring.
    """
    off = int(rng.integers(1, window + 1))
    return window_return(bars, sig_pos + off, hold, cost_pct)


def peer_control(frames, drawdowns, symbols, entry_date, signal_dd,
                 hold, rng, tolerance=5.0, max_tries=25, cost_pct=0.0):
    """
    Different stock on the same date, matched on distance below its own
    52-week high to within `tolerance` percentage points.

    Answers the question a pullback rule actually makes: among stocks that
    are equally beaten down today, does this flag pick better ones?
    """
    if signal_dd is None or not np.isfinite(signal_dd):
        return None
    for _ in range(max_tries):
        other = symbols[int(rng.integers(0, len(symbols)))]
        dd_series = drawdowns.get(other)
        if dd_series is None or entry_date not in dd_series.index:
            continue
        peer_dd = dd_series.loc[entry_date]
        if not np.isfinite(peer_dd) or abs(peer_dd - signal_dd) > tolerance:
            continue
        r = same_date_control(frames[other], entry_date, hold, cost_pct)
        if r is not None:
            return r
    return None


def pct_off_high_series(bars, lookback):
    """Percent below the trailing `lookback`-bar high, per bar."""
    high = bars["Close"].rolling(lookback, min_periods=max(4, lookback // 4)).max()
    return (bars["Close"] / high - 1.0) * 100.0


def bootstrap_diff(a, b, rng, iters=2000, min_n=30):
    a = np.asarray(pd.Series(a).dropna())
    b = np.asarray(pd.Series(b).dropna())
    if len(a) < min_n or len(b) < min_n:
        return None
    diffs = np.empty(iters)
    for i in range(iters):
        diffs[i] = (rng.choice(a, len(a), True).mean()
                    - rng.choice(b, len(b), True).mean())
    return {
        "diff": float(a.mean() - b.mean()),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
        "n": len(a),
    }


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def _synthetic_panel(rng, n_stocks, n_bars, drift, vol, edge_pct=0.0,
                     look=8, drop=0.93):
    """
    Geometric random walk. If edge_pct > 0, a genuine post-signal
    outperformance is injected as a sustained drift bump after each local
    minimum, so the self-test can measure power as well as bias.
    """
    logret = rng.normal(drift, vol, (n_stocks, n_bars))
    px = 100 * np.exp(np.cumsum(logret, axis=1))

    sig_positions = []
    for i in range(n_stocks):
        s = px[i]
        pos = [p for p in range(look, n_bars - 1)
               if s[p] == s[p - look:p + 1].min() and s[p] < s[p - look] * drop]
        sig_positions.append(pos)

    if edge_pct > 0:
        # A TIMING edge, injected as a short burst immediately after the
        # signal rather than a sustained drift.
        #
        # A sustained bump is the wrong null to test against: the
        # forward_only control enters INSIDE the bumped region and holds
        # through it, so it collects as much of the bump as the signal
        # does. The first version of this function did exactly that and
        # reported -16% for a +8% edge — the control looked better than
        # the signal, which is an artifact of the injection, not of the
        # control.
        #
        # A burst confined to BURST bars means entering at the signal
        # captures all of it and entering `off` bars later captures only
        # what remains. That is what a genuine timing edge looks like.
        BURST = 12
        for i in range(n_stocks):
            bump = np.zeros(n_bars)
            for p in sig_positions[i]:
                end = min(p + 1 + BURST, n_bars)
                bump[p + 1:end] += edge_pct / 100.0 / BURST
            px[i] = 100 * np.exp(np.cumsum(logret[i] + bump))
        sig_positions = []
        for i in range(n_stocks):
            s = px[i]
            pos = [p for p in range(look, n_bars - 1)
                   if s[p] == s[p - look:p + 1].min() and s[p] < s[p - look] * drop]
            sig_positions.append(pos)

    idx = pd.date_range("2010-01-03", periods=n_bars, freq="W")
    frames, drawdowns = {}, {}
    for i in range(n_stocks):
        sym = f"S{i:03d}"
        df = pd.DataFrame({"Open": px[i], "Close": px[i]}, index=idx)
        frames[sym] = df
        drawdowns[sym] = pct_off_high_series(df, 52)
    return frames, drawdowns, sig_positions


def run_self_test(n_stocks=250, n_bars=520, hold=40, window=26,
                  seed=17, edge_pct=0.0, verbose=True):
    rng = np.random.default_rng(seed)
    frames, drawdowns, sig_positions = _synthetic_panel(
        rng, n_stocks, n_bars, drift=0.0015, vol=0.030, edge_pct=edge_pct
    )
    symbols = sorted(frames.keys())

    sig, c_date, c_fwd, c_peer, c_legacy = [], [], [], [], []

    for i, sym in enumerate(symbols):
        bars = frames[sym]
        dd = drawdowns[sym]
        for p in sig_positions[i]:
            sig_pos = p + 1
            if not signal_admissible(bars, sig_pos, hold, window):
                continue
            r = window_return(bars, sig_pos, hold)
            if r is None:
                continue
            sig.append(r)
            entry_date = bars.index[sig_pos]
            signal_dd = dd.iloc[sig_pos]

            for _ in range(3):
                other = symbols[int(rng.integers(0, len(symbols)))]
                rr = same_date_control(frames[other], entry_date, hold)
                if rr is not None:
                    c_date.append(rr)

                rr = forward_only_control(bars, sig_pos, hold, window, rng)
                if rr is not None:
                    c_fwd.append(rr)

                rr = peer_control(frames, drawdowns, symbols, entry_date,
                                  signal_dd, hold, rng)
                if rr is not None:
                    c_peer.append(rr)

                # The original control, kept so the self-test reproduces
                # the failure rather than just asserting it.
                off = int(rng.integers(-window, window + 1))
                rr = window_return(bars, sig_pos + off, hold)
                if rr is not None:
                    c_legacy.append(rr)

    results = {}
    for label, ctrl in [("same_date", c_date), ("forward_only", c_fwd),
                        ("peer", c_peer), ("legacy_+/-W", c_legacy)]:
        results[label] = bootstrap_diff(sig, ctrl, rng)

    if verbose:
        truth = f"injected edge = +{edge_pct:.1f}%" if edge_pct else "NO edge (null)"
        print(f"\n  Ground truth: {truth}   signals={len(sig):,}")
        print(f"  {'control':<14}{'n_ctrl':>8}{'diff':>9}{'95% CI':>20}   verdict")
        for label, bs in results.items():
            if bs is None:
                print(f"  {label:<14}{'—':>8}   insufficient n")
                continue
            flags = bs["lo"] > 0 or bs["hi"] < 0
            if edge_pct:
                verdict = "detects edge" if flags else "MISSES edge"
            else:
                verdict = "SPURIOUS FLAG" if flags else "clean"
            n_ctrl = {"same_date": len(c_date), "forward_only": len(c_fwd),
                      "peer": len(c_peer), "legacy_+/-W": len(c_legacy)}[label]
            print(f"  {label:<14}{n_ctrl:>8,}{bs['diff']:>+9.2f}"
                  f"   [{bs['lo']:>+6.2f}, {bs['hi']:>+6.2f}]   {verdict}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if not args.self_test:
        print(__doc__)
        return

    print("=" * 78)
    print(" CONTROL SELF-TEST — signals fire at local minima, as FLIP_DOWN does")
    print("=" * 78)

    print("\nPART 1 — bias. Every control should read 'clean'.")
    for s in range(args.seeds):
        run_self_test(seed=17 + s, edge_pct=0.0)

    print("\n\nPART 2 — power. Surviving controls should detect a real edge.")
    for s in range(args.seeds):
        run_self_test(seed=17 + s, edge_pct=8.0)

    print("\n" + "=" * 78)
    print(" A control that flags in PART 1 is measuring its own construction.")
    print(" A control that misses in PART 2 is too weak to be worth running.")
    print(" Only controls that are clean in 1 AND detect in 2 should be used.")
    print("=" * 78)


if __name__ == "__main__":
    main()
