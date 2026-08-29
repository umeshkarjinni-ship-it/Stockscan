"""
summarise_results.py
====================
Prints the consistency table from a sell_validation CSV you already have.

Use this when a run finished before validate_sell_signals.py gained its
built-in summary — the CSV in the artifact holds every number the table is
derived from, so there is no need to spend another 20 minutes re-running.

    python summarise_results.py signals/sell_validation_weekly.csv
    python summarise_results.py            # globs signals/sell_validation_*.csv

No project imports. Pandas only. Run it anywhere the CSV is.

WHY A TABLE INSTEAD OF READING CELLS
------------------------------------
A full run evaluates 5 hypotheses x 3 eras x 3 holds x 2 controls = 90
cells at a 95% CI. Roughly 4-5 flag on noise alone. Reading one flagged
cell as a finding is exactly the error that made the first version of the
harness gate fail good runs.

A real effect appears in MOST out-of-sample cells pointing the SAME way.
Noise appears in one or two, often in both directions at once.

Note the two out-of-sample eras are nested — 2023+ sits inside 2021+ — so
agreement between them is weaker evidence than it looks.
"""

import glob
import sys

import pandas as pd

VERDICT_COLS = ["vs_date_verdict", "vs_stock_verdict"]
MAJORITY = 0.6


def summarise(path):
    out = pd.read_csv(path)
    oos = out[out["era"].astype(str).str.startswith("OUT_OF_SAMPLE")]

    print(f"\n{path}")
    print("=" * 96)

    if oos.empty:
        print("  No OUT_OF_SAMPLE rows found.")
        return

    cols = [c for c in VERDICT_COLS if c in oos.columns]
    if not cols:
        print(f"  Missing verdict columns. Found: {list(out.columns)}")
        return

    print("  CONSISTENCY ACROSS OUT-OF-SAMPLE CELLS  (do not read single cells)")
    print("=" * 96)
    print(f"  {'hypothesis':<26s} {'cells':>6s} {'SELL EDGE':>10s} {'no edge':>9s} "
          f"{'BACKWARDS':>10s}   read")

    for hyp in out["hypothesis"].unique():
        sub = oos[oos["hypothesis"] == hyp]
        verdicts = [str(v) for c in cols for v in sub[c].dropna()]
        if not verdicts:
            continue
        total = len(verdicts)
        n_edge = verdicts.count("SELL EDGE")
        n_none = verdicts.count("no edge")
        n_back = verdicts.count("BACKWARDS")

        if n_edge >= MAJORITY * total:
            read = "consistent SELL EDGE"
        elif n_back >= MAJORITY * total:
            read = "consistent BACKWARDS"
        elif n_none >= MAJORITY * total:
            read = "no edge"
        else:
            read = "mixed — treat as no edge"

        marker = " <-- gate" if hyp == "random_control" else ""
        print(f"  {hyp:<26s} {total:>6d} {n_edge:>10d} {n_none:>9d} {n_back:>10d}   "
              f"{read}{marker}")

    print()
    print("  Verdicts are sign-inverted for SELL: 'SELL EDGE' means the stock")
    print("  UNDERPERFORMED its control after the signal, which is what a working")
    print("  SELL should do. 'BACKWARDS' means it outperformed — the rule sells lows.")

    # Mean returns give a sense of effect size the verdicts alone hide.
    if "mean_return" in oos.columns:
        print()
        print("  Mean forward return by hypothesis (out-of-sample, net of costs):")
        for hyp in out["hypothesis"].unique():
            sub = oos[oos["hypothesis"] == hyp]
            if sub.empty:
                continue
            print(f"    {hyp:<26s} {sub['mean_return'].mean():>+7.2f}%   "
                  f"(n {int(sub['n'].sum()):,} across {len(sub)} era/hold cells)")


def main():
    paths = sys.argv[1:] or sorted(glob.glob("signals/sell_validation_*.csv"))
    if not paths:
        print("No CSV given and none found at signals/sell_validation_*.csv")
        print("Usage: python summarise_results.py <path-to-csv>")
        return 1
    for p in paths:
        summarise(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
