"""
check_harness_gate.py
=====================
Reads signals/sell_validation_*.csv and checks the one thing that must be
true before any other row in those files means anything:

    random_control must read 'no edge' in every era against both controls.

If random entry shows an edge, the controls are biased and every other
verdict in the run is measuring the bias rather than the rule.

Exits 1 if the gate fails, so the workflow step goes red instead of
quietly passing a result nobody should trust.

    python check_harness_gate.py
    python check_harness_gate.py signals/sell_validation_weekly.csv
"""

import glob
import sys

import pandas as pd

VERDICT_COLS = ["vs_date_verdict", "vs_stock_verdict"]


def check(path):
    df = pd.read_csv(path)
    rc = df[df["hypothesis"] == "random_control"]

    print(f"\n{path}")
    if rc.empty:
        print("  No random_control rows — cannot verify the harness.")
        print("  Treat every result in this file as unvalidated.")
        return False

    cols = [c for c in ["era", "hold", "n"] + VERDICT_COLS if c in rc.columns]
    print(rc[cols].to_string(index=False))

    missing = [c for c in VERDICT_COLS if c not in rc.columns]
    if missing:
        print(f"\n  GATE FAILED — missing column(s): {', '.join(missing)}")
        return False

    bad = rc[
        (rc["vs_date_verdict"] != "no edge") | (rc["vs_stock_verdict"] != "no edge")
    ]
    if bad.empty:
        print("\n  GATE PASSED — random entry shows no edge. Controls are unbiased.")
        return True

    print(f"\n  GATE FAILED — {len(bad)} row(s) show an edge on RANDOM entry:")
    print(bad[cols].to_string(index=False))
    print("\n  The harness is biased. Ignore every other result in this run.")
    return False


def main():
    paths = sys.argv[1:] or sorted(glob.glob("signals/sell_validation_*.csv"))
    if not paths:
        print("No sell_validation_*.csv found — the run did not produce output.")
        return 0  # nothing to gate; the run step itself will have failed

    print("=" * 66)
    print(" HARNESS GATE — random_control must read 'no edge' everywhere")
    print("=" * 66)

    results = [check(p) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
