"""
check_harness_gate.py
=====================
Checks whether the control machinery is biased, by looking at how the
random_control hypothesis behaved.

WHY THE FIRST VERSION OF THIS FILE WAS WRONG
--------------------------------------------
It demanded that random_control read 'no edge' in EVERY cell. That is the
wrong test, and it fails good runs.

random_control is evaluated across 3 eras x 3 holds x 2 control types = 18
cells. Each uses a 95% confidence interval, so each has a 5% chance of
flagging on pure noise BY CONSTRUCTION. The chance that all 18 come back
clean is only 0.95^18 = 40%. An all-clean requirement therefore fails a
perfectly unbiased harness three runs in five.

On the first monthly run it flagged 2 of 18 cells. P(>=2 of 18) = 0.23 —
ordinary noise. Worse, the two flags pointed in OPPOSITE directions
(one BACKWARDS, one SELL EDGE), which is the signature of chance. Real
bias is directional: it pushes every affected cell the same way.

WHAT THIS VERSION TESTS
-----------------------
  1. COUNT     Are there materially more flags than 5% of cells?
               Binomial tail test. Fails only if p < 0.05.

  2. DIRECTION Are 3+ flags all pointing the same way? That is bias even
               when the count alone would pass, so it fails independently.

Both must pass. Exits 1 on failure so the workflow step goes red.

    python check_harness_gate.py
    python check_harness_gate.py signals/sell_validation_weekly.csv

CAVEAT ON THE COUNT TEST
------------------------
The 18 cells are not independent — 2023+ is a subset of 2021+, and the
three hold periods overlap heavily. The binomial p-value is therefore
approximate and somewhat optimistic. It is a screen for gross bias, not a
precise significance test. The direction check is the more reliable of
the two.
"""

import glob
import sys
from math import comb

import pandas as pd

# Printed in the header so a stale deploy is obvious at a glance. v1 was
# the all-or-nothing gate that failed good runs; if a job log shows
# anything other than the current version, the file was not updated.
GATE_VERSION = "v2 (binomial + directional)"

VERDICT_COLS = ["vs_date_verdict", "vs_stock_verdict"]
CLEAN = "no edge"

ALPHA = 0.05               # CI level the verdicts were computed at
COUNT_P_THRESHOLD = 0.05   # fail if the flag count is this unlikely under noise
DIRECTIONAL_MIN = 3        # this many same-direction flags is bias


def binom_tail(k, m, p=ALPHA):
    """P(X >= k) for X ~ Binomial(m, p)."""
    if m == 0:
        return 1.0
    return sum(comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(k, m + 1))


def check(path):
    df = pd.read_csv(path)
    rc = df[df["hypothesis"] == "random_control"]

    print(f"\n{path}")
    print("-" * 66)

    if rc.empty:
        print("  No random_control rows — the harness cannot be verified.")
        print("  Treat every result in this file as unvalidated.")
        return False

    missing = [c for c in VERDICT_COLS if c not in rc.columns]
    if missing:
        print(f"  GATE FAILED — missing column(s): {', '.join(missing)}")
        return False

    show = [c for c in ["era", "hold", "n"] + VERDICT_COLS if c in rc.columns]
    print(rc[show].to_string(index=False))

    flags = [
        (r["era"], r["hold"], col, r[col])
        for _, r in rc.iterrows()
        for col in VERDICT_COLS
        if str(r[col]) != CLEAN
    ]
    m = len(rc) * len(VERDICT_COLS)
    k = len(flags)
    expected = m * ALPHA

    print(f"\n  cells tested        : {m}")
    print(f"  flagged             : {k}   (expected ~{expected:.1f} from noise alone)")

    if flags:
        print("  flagged cells       :")
        for era, hold, col, verdict in flags:
            print(f"      {era:<22s} hold={hold:<3} {col:<17s} {verdict}")

    # --- Test 1: count -------------------------------------------------
    p_count = binom_tail(k, m)
    count_ok = p_count >= COUNT_P_THRESHOLD
    print(f"\n  [1] count test      : P(>={k} of {m} by chance) = {p_count:.3f} "
          f"-> {'PASS' if count_ok else 'FAIL'}")

    # --- Test 2: direction ---------------------------------------------
    directions = {}
    for _, _, _, verdict in flags:
        directions[verdict] = directions.get(verdict, 0) + 1
    worst = max(directions.values()) if directions else 0
    dir_ok = worst < DIRECTIONAL_MIN

    if flags:
        summary = ", ".join(f"{v} x{n}" for v, n in sorted(directions.items()))
        print(f"  [2] direction test  : {summary} "
              f"-> {'PASS' if dir_ok else 'FAIL'}")
        if len(directions) > 1:
            print("      (flags point in OPPOSITE directions — the signature of "
                  "noise, not bias)")
    else:
        print("  [2] direction test  : no flags -> PASS")

    if count_ok and dir_ok:
        print("\n  GATE PASSED — no evidence of bias in the controls.")
        if k:
            print(f"  {k} scattered flag(s) at a {int((1 - ALPHA) * 100)}% CI is what "
                  "chance produces. Not a defect.")
        return True

    print("\n  GATE FAILED — random entry shows a systematic edge.")
    if not count_ok:
        print("  Too many cells flagged to explain as noise.")
    if not dir_ok:
        print(f"  {worst} flags all point the same way — that is directional bias.")
    print("  Ignore every other result in this run until the controls are fixed.")
    return False


def main():
    paths = sys.argv[1:] or sorted(glob.glob("signals/sell_validation_*.csv"))
    if not paths:
        print("No sell_validation_*.csv found — the run produced no output.")
        return 0

    print("=" * 66)
    print(f" HARNESS GATE {GATE_VERSION} — is random entry showing a systematic edge?")
    print("=" * 66)

    results = [check(p) for p in paths]

    print()
    print("=" * 66)
    print(" REMINDER — the same arithmetic applies to the RESULTS")
    print("=" * 66)
    print(" A full run evaluates 5 hypotheses x 3 eras x 3 holds x 2 controls")
    print(" = 90 cells. At a 95% CI, roughly 4-5 will flag on noise alone.")
    print(" Do NOT read a single flagged cell as a finding. Look for the same")
    print(" direction across multiple holds and both out-of-sample eras.")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
