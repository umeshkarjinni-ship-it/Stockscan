"""
run_backtest.py
================
Runs the historical signal replay and generates backtest_signals.csv.

IMPORTANT — replays BOTH timeframes.
--------------------------------------
This previously hardcoded timeframe="Monthly", which meant the Weekly
half of the strategy was never backtested at all. It now replays Weekly
AND Monthly and combines the results, so the backtest reflects the same
two timeframes the live scanner actually trades.

Set TIMEFRAMES below to ["Monthly"] or ["Weekly"] if you specifically
want to isolate one of them.
"""

from dataclasses import asdict

import pandas as pd

from backtest import config as cfg
from backtest.engine import BacktestEngine

from nse_scanner import (
    load_universe,
    fetch_nifty_daily,
    UNIVERSE_CSV,
    INCLUDE_OTHER_CATEGORY,
)

# The live scanner runs both timeframes, so the backtest should too —
# testing only one gives a misleading picture of the system.
TIMEFRAMES = ["Weekly", "Monthly"]

print("=" * 70)
print("NIFTYPULSE PRO BACKTEST")
print("=" * 70)

print("Loading universe...")

un = load_universe(UNIVERSE_CSV)

# Apply the same "Other" category filter the live scanner uses.
#
# NOTE: that filter lives inside nse_scanner.scan(), NOT in
# load_universe(), so the backtest previously loaded all ~2,075 symbols
# regardless of INCLUDE_OTHER_CATEGORY — meaning the backtest and the
# live scan were testing different universes. Mirroring it here keeps
# them consistent.
#
# "Other" = micro-caps, recent listings, and thin-liquidity names. Beyond
# being slow to backtest, they're the main source of corrupt price data
# (unadjusted splits show up as implausible one-day moves) and generally
# aren't executable at size anyway.
if not INCLUDE_OTHER_CATEGORY and "Category" in un.columns:
    before = len(un)
    un = un[un["Category"].str.strip().str.lower() != "other"]
    skipped = before - len(un)
    if skipped:
        print(f"Skipping {skipped} 'Other' category symbols "
              f"(set INCLUDE_OTHER_CATEGORY=True in nse_scanner.py to include).")

symbols = (
    un["Symbol"]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

# ---------- DEVELOPMENT MODE ----------
DEV_MODE = True
DEV_SYMBOLS = 50

if DEV_MODE:
    symbols = symbols[:DEV_SYMBOLS]
    print(f"Development Mode: Running first {len(symbols)} symbols")
# --------------------------------------
print(f"Loaded {len(symbols)} symbols")

print("Downloading NIFTY...")

nifty = fetch_nifty_daily()

if nifty is None or nifty.empty:
    raise RuntimeError("Unable to download NIFTY")

nifty_close = nifty["Close"]

print(f"NIFTY bars : {len(nifty_close)}")

all_events = []

for tf in TIMEFRAMES:
    print()
    print("-" * 70)
    print(f"REPLAYING {tf.upper()} TIMEFRAME")
    print("-" * 70)

    engine = BacktestEngine(
        symbols=symbols,
        nifty_close=nifty_close,
        timeframe=tf,
    )

    engine.run()

    if engine.events:
        all_events.extend([asdict(e) for e in engine.events])
        print(f"{tf}: {len(engine.events)} signals recorded.")
    else:
        print(f"{tf}: no signals recorded.")

    # Save after EVERY timeframe rather than only at the very end. A
    # full-universe two-timeframe run can push against GitHub's 6-hour
    # job limit, and without this an overrun would discard hours of
    # completed work. Writing here means a killed job still leaves usable
    # partial results behind.
    if all_events:
        pd.DataFrame(all_events).to_csv(cfg.SIGNALS_FILE, index=False)
        print(f"  (saved {len(all_events)} signals so far to {cfg.SIGNALS_FILE})")

# engine.run() writes each timeframe's own pass to SIGNALS_FILE, so the
# second pass would otherwise overwrite the first. The combined set has
# already been written above; this just reports the final tally.
if all_events:
    combined = pd.DataFrame(all_events)
    print()
    print(f"Combined {len(combined)} signals across {len(TIMEFRAMES)} timeframe(s)")
    print(f"Saved to {cfg.SIGNALS_FILE}")
    print()
    print(combined["timeframe"].value_counts().to_string())
else:
    print()
    print("No signals generated across any timeframe.")

print()
print("Replay finished.")
