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
)

# The live scanner runs both timeframes, so the backtest should too —
# testing only one gives a misleading picture of the system.
TIMEFRAMES = ["Weekly", "Monthly"]

print("=" * 70)
print("NIFTYPULSE PRO BACKTEST")
print("=" * 70)

print("Loading universe...")

un = load_universe(UNIVERSE_CSV)

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

# engine.run() writes each timeframe's own pass to SIGNALS_FILE, so the
# second pass would otherwise overwrite the first. Write the combined
# set explicitly at the end.
if all_events:
    combined = pd.DataFrame(all_events)
    combined.to_csv(cfg.SIGNALS_FILE, index=False)
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
