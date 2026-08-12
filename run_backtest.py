"""
Run historical replay and generate backtest_signals.csv
"""

from backtest.engine import BacktestEngine

from nse_scanner import (
    load_universe,
    fetch_nifty_daily,
    UNIVERSE_CSV,
)

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

engine = BacktestEngine(
    symbols=symbols,
    nifty_close=nifty_close,
    timeframe="Monthly",
)

engine.run()

print()
print("Replay finished.")