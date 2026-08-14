"""
check_conflicting_signals.py
==============================
Flags any symbol that shows up as a BUY signal on one timeframe AND a
SELL signal on another timeframe on the same day (e.g. Monthly BUY +
Weekly SELL). Neither signal is "wrong" — different timeframes are
allowed to disagree — but silently acting on the BUY without knowing a
shorter timeframe is bearish is exactly the kind of thing that should be
visible before you commit to a trade, paper or real.

Reads:  signals/daily_top20_buy_ranked.csv
        signals/daily_top20_sell.csv
Writes: signals/conflicting_signals.csv (empty file with header if none found)

USAGE
-----
    python check_conflicting_signals.py

Runs safely even if either input file is missing or empty.
"""

import os

import pandas as pd

SIGNALS_DIR = "signals"
BUY_FILE = os.path.join(SIGNALS_DIR, "daily_top20_buy_ranked.csv")
SELL_FILE = os.path.join(SIGNALS_DIR, "daily_top20_sell.csv")
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "conflicting_signals.csv")


def read_csv_safe(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                return df
        except Exception:
            pass
    return None


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    buy_df = read_csv_safe(BUY_FILE)
    sell_df = read_csv_safe(SELL_FILE)

    if buy_df is None or sell_df is None:
        print("Nothing to compare — one or both signal files are missing/empty today.")
        pd.DataFrame(columns=[
            "Symbol", "BuyTimeframe", "BuyPrice", "SellTimeframe", "SellPrice",
        ]).to_csv(OUTPUT_FILE, index=False)
        return

    buy_symbols = set(buy_df["Symbol"])
    sell_symbols = set(sell_df["Symbol"])
    overlap = buy_symbols & sell_symbols

    if not overlap:
        print("No conflicting signals today — clean.")
        pd.DataFrame(columns=[
            "Symbol", "BuyTimeframe", "BuyPrice", "SellTimeframe", "SellPrice",
        ]).to_csv(OUTPUT_FILE, index=False)
        return

    rows = []
    for symbol in sorted(overlap):
        buy_rows = buy_df[buy_df["Symbol"] == symbol]
        sell_rows = sell_df[sell_df["Symbol"] == symbol]
        for _, b in buy_rows.iterrows():
            for _, s in sell_rows.iterrows():
                rows.append({
                    "Symbol": symbol,
                    "BuyTimeframe": b.get("Timeframe", ""),
                    "BuyPrice": b.get("Price", ""),
                    "SellTimeframe": s.get("Timeframe", ""),
                    "SellPrice": s.get("Price", ""),
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"  {len(overlap)} SYMBOL(S) HAVE CONFLICTING SIGNALS TODAY")
    print(f"{'=' * 60}")
    for _, r in out.iterrows():
        print(f"  {r['Symbol']}: BUY on {r['BuyTimeframe']} @ {r['BuyPrice']}  "
              f"vs  SELL on {r['SellTimeframe']} @ {r['SellPrice']}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
