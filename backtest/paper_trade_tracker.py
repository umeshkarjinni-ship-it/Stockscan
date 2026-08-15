"""
paper_trade_tracker.py
=======================
Tracks NiftyPulsePro's daily BUY recommendations forward in time as
simulated ("paper") trades — so you can see how the SPECIFIC stocks that
were actually recommended on a given day went on to perform.

This is different from, and complements, run_backtest.py:

- run_backtest.py / backtest/  -> "Historically, whenever this strategy's
  rules fired a BUY on ANY stock at ANY past date, how did that tend to
  go?" (general track record of the rule set)
- paper_trade_tracker.py       -> "Specifically, how did TODAY'S actual
  recommended picks do afterward?" (forward tracking of live signals)

WHAT IT DOES EACH RUN
----------------------
1. Reads signals/daily_top20_buy_ranked.csv (produced by rank_buy.py) and
   opens a new tracked position for any Symbol/Timeframe pair that isn't
   already open.
2. Marks every currently OPEN position to market using the latest price.
3. Closes a position once it's been held for HOLD_TRADING_DAYS trading
   bars, logging its realized return.
4. Writes/updates:
     signals/paper_trades.csv         (every position, open or closed)
     signals/paper_trade_summary.txt  (win rate / avg return on closed trades)

IMPORTANT — STATE MUST PERSIST
--------------------------------
paper_trades.csv only becomes useful if it survives between runs. If
you're using the GitHub Actions workflow, it commits this file back to
the repo after each run — see .github/workflows/daily-scan.yml. If you
run this locally, just don't delete signals/paper_trades.csv between runs
(it's excluded from .gitignore's blanket signals/* rule for this reason —
see the .gitignore comment near paper_trades.csv).
"""

import os
from datetime import datetime

import pandas as pd

from nse_scanner import fetch_single

SIGNALS_DIR = "signals"
RANKED_BUY_FILE = os.path.join(SIGNALS_DIR, "daily_top20_buy_ranked.csv")
TRACKER_FILE = os.path.join(SIGNALS_DIR, "paper_trades.csv")
SUMMARY_FILE = os.path.join(SIGNALS_DIR, "paper_trade_summary.txt")

# A single-day move this large is more likely a bad/stale price tick,
# thin-liquidity gap, or corporate action (split/bonus) than genuine price
# action — flag it instead of silently trusting it. Doesn't exclude the
# position, just marks it so you know to sanity-check before relying on it.
ANOMALY_1DAY_MOVE_PCT = 15

# How many trading bars to hold a position before force-closing it.
#
# IMPORTANT: this must match backtest/config.py's MAX_HOLD_DAYS, or the
# paper tracker measures a DIFFERENT strategy than the one the backtest
# validated — making the two sets of results non-comparable. This was 20
# while the backtest used 40.
#
# Note the units differ by design: the backtest operates on resampled
# Weekly/Monthly bars, whereas this tracker checks DAILY bars, so 40 here
# is 40 trading days (~2 months) rather than 40 weeks. Treat the live
# numbers as a directional check on signal quality, not an exact replica
# of the backtest.
HOLD_TRADING_DAYS = 40

TRACKER_COLUMNS = [
    "Symbol", "Timeframe", "EntryDate", "EntryPrice", "BuyScore",
    "Status", "LastCheckedDate", "LastPrice", "UnrealizedReturnPct",
    "ExitDate", "ExitPrice", "ReturnPct", "HoldingDays", "ExitReason",
    "Flag",
]


def load_tracker() -> pd.DataFrame:
    if os.path.exists(TRACKER_FILE):
        df = pd.read_csv(TRACKER_FILE)
        for col in TRACKER_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[TRACKER_COLUMNS]
    return pd.DataFrame(columns=TRACKER_COLUMNS)


def open_new_positions(tracker: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(RANKED_BUY_FILE):
        print(f"No {RANKED_BUY_FILE} found — run rank_buy.py first. Skipping new entries.")
        return tracker

    ranked = pd.read_csv(RANKED_BUY_FILE)
    today = datetime.now().strftime("%Y-%m-%d")

    new_rows = []
    for _, r in ranked.iterrows():
        symbol = str(r.get("Symbol", "")).strip()
        timeframe = str(r.get("Timeframe", "")).strip()
        if not symbol:
            continue

        already_open = (
            not tracker.empty
            and (
                (tracker["Symbol"] == symbol)
                & (tracker["Timeframe"] == timeframe)
                & (tracker["Status"] == "OPEN")
            ).any()
        )
        if already_open:
            continue  # don't double-enter a pick that's already being tracked

        price = pd.to_numeric(r.get("Price"), errors="coerce")
        if pd.isna(price):
            continue

        new_rows.append({
            "Symbol": symbol,
            "Timeframe": timeframe,
            "EntryDate": today,
            "EntryPrice": round(float(price), 2),
            "BuyScore": r.get("BuyScore", ""),
            "Status": "OPEN",
            "LastCheckedDate": today,
            "LastPrice": round(float(price), 2),
            "UnrealizedReturnPct": 0.0,
            "ExitDate": "",
            "ExitPrice": "",
            "ReturnPct": "",
            "HoldingDays": 0,
            "ExitReason": "",
            "Flag": "",
        })

    if new_rows:
        print(f"Opening {len(new_rows)} new paper position(s).")
        tracker = pd.concat([tracker, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No new picks to open (already tracked, or today's list is empty).")

    return tracker


def update_open_positions(tracker: pd.DataFrame) -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")
    open_mask = tracker["Status"] == "OPEN"

    if not open_mask.any():
        return tracker

    for idx in tracker[open_mask].index:
        symbol = tracker.at[idx, "Symbol"]
        entry_price = float(tracker.at[idx, "EntryPrice"])
        entry_date = pd.Timestamp(tracker.at[idx, "EntryDate"])

        print(f"Checking {symbol}...")
        hist = fetch_single(symbol)
        if hist is None or hist.empty:
            print(f"  Could not fetch {symbol} — leaving position as-is.")
            continue

        hist = hist.sort_index()
        bars_since_entry = hist[hist.index > entry_date]
        if bars_since_entry.empty:
            continue

        last_price = float(bars_since_entry["Close"].iloc[-1])
        holding_days = len(bars_since_entry)
        unrealized = round(((last_price - entry_price) / entry_price) * 100, 2)

        # Anomaly check: look at every individual day-over-day move since
        # entry (not just the cumulative return) — a single suspiciously
        # large jump is a strong sign of a bad/stale price tick, an
        # illiquid thin-volume gap, or an unadjusted stock split/bonus,
        # rather than genuine price action.
        closes = pd.concat([pd.Series([entry_price]), bars_since_entry["Close"]])
        daily_moves = closes.pct_change().dropna() * 100
        max_daily_move = daily_moves.abs().max() if not daily_moves.empty else 0.0

        flag = ""
        if max_daily_move >= ANOMALY_1DAY_MOVE_PCT:
            flag = (f"Flagged: {max_daily_move:+.1f}% single-day move — "
                    f"verify price data before trusting this result")
            print(f"  ⚠ {symbol}: {flag}")

        tracker.at[idx, "LastCheckedDate"] = today
        tracker.at[idx, "LastPrice"] = round(last_price, 2)
        tracker.at[idx, "UnrealizedReturnPct"] = unrealized
        tracker.at[idx, "HoldingDays"] = holding_days
        tracker.at[idx, "Flag"] = flag

        if holding_days >= HOLD_TRADING_DAYS:
            tracker.at[idx, "Status"] = "CLOSED"
            tracker.at[idx, "ExitDate"] = today
            tracker.at[idx, "ExitPrice"] = round(last_price, 2)
            tracker.at[idx, "ReturnPct"] = unrealized
            tracker.at[idx, "ExitReason"] = f"Held {HOLD_TRADING_DAYS} trading days"
            print(f"  Closed {symbol}: {unrealized:+.2f}% after {holding_days} trading days.")
        else:
            print(f"  {symbol}: {unrealized:+.2f}% unrealized, day {holding_days}/{HOLD_TRADING_DAYS}")

    return tracker


def build_summary(tracker: pd.DataFrame) -> str:
    closed = tracker[tracker["Status"] == "CLOSED"].copy()
    open_count = int((tracker["Status"] == "OPEN").sum())
    flagged = tracker[tracker["Flag"].astype(str).str.len() > 0]

    lines = ["=" * 60, "  NIFTYPULSEPRO — DAILY RECOMMENDATION TRACKER", "=" * 60, ""]
    lines.append(f"Generated        : {datetime.now().strftime('%d-%b-%Y %H:%M')}")
    lines.append(f"Open positions   : {open_count}")
    lines.append(f"Closed positions : {len(closed)}")

    if not closed.empty:
        # Exclude flagged (likely bad-data) trades from stats so a single
        # anomalous tick doesn't distort the win rate / average return.
        clean_closed = closed[closed["Flag"].astype(str).str.len() == 0]
        returns = pd.to_numeric(clean_closed["ReturnPct"], errors="coerce").dropna()
        if len(returns) > 0:
            wins = int((returns > 0).sum())
            lines.append(f"Win rate         : {wins}/{len(returns)} ({(wins/len(returns)*100):.1f}%)")
            lines.append(f"Average return   : {returns.mean():+.2f}%")
            lines.append(f"Best trade       : {returns.max():+.2f}%")
            lines.append(f"Worst trade      : {returns.min():+.2f}%")
        if len(clean_closed) < len(closed):
            lines.append(f"(excluded {len(closed) - len(clean_closed)} flagged trade(s) from these stats — see below)")
    else:
        lines.append("")
        lines.append("No closed trades yet — check back once positions reach")
        lines.append(f"the {HOLD_TRADING_DAYS}-trading-day holding period.")

    if not flagged.empty:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  {len(flagged)} POSITION(S) FLAGGED FOR SUSPICIOUS PRICE MOVES")
        lines.append("-" * 60)
        for _, r in flagged.iterrows():
            lines.append(f"  {r['Symbol']} ({r['Timeframe']}): {r['Flag']}")

    lines.append("")
    lines.append("Not financial advice — forward-tracked results only reflect")
    lines.append("this specific rule set and holding period, not any trade you")
    lines.append("actually placed.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    tracker = load_tracker()
    tracker = open_new_positions(tracker)
    tracker = update_open_positions(tracker)

    tracker.to_csv(TRACKER_FILE, index=False)

    summary = build_summary(tracker)
    print("\n" + summary)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nSaved: {TRACKER_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
