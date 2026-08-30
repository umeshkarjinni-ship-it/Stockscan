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

import numpy as np
import pandas as pd

from nse_scanner import fetch_single

SIGNALS_DIR = "signals"
RANKED_BUY_FILE = os.path.join(SIGNALS_DIR, "daily_top20_buy_ranked.csv")
SELL_FILE = os.path.join(SIGNALS_DIR, "daily_top20_sell.csv")
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
    # Direction is new. Rows written before it existed are BUY — see
    # load_tracker(), which backfills rather than dropping them.
    "Symbol", "Direction", "Timeframe", "EntryDate", "EntryPrice", "SignalBarPrice", "BuyScore",
    "Status", "LastCheckedDate", "LastPrice", "UnrealizedReturnPct", "SignalReturnPct",
    "ExitDate", "ExitPrice", "ReturnPct", "HoldingDays", "ExitReason",
    "Flag",
]

# PriceMovePct vs SignalReturnPct — the distinction matters once SELLs
# are tracked, and conflating them is the easiest way to get this wrong.
#
#   UnrealizedReturnPct / ReturnPct  = what the STOCK did (price move).
#                                      Same arithmetic for every row.
#   SignalReturnPct                  = whether the SIGNAL was right.
#                                      Negated for SELL.
#
# A SELL on a stock that then fell 5% is a price move of -5% and a signal
# return of +5%. Storing only one number would make every correct SELL
# look like a loss in the win rate, or make the price column mean two
# different things depending on the row. So both are stored.
DIRECTIONS = ("BUY", "SELL")


def clean_flag(value) -> str:
    """
    Normalise the Flag column to a plain string.

    pandas reads empty CSV cells as NaN, and str(NaN) == "nan" — a
    non-empty, length-3 string. Left unhandled that made every unflagged
    position look flagged: the summary counted them all as anomalies and
    excluded them from the win-rate stats.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


def load_tracker() -> pd.DataFrame:
    if os.path.exists(TRACKER_FILE):
        df = pd.read_csv(TRACKER_FILE)
        for col in TRACKER_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        # Existing CSVs predate the Direction column. Every row already in
        # the file was opened from the ranked BUY list, so backfilling is
        # correct — and it keeps the 25 open positions intact rather than
        # orphaning them with a blank direction.
        blank = df["Direction"].isna() | (df["Direction"].astype(str).str.strip() == "")
        df.loc[blank, "Direction"] = "BUY"

        # Force the exit columns to object dtype before anything writes to
        # them.
        #
        # While no position has closed, ExitDate/ExitPrice/ReturnPct/
        # ExitReason are entirely empty, so pandas reads them back as
        # float64 NaN. Assigning a date string into a float64 column
        # raises TypeError on modern pandas:
        #
        #   TypeError: Invalid value '2026-08-30' for dtype 'float64'
        #
        # That crash would have fired on the FIRST closure — around
        # October 2026 — and taken out the run that produces the only
        # forward evidence this project has. It is latent right now
        # precisely BECAUSE nothing has closed yet.
        for col in ("ExitDate", "ExitPrice", "ReturnPct", "ExitReason", "Flag"):
            df[col] = df[col].astype(object)
        return df[TRACKER_COLUMNS]
    return pd.DataFrame(columns=TRACKER_COLUMNS)


def open_new_positions(tracker: pd.DataFrame,
                      source_file: str = RANKED_BUY_FILE,
                      direction: str = "BUY") -> pd.DataFrame:
    """
    Open paper positions from a signal file.

    Called once per direction. The SELL side has never been tracked, so
    the SELL rule has no forward evidence at all — every test of it has
    been a backtest against controls. This is the only way that changes.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    if not os.path.exists(source_file):
        print(f"No {source_file} found — skipping new {direction} entries.")
        return tracker

    ranked = pd.read_csv(source_file)
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
                & (tracker["Direction"] == direction)
                & (tracker["Status"] == "OPEN")
            ).any()
        )
        if already_open:
            continue  # don't double-enter a pick that's already being tracked

        # CRITICAL: fetch the CURRENT daily close as the entry price.
        #
        # The "Price" column in the ranked CSV is the close of the last
        # COMPLETED bar of the signal's timeframe — nse_scanner.resample()
        # deliberately drops the in-progress candle, so a Monthly scan run
        # on 13-Aug reports the 31-JULY close. Using that as the entry
        # price while marking to today's daily close silently measured a
        # two-week price move and labelled it a one-day return. That made
        # every position look like a huge gain or loss on day one
        # (+24.66%, -18.35%) and tripped the anomaly flag on all of them.
        #
        # Entering at today's real price is also what actually happens if
        # you act on a signal: you pay today's price, not last month's.
        signal_price = pd.to_numeric(r.get("Price"), errors="coerce")

        hist = fetch_single(symbol)
        if hist is None or hist.empty:
            print(f"  Could not fetch {symbol} — skipping this pick.")
            continue

        entry_price = float(hist.sort_index()["Close"].iloc[-1])
        if not entry_price or entry_price <= 0:
            continue

        note = ""
        if not pd.isna(signal_price) and signal_price > 0:
            drift = (entry_price - float(signal_price)) / float(signal_price) * 100
            if abs(drift) >= 10:
                note = (f"Price moved {drift:+.1f}% since the signal bar "
                        f"({signal_price:.2f} -> {entry_price:.2f})")

        new_rows.append({
            "Symbol": symbol,
            "Direction": direction,
            "Timeframe": timeframe,
            "EntryDate": today,
            "EntryPrice": round(entry_price, 2),
            "SignalBarPrice": round(float(signal_price), 2) if not pd.isna(signal_price) else "",
            "BuyScore": r.get("BuyScore", ""),
            "Status": "OPEN",
            "LastCheckedDate": today,
            "LastPrice": round(entry_price, 2),
            "UnrealizedReturnPct": 0.0,
            "SignalReturnPct": 0.0,
            "ExitDate": "",
            "ExitPrice": "",
            "ReturnPct": "",
            "HoldingDays": 0,
            "ExitReason": "",
            "Flag": note,
        })

    if new_rows:
        print(f"Opening {len(new_rows)} new {direction} paper position(s) at today's prices.")
        tracker = pd.concat([tracker, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print(f"No new {direction} picks to open (already tracked, or today's list is empty).")

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

        # Drop bars with no close before marking to market.
        #
        # yfinance can hand back a row with NaN OHLC — a placeholder for
        # the current session, a weekend/holiday artefact, or a bad tick.
        # Taking .iloc[-1] blindly then wrote NaN into LastPrice AND
        # UnrealizedReturnPct, which is how every open position lost its
        # return on 2026-08-29 while HoldingDays still updated: the writes
        # are not atomic, so a NaN price corrupted half the row.
        #
        # Skipping is the right response, not zeroing. update_open_positions
        # only ever writes values, so `continue` leaves the last good
        # figures in place rather than overwriting them with garbage.
        valid_bars = bars_since_entry[bars_since_entry["Close"].notna()]
        if valid_bars.empty:
            print(f"  {symbol}: no valid close since entry — leaving position as-is.")
            continue

        last_price = float(valid_bars["Close"].iloc[-1])
        if not np.isfinite(last_price) or last_price <= 0:
            print(f"  {symbol}: implausible last price ({last_price}) — leaving position as-is.")
            continue
        if not np.isfinite(entry_price) or entry_price <= 0:
            print(f"  {symbol}: bad entry price ({entry_price}) — cannot mark to market.")
            continue

        # Count only bars that actually traded, so a placeholder row does
        # not inflate the holding period and trip the exit rule early.
        holding_days = len(valid_bars)
        unrealized = round(((last_price - entry_price) / entry_price) * 100, 2)

        # Direction-adjusted. A SELL that was right shows a NEGATIVE price
        # move and a POSITIVE signal return.
        direction = str(tracker.at[idx, "Direction"] or "BUY").strip().upper()
        signal_return = round(unrealized if direction == "BUY" else -unrealized, 2)

        # Anomaly check: look at every individual day-over-day move since
        # entry (not just the cumulative return) — a single suspiciously
        # large jump is a strong sign of a bad/stale price tick, an
        # illiquid thin-volume gap, or an unadjusted stock split/bonus,
        # rather than genuine price action.
        closes = pd.concat([pd.Series([entry_price]), valid_bars["Close"]])
        daily_moves = closes.pct_change().dropna() * 100
        max_daily_move = daily_moves.abs().max() if not daily_moves.empty else 0.0

        flag = clean_flag(tracker.at[idx, "Flag"])
        # Preserve any note recorded at entry (e.g. large drift between
        # the signal bar and the actual entry price) rather than
        # overwriting it, but don't stack duplicate anomaly warnings.
        flag = "" if flag.startswith("Flagged:") else flag

        if max_daily_move >= ANOMALY_1DAY_MOVE_PCT:
            anomaly = (f"Flagged: {max_daily_move:+.1f}% single-day move — "
                       f"verify price data before trusting this result")
            flag = f"{flag} | {anomaly}" if flag else anomaly
            print(f"  ⚠ {symbol}: {anomaly}")

        tracker.at[idx, "LastCheckedDate"] = today
        tracker.at[idx, "LastPrice"] = round(last_price, 2)
        tracker.at[idx, "UnrealizedReturnPct"] = unrealized
        tracker.at[idx, "SignalReturnPct"] = signal_return
        tracker.at[idx, "HoldingDays"] = holding_days
        tracker.at[idx, "Flag"] = flag

        if holding_days >= HOLD_TRADING_DAYS:
            tracker.at[idx, "Status"] = "CLOSED"
            tracker.at[idx, "ExitDate"] = today
            tracker.at[idx, "ExitPrice"] = round(last_price, 2)
            tracker.at[idx, "ReturnPct"] = unrealized
            tracker.at[idx, "ExitReason"] = f"Held {HOLD_TRADING_DAYS} trading days"
            print(f"  Closed {symbol} [{direction}]: stock {unrealized:+.2f}%, "
                  f"signal {signal_return:+.2f}% after {holding_days} trading days.")
        else:
            print(f"  {symbol} [{direction}]: stock {unrealized:+.2f}%, "
                  f"signal {signal_return:+.2f}% unrealized, "
                  f"day {holding_days}/{HOLD_TRADING_DAYS}")

    return tracker


def build_summary(tracker: pd.DataFrame) -> str:
    closed = tracker[tracker["Status"] == "CLOSED"].copy()
    open_count = int((tracker["Status"] == "OPEN").sum())
    flagged = tracker[tracker["Flag"].map(clean_flag).str.len() > 0]

    lines = ["=" * 60, "  NIFTYPULSEPRO — DAILY RECOMMENDATION TRACKER", "=" * 60, ""]
    lines.append(f"Generated        : {datetime.now().strftime('%d-%b-%Y %H:%M')}")
    lines.append(f"Open positions   : {open_count}")
    lines.append(f"Closed positions : {len(closed)}")

    for d in DIRECTIONS:
        n = int(((tracker["Status"] == "OPEN") & (tracker["Direction"] == d)).sum())
        lines.append(f"  open {d:<4s}     : {n}")

    if not closed.empty:
        # Exclude only GENUINE data anomalies, not informational notes.
        #
        # Flag holds two different kinds of message. "Flagged: ..." is a
        # real data-quality problem (an implausible single-day move).
        # Anything else is informational — most often "Price moved X%
        # since the signal bar", which fires on nearly every Monthly row
        # because the signal bar closed up to 30 days earlier.
        #
        # Excluding both meant every Monthly SELL was dropped from the
        # stats, leaving the SELL side with no numbers at all — the exact
        # thing this tracking was added to produce. Drift is a property of
        # the signal, not a corrupt price.
        def _is_anomaly(v):
            return clean_flag(v).startswith("Flagged:")

        clean_closed = closed[~closed["Flag"].map(_is_anomaly)]

        # Reported PER DIRECTION, never pooled.
        #
        # A combined win rate over BUY and SELL is meaningless: the two
        # test opposite claims, and mixing them lets a good result on one
        # side mask a bad one on the other. They are separate experiments
        # that happen to share a tracker file.
        for d in DIRECTIONS:
            sub = clean_closed[clean_closed["Direction"] == d]
            if sub.empty:
                continue
            stock = pd.to_numeric(sub["ReturnPct"], errors="coerce").dropna()
            if stock.empty:
                continue
            # Win = the signal was right, so SELL wins when price fell.
            signal = stock if d == "BUY" else -stock
            wins = int((signal > 0).sum())
            lines.append("")
            lines.append(f"{d} signals ({len(signal)} closed)")
            lines.append(f"  Win rate       : {wins}/{len(signal)} ({wins/len(signal)*100:.1f}%)")
            lines.append(f"  Avg signal ret : {signal.mean():+.2f}%")
            lines.append(f"  Avg stock move : {stock.mean():+.2f}%")
            lines.append(f"  Best / worst   : {signal.max():+.2f}% / {signal.min():+.2f}%")

        if len(clean_closed) < len(closed):
            lines.append("")
            lines.append(f"(excluded {len(closed) - len(clean_closed)} trade(s) with data "
                         f"anomalies from these stats — see below)")
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
            lines.append(f"  {r['Symbol']} [{r.get('Direction', 'BUY')}] "
                         f"({r['Timeframe']}): {r['Flag']}")

    lines.append("")
    lines.append("Not financial advice — forward-tracked results only reflect")
    lines.append("this specific rule set and holding period, not any trade you")
    lines.append("actually placed.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    tracker = load_tracker()
    tracker = open_new_positions(tracker, RANKED_BUY_FILE, "BUY")
    tracker = open_new_positions(tracker, SELL_FILE, "SELL")
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
