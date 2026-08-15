"""
compare_exits.py
=================
Runs the SAME set of backtest signals through several different exit
strategies and reports which produces the best risk-adjusted result.

Your backtest has only ever tested one exit rule: a fixed ~40-day time
exit. backtest/config.py had flags for ATR stops, trailing stops, VSTOP
reversals and profit targets, but backtest/exit_engine.py ignored them
all and always returned TIME_EXIT. Those rules are implemented now, and
this script compares them head to head.

In trend-following systems the exit usually matters more than the entry,
so this is often where the biggest remaining improvement lives.

REQUIRES: signals/backtest_signals.csv (from run_backtest.py)

USAGE
-----
    python compare_exits.py

Writes:
    signals/exit_comparison.csv          summary table, one row per strategy
    signals/exit_trades_<strategy>.csv   full trade list per strategy

PERFORMANCE NOTE
----------------
Price history is downloaded ONCE per symbol and reused across every
strategy. Re-fetching per strategy would multiply an already slow job by
the number of strategies being tested.
"""

import os
from dataclasses import asdict

import pandas as pd

from backtest import config as cfg
from backtest.exit_engine import ExitEngine
from backtest.simulator import Trade
from backtest.utils import pct_return
from nse_scanner import fetch_single, compute_signals

SIGNALS_DIR = "signals"
OUTPUT_FILE = os.path.join(SIGNALS_DIR, "exit_comparison.csv")

# Each entry: (label, ExitEngine kwargs)
#
# Tests TWO dimensions:
#   1. Which exit RULE fires (time / VStop flip / ATR stop / trailing / target)
#   2. How long the maximum HOLD is
#
# On MAX_HOLD_DAYS: the name is misleading — data is resampled to Weekly
# or Monthly bars, so this counts BARS, not days. The default of 40 means
# ~9 months on Weekly and ~3.3 years on Monthly. That default was never
# tested against alternatives, and since no other exit rule was actually
# implemented until now, essentially every trade ran to this limit.
STRATEGIES = [
    # --- Exit rule comparison, all at the current 40-bar default ---
    ("time_only_40bar", dict(
        use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("vstop_flip", dict(
        use_vstop_exit=True, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("atr_stop_2x", dict(
        use_vstop_exit=False, use_atr_stop=True, atr_stop_multiplier=2.0,
        use_trailing_stop=False, use_profit_target=False)),

    ("trailing_2.5x", dict(
        use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=True, trailing_stop_multiplier=2.5,
        use_profit_target=False)),

    ("profit_target_15", dict(
        use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False,
        use_profit_target=True, profit_target=15.0)),

    ("atr_stop_plus_vstop", dict(
        use_vstop_exit=True, use_atr_stop=True, atr_stop_multiplier=2.0,
        use_trailing_stop=False, use_profit_target=False)),

    ("trailing_plus_vstop", dict(
        use_vstop_exit=True, use_atr_stop=False,
        use_trailing_stop=True, trailing_stop_multiplier=2.5,
        use_profit_target=False)),

    # --- Hold-length comparison, pure time exit so the only variable
    #     is duration. Directly answers "is 40 bars the right number?" ---
    #
    # MEASURED (1,430 signals): profit factor rose monotonically with
    # hold length — 5 bars 0.85, 10 bars 1.04, 20 bars 1.31, 40 bars
    # 1.92, 60 bars 2.42 — and return-per-bar peaked at 60 too. That
    # clean staircase is unlikely to be noise. But 60 was just the
    # longest value tested, so the sweep is extended below to find where
    # the improvement actually plateaus or reverses.
    ("time_only_5bar", dict(
        max_hold_days=5, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_10bar", dict(
        max_hold_days=10, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_20bar", dict(
        max_hold_days=20, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_60bar", dict(
        max_hold_days=60, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_80bar", dict(
        max_hold_days=80, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_100bar", dict(
        max_hold_days=100, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_130bar", dict(
        max_hold_days=130, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("time_only_200bar", dict(
        max_hold_days=200, use_vstop_exit=False, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    # --- Best-guess combination: let the trend decide the exit, with a
    #     long backstop rather than a short arbitrary cutoff ---
    ("vstop_flip_60bar", dict(
        max_hold_days=60, use_vstop_exit=True, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),

    ("vstop_flip_130bar", dict(
        max_hold_days=130, use_vstop_exit=True, use_atr_stop=False,
        use_trailing_stop=False, use_profit_target=False)),
]


def load_price_cache(symbols):
    """Fetch and pre-compute indicators once per symbol."""
    cache = {}
    total = len(symbols)
    for n, sym in enumerate(symbols, start=1):
        print(f"  [{n}/{total}] {sym}", flush=True)
        try:
            raw = fetch_single(sym)
            if raw is None or len(raw) == 0:
                continue
            sig = compute_signals(raw.sort_index(), None)
            if sig is not None and not sig.empty:
                cache[sym] = sig
        except Exception as e:
            print(f"      skipped ({e})")
    return cache


def simulate(signals_df, cache, engine):
    """Replay every signal with the given exit engine."""
    trades = []
    for _, signal in signals_df.iterrows():
        try:
            sig = cache.get(signal["symbol"])
            if sig is None:
                continue

            signal_date = pd.Timestamp(signal["signal_date"])
            future = sig[sig.index > signal_date]
            if len(future) < 2:
                continue

            entry_bar = future.iloc[0]
            trade_data = future.reset_index(drop=False)

            result = engine.find_exit(trade_data)
            exit_bar = future.iloc[result.exit_index]

            entry_price = float(entry_bar["Open"])
            exit_price = float(exit_bar["Close"])
            if entry_price <= 0:
                continue

            gross = pct_return(entry_price, exit_price)
            net = gross - cfg.ROUND_TRIP_COST_PCT

            trades.append({
                "symbol": signal["symbol"],
                "timeframe": signal.get("timeframe", ""),
                "signal_date": signal["signal_date"],
                "holding_days": result.exit_index,
                "net_return": round(net, 2),
                "exit_reason": result.reason,
            })
        except Exception:
            continue
    return pd.DataFrame(trades)


def summarize(df, label, max_hold=None):
    if df.empty:
        return {"strategy": label, "trades": 0}

    wins = df[df.net_return > 0]
    losses = df[df.net_return <= 0]
    gross_win = wins.net_return.sum()
    gross_loss = abs(losses.net_return.sum())

    # How many trades ended early only because history ran out, rather
    # than because a rule fired. This matters most for LONG holds: if a
    # 200-bar strategy has half its trades truncated, its numbers reflect
    # a much shorter hold than the label claims.
    truncated = None
    if max_hold:
        truncated = int(
            ((df.exit_reason == "TIME_EXIT") & (df.holding_days < max_hold)).sum()
        )

    return {
        "strategy": label,
        "trades": len(df),
        "win_rate": round(len(wins) / len(df) * 100, 1),
        "expectancy": round(df.net_return.mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "avg_winner": round(wins.net_return.mean(), 2) if len(wins) else 0,
        "avg_loser": round(losses.net_return.mean(), 2) if len(losses) else 0,
        "avg_hold": round(df.holding_days.mean(), 1),
        # Return per BAR held. Essential when comparing different hold
        # lengths: a 60-bar strategy ties up capital 12x longer than a
        # 5-bar one, so raw expectancy alone would unfairly favour long
        # holds. This is the closest thing here to a capital-efficiency
        # measure.
        "return_per_bar": round(
            df.net_return.mean() / df.holding_days.mean(), 4
        ) if df.holding_days.mean() else None,
        "truncated": truncated,
        "total_return": round(df.net_return.sum(), 1),
        "worst_trade": round(df.net_return.min(), 2),
    }


def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    if not os.path.exists(cfg.SIGNALS_FILE):
        raise SystemExit(
            f"\n{cfg.SIGNALS_FILE} not found — run run_backtest.py first.\n"
        )

    signals_df = pd.read_csv(cfg.SIGNALS_FILE, parse_dates=["signal_date"])
    print(f"Loaded {len(signals_df)} signals")

    symbols = sorted(signals_df["symbol"].dropna().unique())
    print(f"Fetching price history for {len(symbols)} unique symbols (once, reused for all strategies)...")
    cache = load_price_cache(symbols)
    print(f"Cached {len(cache)} symbols\n")

    rows = []
    for label, kwargs in STRATEGIES:
        print(f"Simulating: {label}")
        engine = ExitEngine(**kwargs)
        trades = simulate(signals_df, cache, engine)
        trades.to_csv(os.path.join(SIGNALS_DIR, f"exit_trades_{label}.csv"), index=False)
        row = summarize(trades, label, max_hold=kwargs.get("max_hold_days", cfg.MAX_HOLD_DAYS))
        rows.append(row)
        if row.get("trades"):
            print(f"   n={row['trades']}  win={row['win_rate']}%  "
                  f"exp={row['expectancy']:+.2f}%  PF={row['profit_factor']}  "
                  f"hold={row['avg_hold']}d")
        print()

    summary = pd.DataFrame(rows).sort_values("profit_factor", ascending=False, na_position="last")
    summary.to_csv(OUTPUT_FILE, index=False)

    print("=" * 78)
    print("  EXIT STRATEGY COMPARISON (sorted by profit factor)")
    print("=" * 78)
    print(summary.to_string(index=False))
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()
    print("Read this with care:")
    print("  - Compare return_per_bar, not just profit factor: a 60-bar")
    print("    strategy ties up capital 12x longer than a 5-bar one.")
    print("  - Check 'truncated': trades that ended early only because")
    print("    history ran out. If a long-hold strategy has many, its real")
    print("    hold is shorter than the label suggests and the numbers are")
    print("    less trustworthy.")
    print("  - Check worst_trade: a strategy you'd abandon mid-drawdown has")
    print("    no edge in practice.")
    print("  - These results exclude position sizing and the fact that long")
    print("    holds mean more overlapping positions competing for capital.")


if __name__ == "__main__":
    main()
