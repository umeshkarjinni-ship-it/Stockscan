from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Load Trades
# ---------------------------------------------------------

def load_trades(csv_file: str) -> pd.DataFrame:

    df = pd.read_csv(
        csv_file,
        parse_dates=[
            "entry_date",
            "exit_date",
            "signal_date",
        ],
    )

    if len(df) == 0:
        raise ValueError("No trades found.")

    return df


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def calculate_summary(df: pd.DataFrame):

    total = len(df)

    wins = len(df[df["net_return"] > 0])

    losses = len(df[df["net_return"] <= 0])

    win_rate = wins / total * 100 if total else 0

    avg_win = (
        df[df["net_return"] > 0]["net_return"].mean()
        if wins
        else 0
    )

    avg_loss = (
        df[df["net_return"] <= 0]["net_return"].mean()
        if losses
        else 0
    )

    gross_profit = df[df["net_return"] > 0]["net_return"].sum()

    gross_loss = abs(
        df[df["net_return"] < 0]["net_return"].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    expectancy = df["net_return"].mean()

    avg_hold = df["holding_days"].mean()

    best_trade = df["net_return"].max()

    worst_trade = df["net_return"].min()

    return {

        "Total Trades": total,

        "Winning Trades": wins,

        "Losing Trades": losses,

        "Win Rate": round(win_rate, 2),

        "Average Winner": float(round(avg_win, 2)),

        "Average Loser": float(round(avg_loss, 2)),

        "Profit Factor": float(round(profit_factor, 2)),

        "Expectancy": float(round(expectancy, 2)),

        "Average Hold": float(round(avg_hold, 1)),

        "Best Trade": float(round(best_trade, 2)),

        "Worst Trade": float(round(worst_trade, 2)),
    }


# ---------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------

def calculate_equity_curve(df):

    equity = (1 + df["net_return"] / 100).cumprod()

    out = pd.DataFrame()

    out["Date"] = df["exit_date"]

    out["Equity"] = equity

    return out


# ---------------------------------------------------------
# Drawdown
# ---------------------------------------------------------

def calculate_drawdown(equity):

    running_max = equity["Equity"].cummax()

    dd = (equity["Equity"] - running_max) / running_max * 100

    out = equity.copy()

    out["Drawdown"] = dd

    return out


# ---------------------------------------------------------
# Monthly Returns
# ---------------------------------------------------------

def calculate_monthly_returns(df):

    temp = df.copy()

    temp["Month"] = temp["exit_date"].dt.to_period("M")

    return (
        temp.groupby("Month")["net_return"]
        .sum()
        .reset_index()
    )