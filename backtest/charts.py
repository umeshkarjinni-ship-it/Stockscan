from pathlib import Path

import matplotlib.pyplot as plt

from backtest import config as cfg
from backtest.metrics import (
    load_trades,
    calculate_equity_curve,
    calculate_drawdown,
    calculate_monthly_returns,
)


class BacktestCharts:

    def __init__(self):

        self.df = load_trades(cfg.TRADES_FILE)

        self.output = Path(cfg.SIGNALS_DIR)

        self.output.mkdir(exist_ok=True)

    # ----------------------------------------------------

    def equity_curve(self):

        eq = calculate_equity_curve(self.df)

        plt.figure(figsize=(10,5))

        plt.plot(eq["Date"], eq["Equity"])

        plt.title("Equity Curve")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output / "equity_curve.png")

        plt.close()

    # ----------------------------------------------------

    def drawdown(self):

        eq = calculate_equity_curve(self.df)

        dd = calculate_drawdown(eq)

        plt.figure(figsize=(10,5))

        plt.plot(dd["Date"], dd["Drawdown"])

        plt.title("Drawdown")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output / "drawdown_curve.png")

        plt.close()

    # ----------------------------------------------------

    def monthly_returns(self):

        monthly = calculate_monthly_returns(self.df)

        plt.figure(figsize=(10,5))

        plt.bar(
            monthly["Month"].astype(str),
            monthly["net_return"]
        )

        plt.xticks(rotation=90)

        plt.title("Monthly Returns")

        plt.tight_layout()

        plt.savefig(self.output / "monthly_returns.png")

        plt.close()

    # ----------------------------------------------------

    def trade_histogram(self):

        plt.figure(figsize=(8,5))

        plt.hist(self.df["net_return"], bins=15)

        plt.title("Trade Return Distribution")

        plt.tight_layout()

        plt.savefig(self.output / "trade_return_histogram.png")

        plt.close()

    # ----------------------------------------------------

    def win_loss(self):

        wins = len(self.df[self.df.net_return > 0])

        losses = len(self.df[self.df.net_return <= 0])

        plt.figure(figsize=(5,5))

        plt.pie(
            [wins, losses],
            labels=["Wins","Losses"],
            autopct="%1.1f%%"
        )

        plt.savefig(self.output / "win_loss_pie.png")

        plt.close()

    # ----------------------------------------------------

    def create_all(self):

        self.equity_curve()

        self.drawdown()

        self.monthly_returns()

        self.trade_histogram()

        self.win_loss()

        print("Charts created successfully.")


if __name__ == "__main__":

    BacktestCharts().create_all()