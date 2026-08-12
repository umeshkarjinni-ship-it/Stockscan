from pathlib import Path

from backtest import config as cfg
from backtest.metrics import (
    load_trades,
    calculate_summary,
)


class BacktestReport:

    def __init__(self):

        self.df = load_trades(cfg.TRADES_FILE)

        self.summary = calculate_summary(self.df)

    # -----------------------------------------------------

    def build_text(self):

        s = self.summary

        lines = []

        lines.append("=" * 60)
        lines.append("           NIFTYPULSE PRO BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append("")

        for k, v in s.items():
            lines.append(f"{k:<25}: {v}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    # -----------------------------------------------------

    def save(self):

        report = self.build_text()

        print(report)

        outfile = Path(cfg.SIGNALS_DIR) / "backtest_report.txt"

        outfile.write_text(report, encoding="utf-8")

        print(f"\nReport saved to {outfile}")


if __name__ == "__main__":

    BacktestReport().save()