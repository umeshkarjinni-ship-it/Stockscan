"""
backtest.simulator

Converts historical BUY signals into simulated trades.

NiftyPulse Pro
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional
from unittest import signals

import pandas as pd

import backtest.config as cfg
from backtest.utils import (
    pct_return,
    setup_logger,
)
from nse_scanner import (
    fetch_single,
    compute_signals,
)

from backtest.exit_engine import ExitEngine

logger = setup_logger(cfg.LOG_FILE)


# ----------------------------------------------------------------------

@dataclass
class Trade:

    symbol: str

    timeframe: str

    signal_date: str

    entry_date: str

    exit_date: str

    holding_days: int

    entry_price: float

    exit_price: float

    gross_return: float

    net_return: float

    buy_score: float

    rsi: float

    adx: float

    volume_ratio: float

    relative_strength: float

    # Non-gated features carried through from SignalEvent for ML training.
    pct_from_52w_high: float = 0.0
    atr_pct: float = 0.0
    dist_from_kama_pct: float = 0.0
    nifty_regime_up: int = 0

    exit_reason: str = ""


# ----------------------------------------------------------------------

class TradeSimulator:

    def __init__(
        self,
        holding_period: int = 20,
    ):

        self.holding_period = holding_period

        self.trades: List[Trade] = []
        self.exit_engine = ExitEngine()

    # ------------------------------------------------------------------

    def run(self):

        logger.info("Loading signals...")

        signals = pd.read_csv(
            cfg.SIGNALS_FILE,
            parse_dates=["signal_date"],
        )

        logger.info(
            "Signals loaded : %d",
            len(signals),
        )

        for _, signal in signals.iterrows():

            try:

                trade = self.simulate_trade(signal)

                if trade is not None:

                    self.trades.append(trade)

            except Exception:

                logger.exception(
                    "Trade simulation failed for %s",
                    signal["symbol"],
                )

        self.save()

    # ------------------------------------------------------------------

    def simulate_trade(
        self,
        signal,
    ) -> Optional[Trade]:

        symbol = signal["symbol"]

        signals = fetch_single(symbol)

        if signals is None:
            return None

        if len(signals) == 0:
            return None

        signals = signals.sort_index()
        signals = compute_signals(signals, None)

        if signals is None or signals.empty:
            return None
        signal_date = pd.Timestamp(signal["signal_date"])

        future = signals[signals.index > signal_date]

        if len(future) < self.holding_period + 1:
            return None

        entry_bar = future.iloc[0]

        # Future data beginning from entry day
        trade_data = future.reset_index(drop=False)

        exit_result = self.exit_engine.find_exit(trade_data)

        exit_bar = future.iloc[exit_result.exit_index]

        exit_reason = exit_result.reason
        
        entry_price = float(entry_bar["Open"])

        exit_price = float(exit_bar["Close"])

        gross = pct_return(
            entry_price,
            exit_price,
        )

        cost = (
            cfg.BROKERAGE
            + cfg.SLIPPAGE
        ) * 100

        net = gross - cost

        return Trade(

            symbol=symbol,

            timeframe=signal["timeframe"],

            signal_date=str(signal_date.date()),

            entry_date=str(entry_bar.name.date()),

            exit_date=str(exit_bar.name.date()),

            holding_days=exit_result.exit_index,
            
            entry_price=round(entry_price, 2),

            exit_price=round(exit_price, 2),

            gross_return=round(gross, 2),

            net_return=round(net, 2),

            buy_score=float(signal["buy_score"]),

            rsi=float(signal["rsi"]),

            adx=float(signal["adx"]),

            volume_ratio=float(signal["volume_ratio"]),

            relative_strength=float(signal["relative_strength"]),

            pct_from_52w_high=float(signal.get("pct_from_52w_high", 0.0) or 0.0),

            atr_pct=float(signal.get("atr_pct", 0.0) or 0.0),

            dist_from_kama_pct=float(signal.get("dist_from_kama_pct", 0.0) or 0.0),

            nifty_regime_up=int(signal.get("nifty_regime_up", 0) or 0),

            exit_reason=exit_result.reason,

        )

    # ------------------------------------------------------------------

    def save(self):

        if len(self.trades) == 0:

            logger.warning("No trades generated.")

            return

        signals = pd.DataFrame(
            [asdict(t) for t in self.trades]
        )

        signals.sort_values(
            "entry_date",
            inplace=True,
        )

        signals.to_csv(
            cfg.TRADES_FILE,
            index=False,
        )

        logger.info(
            "Saved %d trades",
            len(signals),
        )

    # ------------------------------------------------------------------

    def dataframe(self):

        if len(self.trades) == 0:
            return pd.DataFrame()

        return pd.DataFrame(
            [asdict(x) for x in self.trades]
        )


# ----------------------------------------------------------------------

if __name__ == "__main__":
    TradeSimulator(holding_period=20).run()