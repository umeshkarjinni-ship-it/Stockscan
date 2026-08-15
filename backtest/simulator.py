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
    fetch_nifty_daily,
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
        exit_engine: ExitEngine = None,
        save_to: str = None,
    ):

        self.holding_period = holding_period

        self.trades: List[Trade] = []

        # Allow a caller to inject a pre-configured ExitEngine so several
        # exit strategies can be compared against the same signal set
        # without editing backtest/config.py.
        self.exit_engine = exit_engine if exit_engine is not None else ExitEngine()

        # Optional override of the output path, so comparison runs don't
        # overwrite the main backtest_trades.csv.
        self.save_to = save_to

        # Per-symbol cache of computed signals, so repeated trades on the
        # same symbol don't re-download and re-compute.
        self._signal_cache = {}

        # Benchmark series for the relative-strength filter. Fetched
        # lazily on first use so constructing a simulator stays cheap.
        self._nifty_close = None
        self._nifty_loaded = False

    # ------------------------------------------------------------------

    @property
    def nifty_close(self):
        if not self._nifty_loaded:
            self._nifty_loaded = True
            try:
                nifty = fetch_nifty_daily()
                if nifty is not None and not nifty.empty:
                    self._nifty_close = nifty["Close"]
            except Exception:
                logger.warning("Could not fetch NIFTY; RS filter disabled in simulation.")
        return self._nifty_close

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

        # Cache per symbol. This used to call fetch_single() + 
        # compute_signals() once per TRADE — with 1,236 trades across only
        # 377 unique symbols that meant ~3.3x redundant downloads and
        # indicator recomputation.
        if symbol in self._signal_cache:
            signals = self._signal_cache[symbol]
        else:
            signals = fetch_single(symbol)

            if signals is None or len(signals) == 0:
                self._signal_cache[symbol] = None
                return None

            signals = signals.sort_index()

            # NOTE: nifty_close is passed through so the relative-strength
            # line matches what the signal engine computed. Passing None
            # here (as this previously did) makes compute_signals() skip
            # the RS filter entirely, so the bars used for exits were
            # computed under different conditions than the bars used to
            # generate the signal.
            signals = compute_signals(signals, self.nifty_close)

            self._signal_cache[symbol] = signals

        if signals is None or signals.empty:
            return None
        signal_date = pd.Timestamp(signal["signal_date"])

        future = signals[signals.index > signal_date]

        # Require at least 2 bars (entry + one more) rather than a fixed
        # holding_period. The exit engine decides how long to hold, so
        # gating on holding_period here silently DISCARDED every signal
        # too close to the end of history — biasing the sample by dropping
        # the most recent trades.
        if len(future) < 2:
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

        # Round-trip cost: brokerage, slippage, STT, exchange charges,
        # GST and stamp duty across BOTH legs. See backtest/config.py.
        cost = cfg.ROUND_TRIP_COST_PCT

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
            self.save_to if self.save_to else cfg.TRADES_FILE,
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