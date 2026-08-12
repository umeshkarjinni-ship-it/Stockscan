"""
backtest.engine

Historical signal replay engine for NiftyPulse Pro.

This module reuses the production scanner logic in nse_scanner.py.
No indicator calculations are duplicated.

Author: NiftyPulse Pro
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

from . import config as cfg
from .utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
)

# ---------------------------------------------------------
# IMPORT FUNCTIONS FROM YOUR EXISTING SCANNER
# ---------------------------------------------------------

from nse_scanner import (
    fetch_single,
    resample,
    compute_signals,
)

logger = setup_logger(cfg.LOG_FILE)


# ---------------------------------------------------------
# SIGNAL EVENT
# ---------------------------------------------------------

@dataclass
class SignalEvent:

    symbol: str

    timeframe: str

    signal_date: str

    entry_price: float

    buy_score: float

    rsi: float

    adx: float

    volume_ratio: float

    relative_strength: float


# ---------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------

class BacktestEngine:

    def __init__(
        self,
        symbols: List[str],
        nifty_close: pd.Series | None = None,
        timeframe: str = "Weekly",
    ):

        self.symbols = symbols
        self.nifty_close = nifty_close
        self.timeframe = timeframe

        self.events: List[SignalEvent] = []

        self.checkpoint = load_checkpoint(cfg.CHECKPOINT_FILE)

    # -----------------------------------------------------

    def save_events(self):

        if not self.events:
            logger.warning("No events found.")
            return

        df = pd.DataFrame([asdict(x) for x in self.events])

        df.to_csv(cfg.SIGNALS_FILE, index=False)

        logger.info(
            "Saved %d signals to %s",
            len(df),
            cfg.SIGNALS_FILE,
        )

    # -----------------------------------------------------

    def replay_symbol(self, symbol):

        logger.info("Replay %s", symbol)

        try:

            df = fetch_single(symbol)

            if df is None:
                return

            if len(df) < 250:
                return

            if self.timeframe.lower() == "weekly":
                hist = resample(df, "W")

            elif self.timeframe.lower() == "monthly":
                hist = resample(df, "ME")

            else:
                hist = df.copy()

            self._scan_history(symbol, hist)

        except Exception:

            logger.exception("Replay failed for %s", symbol)

    # -----------------------------------------------------
    # -----------------------------------------------------

    def _scan_history(self, symbol: str, hist: pd.DataFrame):

        """
        Walk forward through history one completed candle at a time.

        This guarantees there is NO look-ahead bias because
        compute_signals() only sees data available up to that candle.
        """

        # Determine warm-up based on timeframe
        if self.timeframe.lower() == "monthly":
            warmup = 36          # 3 years
        elif self.timeframe.lower() == "weekly":
            warmup = 60          # ~1 year
        else:
            warmup = 200         # Daily

        if len(hist) < warmup:
            return

        previous_buy = False

        for i in range(warmup, len(hist)):

            window = hist.iloc[: i + 1].copy()

            try:
                # Align benchmark history to the same point in time
                nifty_window = None

                if self.nifty_close is not None:
                    nifty_window = self.nifty_close.loc[:window.index[-1]]

                signals = compute_signals(window, nifty_window)

            except Exception:

                logger.exception(
                    "Signal calculation failed for %s at index %d",
                    symbol,
                    i,
                )
                continue

            if signals is None:
                continue

            if signals.empty:
                continue

            last = signals.iloc[-1]

            buy_now = self._is_buy_signal(last)

            # Record only NEW BUY events
            if buy_now and not previous_buy:

                event = self._create_event(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    row=last,
                )

                if event is not None:
                    self.events.append(event)

            previous_buy = buy_now

    # -----------------------------------------------------

    def _is_buy_signal(self, row: pd.Series) -> bool:
        """
        Detect whether this row represents a BUY signal.

        Compatible with future scanner versions.
        """

        buy_columns = [
            "FINAL_BUY",
            "BUY_SIGNAL",
            "BUY_SIGNAL_CORE",
        ]

        for col in buy_columns:

            if col in row.index:

                try:

                    return bool(row[col])

                except Exception:

                    return False

        return False

    # -----------------------------------------------------

    def _get_value(
        self,
        row: pd.Series,
        candidates,
        default=0.0,
    ):
        """
        Return the first available column.
        """

        for col in candidates:

            if col in row.index:

                try:
                    return float(row[col])

                except Exception:
                    return default

        return default

    # -----------------------------------------------------

    def _create_event(
        self,
        symbol,
        timeframe,
        row,
    ) -> Optional[SignalEvent]:

        try:

            date = row.name

            price = self._get_value(
                row,
                [
                    "Close",
                    "close",
                    "Adj Close",
                ],
            )

            # ----------------------------
# Calculate Buy Score
# ----------------------------

            buy_checks = [
                "BUY_FLIP_OK",
                "BUY_MA_OK",
                "BUY_ADX_OK",
                "BUY_RSI_OK",
                "BUY_VOL_OK",
                "BUY_OBV_OK",
                "BUY_RS_OK",
            ]

            passed = 0

            for c in buy_checks:
                if c in row.index and bool(row[c]):
                    passed += 1

            score = round(100.0 * passed / len(buy_checks), 1)

            # ----------------------------
            # RSI
            # ----------------------------

            rsi = self._get_value(
                row,
                ["RSI"],
            )

            # ----------------------------
            # ADX
            # ----------------------------

            adx = self._get_value(
                row,
                ["ADX"],
            )

            # ----------------------------
            # Volume Ratio
            # ----------------------------

            if "VOL_MA" in row.index and row["VOL_MA"] != 0:
                volume_ratio = float(row["Volume"]) / float(row["VOL_MA"])
            else:
                volume_ratio = 0.0

            # ----------------------------
            # Relative Strength
            # ----------------------------

            rs = self._get_value(
                row,
                [
                    "RS_LINE",
                    "RS",
                ],
            )

            return SignalEvent(

                symbol=symbol,

                timeframe=timeframe,

                signal_date=str(date),

                entry_price=price,

                buy_score=score,

                rsi=rsi,

                adx=adx,

                volume_ratio=volume_ratio,

                relative_strength=rs,

            )

        except Exception:

            logger.exception(
                "Unable to create SignalEvent for %s",
                symbol,
            )

            return None


    def run(self):

        total = len(self.symbols)

        logger.info(
            "Starting replay for %d symbols",
            total,
        )

        for i, symbol in enumerate(self.symbols, start=1):

            print(f"[{i}/{total}] {symbol}")

            self.replay_symbol(symbol)

            save_checkpoint(
                cfg.CHECKPOINT_FILE,
                {
                    "last_symbol": symbol,
                    "completed": i,
                },
            )

        self.save_events()

        logger.info("Replay complete.")
