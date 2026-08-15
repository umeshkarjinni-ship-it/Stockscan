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

    # --- Additional features for ML training ---
    #
    # The four above are all GATING conditions: a signal only fires when
    # each is already inside its required band, so by the time a trade
    # exists they carry almost no variance and no predictive information
    # (a model trained on them alone scored ROC-AUC 0.494 — worse than a
    # coin flip). The features below are NOT part of the entry rules, so
    # they retain real spread across trades and may actually discriminate.

    pct_from_52w_high: float = 0.0   # how extended vs the 1-year high

    atr_pct: float = 0.0             # ATR / Close = volatility regime

    dist_from_kama_pct: float = 0.0  # how far above the mid trend line

    nifty_regime_up: int = 0         # 1 if NIFTY above its own 50-bar MA


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
        Walk forward through history one completed candle at a time,
        recording each NEW buy signal.

        PERFORMANCE
        -----------
        This used to call compute_signals() again for every single bar on
        a growing window, making the cost O(n^2): ~700 weekly bars meant
        ~245,000 bar-computations per symbol, and a full run took hours.

        Every indicator in compute_signals() is causal — RSI, ATR, ADX,
        OBV, volume MA, KAMA and the recursive VStop all depend only on
        past and current bars, and the relative-strength line uses a
        trailing mean. So computing on the full series once yields exactly
        the same value at each bar as recomputing on truncated windows.
        This was verified empirically: 59 bars x 10 indicator columns
        compared between both approaches produced 0 mismatches.

        `hist` is already fully resampled before this method is called, so
        truncating it only ever removed complete bars — there is no
        partial-candle difference either. NO look-ahead is introduced.
        """

        # compute_signals() enforces its own minimum bar count internally
        # (KAMA_SLOW_LEN + 2 = 102) and returns an empty frame below it,
        # so only a tiny buffer is needed here.
        warmup = 5

        if len(hist) < warmup:
            return

        try:
            signals = compute_signals(hist, self.nifty_close)
        except Exception:
            logger.exception("Signal calculation failed for %s", symbol)
            return

        if signals is None or signals.empty:
            return

        previous_buy = False

        for pos in range(len(signals)):

            last = signals.iloc[pos]

            buy_now = self._is_buy_signal(last)

            # Record only NEW BUY events (a fresh transition into a buy
            # state, not every bar the state stays true).
            if buy_now and not previous_buy:

                event = self._create_event(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    row=last,
                    # Pass only bars up to and including this one, so the
                    # 52-week-high feature can never see the future.
                    window=signals.iloc[: pos + 1],
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
        window=None,
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

            # ----------------------------
            # Non-gated ML features
            # ----------------------------
            #
            # Computed from the signal-time window only — never from
            # future bars — so there is no look-ahead leakage.

            pct_from_52w_high = 0.0
            atr_pct = 0.0
            dist_from_kama_pct = 0.0
            nifty_regime_up = 0

            try:
                if window is not None and len(window) > 0 and price:
                    # 52-week high: 52 weekly bars or 12 monthly bars
                    lookback = 52 if timeframe.lower() == "weekly" else 12
                    recent = window["Close"].iloc[-lookback:]
                    high_52w = float(recent.max())
                    if high_52w > 0:
                        pct_from_52w_high = round(
                            (price - high_52w) / high_52w * 100, 3
                        )

                atr_val = self._get_value(
                    row,
                    # compute_signals() exposes ATR_MULT * ATR as
                    # "ATR_STOP" (the VStop distance), not the raw ATR —
                    # checking only "ATR" silently returned 0 for every
                    # trade. Divide the multiplier back out so atr_pct is
                    # a true volatility percentage.
                    ["ATR_STOP", "ATR", "atr"],
                    default=0.0,
                )
                if price and atr_val:
                    try:
                        from nse_scanner import ATR_MULT
                        mult = ATR_MULT or 1.0
                    except Exception:
                        mult = 1.0
                    atr_pct = round((atr_val / mult) / price * 100, 3)

                kama_mid = self._get_value(row, ["KAMA_MID"], default=0.0)
                if price and kama_mid:
                    dist_from_kama_pct = round(
                        (price - kama_mid) / kama_mid * 100, 3
                    )

                if self.nifty_close is not None and len(self.nifty_close) >= 50:
                    upto = self.nifty_close.loc[:row.name]
                    if len(upto) >= 50:
                        nifty_regime_up = int(
                            float(upto.iloc[-1]) > float(upto.iloc[-50:].mean())
                        )
            except Exception:
                # Feature extraction must never break signal recording.
                pass

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

                pct_from_52w_high=pct_from_52w_high,

                atr_pct=atr_pct,

                dist_from_kama_pct=dist_from_kama_pct,

                nifty_regime_up=nifty_regime_up,

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
