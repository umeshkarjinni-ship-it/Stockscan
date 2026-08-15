"""
Exit Engine for NiftyPulse Pro

Decides WHEN a trade should be closed.

Previously this only ever returned a fixed time-based exit — the ATR
stop, trailing stop, VSTOP flip and profit target were all listed as
"later this method will also check..." placeholders, even though
backtest/config.py already had flags for them. They are implemented here
now, so the config flags actually do something.

Exit rules are evaluated bar by bar from the entry bar forward, and the
FIRST one that triggers wins. Every check uses only data available at or
before that bar, so there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import backtest.config as cfg


@dataclass
class ExitResult:
    exit_index: int
    reason: str


class ExitEngine:

    def __init__(
        self,
        use_vstop_exit: bool = None,
        use_atr_stop: bool = None,
        use_trailing_stop: bool = None,
        use_profit_target: bool = None,
        atr_stop_multiplier: float = None,
        trailing_stop_multiplier: float = None,
        profit_target: float = None,
        max_hold_days: int = None,
    ):
        """
        Any argument left as None falls back to the value in
        backtest/config.py. Passing them explicitly lets a comparison
        harness try several exit strategies against the same signals
        without editing config.
        """
        self.use_vstop_exit = (
            cfg.USE_VSTOP_EXIT if use_vstop_exit is None else use_vstop_exit
        )
        self.use_atr_stop = (
            cfg.USE_ATR_STOP if use_atr_stop is None else use_atr_stop
        )
        self.use_trailing_stop = (
            cfg.USE_TRAILING_STOP if use_trailing_stop is None else use_trailing_stop
        )
        self.use_profit_target = (
            cfg.USE_PROFIT_TARGET if use_profit_target is None else use_profit_target
        )
        self.atr_stop_multiplier = (
            cfg.ATR_STOP_MULTIPLIER if atr_stop_multiplier is None else atr_stop_multiplier
        )
        self.trailing_stop_multiplier = (
            cfg.TRAILING_STOP_MULTIPLIER if trailing_stop_multiplier is None else trailing_stop_multiplier
        )
        self.profit_target = (
            cfg.PROFIT_TARGET if profit_target is None else profit_target
        )
        self.max_hold_days = (
            cfg.MAX_HOLD_DAYS if max_hold_days is None else max_hold_days
        )

    # ------------------------------------------------------------------

    def find_exit(self, future_df):
        """
        future_df:
            DataFrame beginning from the ENTRY BAR (index 0 = entry day),
            carrying the columns produced by compute_signals().

        Returns:
            ExitResult
        """

        max_days = min(self.max_hold_days, len(future_df) - 1)

        if max_days <= 0:
            return ExitResult(exit_index=0, reason="NO_DATA")

        entry_price = self._col(future_df, 0, "Open")
        if entry_price is None or entry_price <= 0:
            entry_price = self._col(future_df, 0, "Close")
        if entry_price is None or entry_price <= 0:
            return ExitResult(exit_index=max_days, reason="TIME_EXIT")

        # Highest close seen so far — the anchor for the trailing stop.
        highest_close = entry_price

        # NOTE ON "ATR_STOP":
        # compute_vstop() stores atr_mult * ATR in this column (i.e. the
        # VStop *distance*), NOT the raw ATR. Multiplying it again by a
        # stop multiplier gives a level ~6 ATRs below entry, which never
        # triggers — that's why ATR/trailing stops appeared to do nothing.
        # Divide the built-in multiplier back out first.
        def raw_atr(idx):
            v = self._col(future_df, idx, "ATR_STOP")
            if v is None:
                return None
            mult = getattr(cfg, "ATR_MULT", None)
            if mult is None:
                try:
                    from nse_scanner import ATR_MULT as _m
                    mult = _m
                except Exception:
                    mult = 3.0
            return v / mult if mult else v

        self._raw_atr = raw_atr

        # Fixed ATR stop is set once, from conditions at entry.
        atr_at_entry = raw_atr(0)
        atr_stop_level = None
        if self.use_atr_stop and atr_at_entry:
            atr_stop_level = entry_price - self.atr_stop_multiplier * atr_at_entry

        # Start at bar 1: bar 0 is the entry itself.
        for i in range(1, max_days + 1):

            close = self._col(future_df, i, "Close")
            low = self._col(future_df, i, "Low")
            if close is None:
                continue

            # --- Profit target (checked first: intrabar highs would hit
            # it before a close-based stop on the same bar) ---
            if self.use_profit_target:
                gain_pct = (close - entry_price) / entry_price * 100
                if gain_pct >= self.profit_target:
                    return ExitResult(exit_index=i, reason="PROFIT_TARGET")

            # --- Fixed ATR stop from entry ---
            if atr_stop_level is not None:
                trigger = low if low is not None else close
                if trigger <= atr_stop_level:
                    return ExitResult(exit_index=i, reason="ATR_STOP")

            # --- Trailing stop, ratcheting up with the highest close ---
            if self.use_trailing_stop:
                atr_now = self._raw_atr(i)
                if atr_now:
                    trail_level = highest_close - self.trailing_stop_multiplier * atr_now
                    trigger = low if low is not None else close
                    if trigger <= trail_level:
                        return ExitResult(exit_index=i, reason="TRAILING_STOP")

            # --- VSTOP trend reversal ---
            if self.use_vstop_exit:
                flip_down = self._col(future_df, i, "FLIP_DOWN")
                if flip_down:
                    return ExitResult(exit_index=i, reason="VSTOP_FLIP")

            if close > highest_close:
                highest_close = close

        return ExitResult(exit_index=max_days, reason="TIME_EXIT")

    # ------------------------------------------------------------------

    @staticmethod
    def _col(df, i, name):
        """Safely read a column at row i, returning None when absent/NaN."""
        try:
            if name not in df.columns:
                return None
            val = df.iloc[i][name]
            if val is None:
                return None
            # NaN check without importing numpy
            if val != val:
                return None
            if isinstance(val, (bool,)):
                return bool(val)
            return float(val)
        except Exception:
            return None
