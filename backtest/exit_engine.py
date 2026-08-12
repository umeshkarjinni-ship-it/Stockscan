"""
Exit Engine for NiftyPulse Pro

Responsible for deciding WHEN a trade should be closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import backtest.config as cfg


@dataclass
class ExitResult:
    exit_index: int
    reason: str


class ExitEngine:

    def __init__(self):
        pass

    def find_exit(self, future_df):

        """
        future_df:
            DataFrame beginning from ENTRY BAR.
            Index 0 = entry day.

        Returns:
            ExitResult
        """

        max_days = min(
            cfg.MAX_HOLD_DAYS,
            len(future_df) - 1
        )

        # -------------------------------------------------
        # Current implementation:
        # Time-based exit only.
        #
        # Later this method will also check:
        #   • ATR Stop
        #   • Trailing Stop
        #   • VSTOP Flip
        #   • Profit Target
        # -------------------------------------------------

        return ExitResult(
            exit_index=max_days,
            reason="TIME_EXIT",
        )