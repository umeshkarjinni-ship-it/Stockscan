"""
Utility functions for backtesting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------

def setup_logger(log_file: Path):

    logger = logging.getLogger("Backtest")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# -------------------------------------------------------------

def pct_return(entry, exit_price):

    if entry == 0:
        return np.nan

    return ((exit_price - entry) / entry) * 100


# -------------------------------------------------------------

def max_drawdown(equity):

    equity = pd.Series(equity)

    peak = equity.cummax()

    dd = (equity - peak) / peak

    return float(dd.min() * 100)


# -------------------------------------------------------------

def cagr(start_value, end_value, years):

    if years <= 0:
        return np.nan

    return ((end_value / start_value) ** (1 / years) - 1) * 100


# -------------------------------------------------------------

def safe_float(x, default=np.nan):

    try:
        return float(x)
    except Exception:
        return default


# -------------------------------------------------------------

def save_checkpoint(file, obj):

    with open(file, "w") as f:
        json.dump(obj, f, indent=2)


# -------------------------------------------------------------

def load_checkpoint(file):

    file = Path(file)

    if not file.exists():
        return {}

    with open(file) as f:
        return json.load(f)


# -------------------------------------------------------------

def ensure_dataframe(df):

    if df is None:
        return pd.DataFrame()

    return df.copy()


# -------------------------------------------------------------

def rolling_returns(close, days):

    return close.shift(-days) / close - 1