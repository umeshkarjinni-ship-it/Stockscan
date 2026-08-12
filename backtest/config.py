"""
Backtest configuration
NiftyPulse Pro
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIGNALS_DIR = PROJECT_ROOT / "signals"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_DIR = PROJECT_ROOT / "logs"

for folder in (
    SIGNALS_DIR,
    REPORTS_DIR,
    CACHE_DIR,
    LOG_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Historical period
# ---------------------------------------------------------------------

START_DATE = "2018-01-01"
END_DATE = None

# ---------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------

INITIAL_CAPITAL = 1_000_000

# ---------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------

BROKERAGE = 0.0003
SLIPPAGE = 0.0005

# ---------------------------------------------------------------------
# Holding periods
# ---------------------------------------------------------------------

HOLDING_PERIODS = [5, 10, 20, 60]

# ---------------------------------------------------------------------
# Exit Strategy
# ---------------------------------------------------------------------

# Fallback maximum holding period (used if no other exit occurs)
MAX_HOLD_DAYS = 40

# Exit on VSTOP trend reversal
USE_VSTOP_EXIT = True

# ATR Stop Loss
USE_ATR_STOP = True
ATR_STOP_MULTIPLIER = 2.0

# ATR Trailing Stop
USE_TRAILING_STOP = True
TRAILING_STOP_MULTIPLIER = 2.5

# Optional fixed profit target (%)
USE_PROFIT_TARGET = False
PROFIT_TARGET = 15.0
# ---------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------

POSITION_MODE = "equal"

MAX_OPEN_POSITIONS = 10

RISK_PER_TRADE = 0.01

# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

BENCHMARK = "^NSEI"

RISK_FREE_RATE = 0.06

# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------

SIGNALS_FILE = SIGNALS_DIR / "backtest_signals.csv"
TRADES_FILE = SIGNALS_DIR / "backtest_trades.csv"

SUMMARY_FILE = REPORTS_DIR / "backtest_summary.csv"
EQUITY_FILE = REPORTS_DIR / "equity_curve.csv"

CHECKPOINT_FILE = CACHE_DIR / "checkpoint.json"

LOG_FILE = LOG_DIR / "backtest.log"