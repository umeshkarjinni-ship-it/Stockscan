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
#
# All values are ONE-WAY fractions of trade value. The simulator applies
# them to BOTH legs (entry and exit), which is what actually happens on a
# round trip.
#
# NOTE: this previously charged (BROKERAGE + SLIPPAGE) only ONCE per
# trade — 0.08% total — which understated real costs by roughly 4x and
# made backtest returns look better than they could ever be in practice.

BROKERAGE = 0.0003      # 0.03% — discount broker delivery rate
SLIPPAGE = 0.0005       # 0.05% — gap between signal close and real fill

# Indian equity delivery statutory charges (one-way unless noted).
# Adjust to match your actual broker's contract note.
STT = 0.001             # 0.1% Securities Transaction Tax, both legs
EXCHANGE_CHARGES = 0.0000325   # NSE transaction charge
GST_ON_CHARGES = 0.18   # 18% GST, applied to brokerage + exchange charges
STAMP_DUTY = 0.00015    # 0.015% — BUY side only

# Total one-way cost, excluding the buy-only stamp duty.
ONE_WAY_COST = (
    BROKERAGE
    + SLIPPAGE
    + STT
    + EXCHANGE_CHARGES
    + (BROKERAGE + EXCHANGE_CHARGES) * GST_ON_CHARGES
)

# Full round-trip cost as a PERCENTAGE, ready to subtract from a
# percentage return. Works out to roughly 0.33%.
ROUND_TRIP_COST_PCT = (ONE_WAY_COST * 2 + STAMP_DUTY) * 100

# ---------------------------------------------------------------------
# Holding periods
# ---------------------------------------------------------------------

HOLDING_PERIODS = [5, 10, 20, 60]

# ---------------------------------------------------------------------
# Exit Strategy
# ---------------------------------------------------------------------

# Fallback maximum holding period (used if no other exit occurs)
MAX_HOLD_DAYS = 40

# ---------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------
#
# MEASURED RESULT — do not enable these casually.
#
# These flags did nothing until exit_engine.py was actually implemented.
# The first run with all three stops active gave:
#
#     stops ON  : PF 1.27, win 36.7%, avg hold 17.6 bars
#     stops OFF : PF 2.15, win 57.0%, avg hold 39.9 bars
#
# The stops worked as designed — average loser improved from -10.1% to
# -6.2% — but they also cut short the winners the strategy depends on,
# and net profit factor fell ~40%. That's typical for trend following:
# the edge comes from letting a minority of large winners run (best trade
# here was +146%), so anything that truncates them is expensive.
#
# Defaults are therefore OFF. Use compare_exits.py to test variations
# against your own signal set before turning any of them back on.

# Exit on VSTOP trend reversal
USE_VSTOP_EXIT = False

# ATR Stop Loss
USE_ATR_STOP = False
ATR_STOP_MULTIPLIER = 2.0

# ATR Trailing Stop
USE_TRAILING_STOP = False
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