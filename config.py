# Data
DATA_DIR = "data/cached"
TICKER = "NVDA"

# Features
RSI_PERIOD = 14
SMA_PERIODS = [5, 10, 20]
EMA_PERIODS = [12, 26]
MACD_SIGNAL_PERIOD = 9
VOLATILITY_PERIODS = [10, 20]
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 20

# Target
TARGET_COL = "direction"

# Split
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15

# Model
MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
}

# Backtesting
FEE = 0.0005
MAX_ALLOWED_DRAWDOWN = -0.30
TOP_N_STRATEGIES = 10
BUY_THRESHOLDS = [0.35, 0.40, 0.45, 0.50]
SELL_THRESHOLDS = [0.25, 0.30, 0.35, 0.40]
MAX_HOLD_DAYS_LIST = [5, 10, 15]
