# config.py
import os

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_model.zip")

for _p in (DATA_DIR, RESULTS_DIR, MODEL_DIR):
    os.makedirs(_p, exist_ok=True)

# ── Symbols / Trading session ─────────────────────────────────────────────────
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]  # train/trade defaults
INITIAL_CAPITAL = 100_000.0
UPDATE_INTERVAL = 5          # seconds between realtime updates
DEFAULT_DURATION = 60        # minutes for realtime trading
LOOKBACK_DAYS = 60           # days of history to seed buffers
LOOKBACK_WINDOW = 30         # steps of returns in observation

# Costs / risk mgmt used by env & realtime trader
COMMISSION = 0.0005          # 0.05% per trade
SLIPPAGE  = 0.0005           # 0.05%
MAX_POSITION_PCT = 0.9       # cap position size vs initial balance
STOP_LOSS_PCT = 0.05         # 5%
TAKE_PROFIT_PCT = 0.08       # 8%

# ── Indicators used by environment ────────────────────────────────────────────
SMA_PERIODS = [10, 20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
STOCHASTIC_PERIOD = 14
ROC_PERIOD = 12
MACD_SLOW = 26
MACD_FAST = 12
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
ADX_PERIOD = 14

# ── Training config (used by train.py / SB3 DQN) ──────────────────────────────
TRAIN_SPLIT = 0.8
TRAIN_TIMESTEPS = 50_000      # change as needed

LEARNING_RATE = 2.5e-4
BUFFER_SIZE = 100_000
LEARNING_STARTS = 1_000
BATCH_SIZE = 256
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 1_000
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.05
HIDDEN_LAYERS = [256, 256]

EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5

def print_config():
    print("=" * 70)
    print("CONFIG")
    print("=" * 70)
    print(f"Symbols: {SYMBOLS}")
    print(f"Data Dir:     {DATA_DIR}")
    print(f"Results Dir:  {RESULTS_DIR}")
    print(f"Model Dir:    {MODEL_DIR}")
    print(f"Model Path:   {MODEL_PATH}")
    print(f"Initial Cap:  ${INITIAL_CAPITAL:,.2f}")
    print(f"Log Level:    {LOG_LEVEL}")
    print("=" * 70)
