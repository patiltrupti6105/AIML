# trading_environment.py
import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import config

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger("env")


# -----------------------------------------------------------------------------
# Lightweight TA helpers
# -----------------------------------------------------------------------------
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class EnvCfg:
    initial_capital: float = config.INITIAL_CAPITAL
    lookback_window: int = config.LOOKBACK_WINDOW
    commission: float = config.COMMISSION
    slippage: float = config.SLIPPAGE
    max_position_pct: float = config.MAX_POSITION_PCT
    stop_loss_pct: float = config.STOP_LOSS_PCT
    take_profit_pct: float = config.TAKE_PROFIT_PCT


# -----------------------------------------------------------------------------
# Trading Environment
# -----------------------------------------------------------------------------
class TradingEnvironment(gym.Env):
    """
    Actions:
        0 = Hold
        1 = Buy 1 share
        2 = Sell all

    Observation:
        [lookback returns] + 13 scalars
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, symbol: str = "AAPL", env_cfg: Optional[EnvCfg] = None):
        super().__init__()
        self.symbol = symbol
        self.cfg = env_cfg or EnvCfg()

        # --- Prepare DataFrame ---
        self.df = df.copy()
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        self.df = self.df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        self.df.reset_index(drop=True, inplace=True)

        # Basic features
        c = self.df["Close"]
        self.df["ret"] = c.pct_change().fillna(0.0)

        # Indicators
        self.df["sma10"] = _sma(c, 10)
        self.df["sma20"] = _sma(c, 20)
        self.df["ema12"] = _ema(c, 12)
        self.df["ema26"] = _ema(c, 26)
        self.df["macd"] = self.df["ema12"] - self.df["ema26"]
        self.df["macd_signal"] = _ema(self.df["macd"], 9)
        self.df["rsi"] = _rsi(c, getattr(config, "RSI_PERIOD", 14))

        # Bollinger position [-1, 1]
        bb_period = getattr(config, "BB_PERIOD", 20)
        bb_std = getattr(config, "BB_STD", 2)
        mavg = c.rolling(bb_period, min_periods=1).mean()
        mstd = c.rolling(bb_period, min_periods=1).std().replace(0, np.nan).fillna(1e-12)
        upper = mavg + bb_std * mstd
        lower = mavg - bb_std * mstd
        self.df["bb_pos"] = ((c - lower) / (upper - lower + 1e-12)).clip(0, 1) * 2 - 1

        self.df["atr"] = _atr(self.df, getattr(config, "ATR_PERIOD", 14))

        # Z-scores
        for col in ["sma10", "sma20", "ema12", "ema26", "atr"]:
            mu = self.df[col].rolling(50, min_periods=1).mean()
            sd = self.df[col].rolling(50, min_periods=1).std().replace(0, np.nan).fillna(1e-12)
            self.df[col + "_z"] = ((self.df[col] - mu) / sd).clip(-10, 10)

        # Spaces
        self.action_space = Discrete(3)
        # we append 13 scalars below
        obs_len = self.cfg.lookback_window + 13
        self.observation_space = Box(low=-10.0, high=10.0, shape=(obs_len,), dtype=np.float32)

        # Episode state
        self.reset_state()

    # -------------------------------------------------------------------------
    # Gym API
    # -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.reset_state()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        done = False
        info: Dict = {}

        # Price at current step (positional)
        price = float(self.df.iloc[self.current_step]["Close"])
        start_nav = self.net_worth

        # Execute
        if action == 1:
            self._buy_one(price, info)
        elif action == 2:
            self._sell_all(price, info)

        # Risk
        self._apply_risk(price, info)

        # Advance time
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        next_price = float(self.df.iloc[self.current_step]["Close"])
        self.position_value = self.shares_held * next_price
        self.net_worth = self.balance + self.position_value

        reward = (self.net_worth - start_nav)

        obs = self._get_obs()
        info.update(
            {
                "step": self.current_step,
                "price": next_price,
                "balance": self.balance,
                "shares": self.shares_held,
                "position_value": self.position_value,
                "net_worth": self.net_worth,
                "action": int(action),
                "symbol": self.symbol,
            }
        )
        return obs, float(reward), done, False, info

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def reset_state(self):
        self.current_step = max(self.cfg.lookback_window - 1, 0)
        self.balance = float(self.cfg.initial_capital)
        self.shares_held = 0
        self.position_value = 0.0
        self.net_worth = self.balance
        self.max_position_value = self.cfg.max_position_pct * self.cfg.initial_capital

    def _cost_multiplier(self) -> float:
        return self.cfg.commission + self.cfg.slippage

    def _buy_one(self, price: float, info: Dict):
        cost_mult = 1.0 + self._cost_multiplier()
        total_cost = price * cost_mult
        new_pos_value = (self.shares_held + 1) * price

        if total_cost <= self.balance and new_pos_value <= self.max_position_value:
            self.balance -= total_cost
            self.shares_held += 1
            info["buy_executed"] = True
        else:
            info["buy_executed"] = False

        self.position_value = self.shares_held * price
        self.net_worth = self.balance + self.position_value

    def _sell_all(self, price: float, info: Dict):
        if self.shares_held > 0:
            proceeds = self.shares_held * price * (1.0 - self._cost_multiplier())
            self.balance += proceeds
            self.shares_held = 0
            info["sell_executed"] = True
        else:
            info["sell_executed"] = False

        self.position_value = 0.0
        self.net_worth = self.balance

    def _apply_risk(self, price: float, info: Dict):
        if self.shares_held <= 0:
            return
        entry_value = self.position_value
        if entry_value <= 0:
            return
        pnl_ratio = (self.shares_held * price - entry_value) / (entry_value + 1e-12)

        stop_loss = -abs(self.cfg.stop_loss_pct)
        take_profit = abs(self.cfg.take_profit_pct)
        if pnl_ratio <= stop_loss or pnl_ratio >= take_profit:
            self._sell_all(price, info)
            info["risk_exit"] = True

    # -------------------------------------------------------------------------
    # Observation (positional indexing only)
    # -------------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        # Lookback returns
        start = max(0, self.current_step - self.cfg.lookback_window + 1)
        window = self.df.iloc[start : self.current_step + 1]
        rets = window["ret"].to_numpy(dtype=np.float32)
        if len(rets) < self.cfg.lookback_window:
            pad = np.zeros(self.cfg.lookback_window - len(rets), dtype=np.float32)
            rets = np.concatenate([pad, rets], axis=0)

        # Current row
        row = self.df.iloc[self.current_step]
        price = float(row["Close"])

        # 50-bar MA over last 50 closes (safe if <50)
        ma_start = max(0, self.current_step - 49)
        ma50 = float(self.df["Close"].iloc[ma_start : self.current_step + 1].mean())
        if not np.isfinite(ma50) or ma50 <= 0:
            ma50 = price
        price_norm = np.tanh(price / (ma50 + 1e-12))

        balance_norm = np.tanh(self.balance / (self.cfg.initial_capital + 1e-9))
        shares_norm = np.tanh(self.shares_held / 1000.0)
        position_value_norm = np.tanh((self.shares_held * price) / (self.cfg.initial_capital + 1e-9))

        rsi = float(row.get("rsi", 50.0)) / 100.0
        sma10_z = float(row.get("sma10_z", 0.0))
        sma20_z = float(row.get("sma20_z", 0.0))
        ema12_z = float(row.get("ema12_z", 0.0))
        ema26_z = float(row.get("ema26_z", 0.0))
        macd = float(row.get("macd", 0.0))
        macd_signal = float(row.get("macd_signal", 0.0))
        bb_pos = float(row.get("bb_pos", 0.0))
        atr_z = float(row.get("atr_z", 0.0))

        scalars = np.array(
            [
                balance_norm,
                shares_norm,
                price_norm,
                position_value_norm,
                rsi,
                sma10_z,
                sma20_z,
                ema12_z,
                ema26_z,
                macd,
                macd_signal,
                bb_pos,
                atr_z,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([rets, scalars], axis=0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    # -------------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------------
    def render(self):
        price = float(self.df.iloc[self.current_step]["Close"])
        print(
            f"[{self.symbol}] step={self.current_step} price={price:.2f} "
            f"bal={self.balance:.2f} shares={self.shares_held} "
            f"pos_val={self.position_value:.2f} nav={self.net_worth:.2f}"
        )