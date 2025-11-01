# backend/environment.py
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import ta


class StockTradingEnv(Env):
    """
    Realistic Stock Trading Environment.
    Actions:
        0 = Hold
        1 = Buy (buy 1 share)
        2 = Sell (sell 1 share)
    Observation:
        [SMA_10_dist, EMA_20_dist, EMA_50_dist, RSI_norm, MACD_rel, Balance_norm, Holdings_norm]
    Returns:
        obs, reward, terminated, truncated, info
    """

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000):
        super().__init__()

        # Basic checks
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        required_cols = {"Close", "High", "Low", "Volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Compute indicators
        df = df.copy()
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.df = df

        # State
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.shares_held = 0
        self.max_shares = 50
        self.current_step = 0
        self.prev_portfolio_value = float(initial_balance)

        # Spaces
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    # ---------- Helpers ----------
    def _get_price(self):
        return float(self.df.loc[self.current_step, "Close"])

    def _get_value(self):
        return float(self.balance + self.shares_held * self._get_price())

    def _get_obs(self):
        """Return observation for current step."""
        row = self.df.loc[self.current_step]
        price = row["Close"]

        sma10_dist = (price - row["SMA_10"]) / (row["SMA_10"] + 1e-9)
        ema20_dist = (price - row["EMA_20"]) / (row["EMA_20"] + 1e-9)
        ema50_dist = (price - row["EMA_50"]) / (row["EMA_50"] + 1e-9)
        rsi_norm = (row["RSI"] - 50.0) / 50.0  # -1..1
        macd_rel = (row["MACD"] - row["MACD_Signal"]) / (abs(row["MACD_Signal"]) + 1e-9)
        balance_norm = self.balance / (self.initial_balance + 1e-9)
        holdings_norm = (self.shares_held * price) / (self.initial_balance + 1e-9)

        obs = np.array([
            sma10_dist,
            ema20_dist,
            ema50_dist,
            rsi_norm,
            macd_rel,
            balance_norm,
            holdings_norm
        ], dtype=np.float32)
        return obs

    # ---------- Step ----------
    def step(self, action):
        """
        Execute one step. Returns:
            obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        price = self._get_price()
        prev_portfolio = self._get_value()

        invalid_action = False
        # Execute action (buy/sell 1 share)
        if action == 1:  # BUY 1 share
            if self.balance >= price and self.shares_held < self.max_shares:
                self.balance -= price
                self.shares_held += 1
            else:
                # not enough cash or at max shares -> invalid
                invalid_action = True

        elif action == 2:  # SELL 1 share
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += price
            else:
                invalid_action = True

        # action == 0 -> hold (allowed)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # not using Gym truncation here

        # New portfolio value
        current_price = self._get_price() if not terminated else float(self.df.loc[len(self.df)-1, "Close"])
        portfolio_value = float(self.balance + self.shares_held * current_price)

        # Reward: relative portfolio change
        # Small penalty for holding to encourage reasonable activity
        if prev_portfolio <= 0:
            reward = 0.0
        else:
            reward = (portfolio_value - prev_portfolio) / (abs(prev_portfolio) + 1e-9)

        # Bonus / penalty
        if invalid_action:
            reward -= 0.5  # discourage invalid attempts
        else:
            # small bonus when action aligns with simple indicator heuristics (optional)
            row = self.df.loc[max(0, self.current_step - 1)]
            bullish = row["SMA_10"] > row["EMA_20"] > row["EMA_50"]
            bearish = row["SMA_10"] < row["EMA_20"] < row["EMA_50"]
            macd_bull = row["MACD"] > row["MACD_Signal"]
            macd_bear = row["MACD"] < row["MACD_Signal"]
            rsi = row["RSI"]

            if action == 1 and bullish and macd_bull and rsi < 65:
                reward += 0.05
            if action == 2 and bearish and macd_bear and rsi > 35:
                reward += 0.05

        # Update prev value for next step
        self.prev_portfolio_value = portfolio_value

        obs = self._get_obs()

        info = {
            "step": int(self.current_step),
            "price": float(current_price),
            "balance": float(self.balance),
            "shares_held": int(self.shares_held),
            "portfolio_value": float(portfolio_value),
            "invalid_action": bool(invalid_action)
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------- Reset ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        self.current_step = 0
        self.prev_portfolio_value = float(self.initial_balance)
        return self._get_obs(), {}
