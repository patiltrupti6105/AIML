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
        1 = Buy (1 share)
        2 = Sell (1 share)
    Observation:
        [SMA_10, EMA_20, EMA_50, RSI, MACD, Balance_norm, Holdings_norm]
    """

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000):
        super().__init__()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        if not {"Close", "High", "Low", "Volume"}.issubset(df.columns):
            raise ValueError("DataFrame must have Close, High, Low, and Volume columns.")

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

        # Initialize environment state
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.shares_held = 0
        self.max_shares = 50  # cap holdings for realism
        self.current_step = 0
        self.prev_value = self.initial_balance

        # Spaces
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    # ======= Helpers =======
    def _get_price(self):
        return float(self.df.loc[self.current_step, "Close"])

    def _get_value(self):
        return self.balance + self.shares_held * self._get_price()

    def _get_obs(self):
        row = self.df.loc[self.current_step]
        price = row["Close"]
        obs = np.array([
            (price - row["SMA_10"]) / row["SMA_10"],
            (price - row["EMA_20"]) / row["EMA_20"],
            (price - row["EMA_50"]) / row["EMA_50"],
            (row["RSI"] - 50) / 50,
            (row["MACD"] - row["MACD_Signal"]) / (abs(row["MACD_Signal"]) + 1e-6),
            self.balance / self.initial_balance,
            (self.shares_held * price) / self.initial_balance,
        ], dtype=np.float32)
        return obs

    # ======= Step =======
    def step(self, action):
        assert self.action_space.contains(action)
        price = self._get_price()
        prev_value = self._get_value()

        reward = 0.0
        invalid = False

        # Indicator trends
        row = self.df.loc[self.current_step]
        bullish_trend = row["SMA_10"] > row["EMA_20"] > row["EMA_50"]
        bearish_trend = row["SMA_10"] < row["EMA_20"] < row["EMA_50"]
        macd_bull = row["MACD"] > row["MACD_Signal"]
        macd_bear = row["MACD"] < row["MACD_Signal"]
        rsi = row["RSI"]

        buy_signal = bullish_trend and macd_bull and rsi < 65
        sell_signal = bearish_trend and macd_bear and rsi > 35

        # ---- Execute trade ----
        if action == 1:  # BUY
            if self.balance >= price and self.shares_held < self.max_shares:
                if buy_signal:
                    self.balance -= price
                    self.shares_held += 1
                else:
                    invalid = True
            else:
                invalid = True

        elif action == 2:  # SELL
            if self.shares_held > 0:
                if sell_signal:
                    self.balance += price
                    self.shares_held -= 1
                else:
                    invalid = True
            else:
                invalid = True

        # Move forward
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_value = self._get_value()

        # ---- Reward ----
        reward = (current_value - prev_value) / prev_value

        # Bonus / penalties
        if action == 1 and buy_signal:
            reward += 0.1
        if action == 2 and sell_signal:
            reward += 0.1
        if invalid:
            reward -= 0.5  # discourage nonsense trades

        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "price": price,
            "portfolio_value": current_value,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "invalid_action": invalid,
        }

        return self._get_obs(), float(reward), done, False, info

    # ======= Reset =======
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.prev_value = self.initial_balance
        return self._get_obs(), {}
