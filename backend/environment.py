# backend/environment.py
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import ta
from typing import Tuple, Dict, Any


class StockTradingEnv(Env):
    """
    Stock Trading Environment :
      - Actions: 0=Hold, 1=Buy , 2=Sell 
      - Supports fractional shares, transaction costs, slippage
      - Observations include indicator distances + short price window
      - Returns (obs, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000.0,
        max_position_value: float = None,
        commission: float = 1e-3,   # fraction e.g., 0.001 = 0.1%
        slippage: float = 1e-3,
        window: int = 10,
        max_steps: int = None,
    ):
        super().__init__()

        # Basic checks
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        required_cols = {"Close", "High", "Low", "Volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Keep a copy
        df = df.copy().reset_index(drop=True)

        # Compute indicators (must drop NaNs)
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

        # Config
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.shares_held = 0.0  
        self.max_position_value = max_position_value or (self.initial_balance * 2)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.window = int(window)
        self.current_step = 0
        self.prev_portfolio_value = float(initial_balance)
        self.max_steps = max_steps or len(df)

        # Spaces
        # Number of features: 7 indicators + window returns + position flag
        obs_dim = 7 + self.window + 1
        self.action_space = Discrete(3)
        # Observations roughly within -10..10 (Box low/high large to be safe)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # ---------- Helpers ----------
    def _get_price(self, step=None) -> float:
        step = self.current_step if step is None else step
        return float(self.df.loc[step, "Close"])

    def get_portfolio_value(self) -> float:
        price = self._get_price()
        return float(self.balance + self.shares_held * price)

    def _get_obs(self):
        """Return observation for current step: indicators + recent returns + position flag"""
        row = self.df.loc[self.current_step]
        price = float(row["Close"])

        sma10_dist = (price - row["SMA_10"]) / (row["SMA_10"] + 1e-9)
        ema20_dist = (price - row["EMA_20"]) / (row["EMA_20"] + 1e-9)
        ema50_dist = (price - row["EMA_50"]) / (row["EMA_50"] + 1e-9)
        rsi_norm = (row["RSI"] - 50.0) / 50.0  # -1..1
        macd_rel = (row["MACD"] - row["MACD_Signal"]) / (abs(row["MACD_Signal"]) + 1e-9)
        balance_norm = (self.balance - self.initial_balance) / (self.initial_balance + 1e-9)
        holdings_norm = (self.shares_held * price) / (self.initial_balance + 1e-9)

        indicators = [sma10_dist, ema20_dist, ema50_dist, rsi_norm, macd_rel, balance_norm, holdings_norm]

        # window of recent returns (normalized)
        start = max(0, self.current_step - self.window + 1)
        window_prices = self.df.loc[start:self.current_step, "Close"].values
        # pad if necessary
        if len(window_prices) < self.window:
            pad = np.full(self.window - len(window_prices), window_prices[0] if len(window_prices) > 0 else 0.0)
            window_prices = np.concatenate([pad, window_prices])
        window_returns = np.diff(window_prices) / (window_prices[:-1] + 1e-9)
        # if window_returns shorter, pad
        if len(window_returns) < self.window:
            window_returns = np.concatenate([np.zeros(self.window - len(window_returns)), window_returns])

        pos_flag = np.array([self.shares_held * price / (self.max_position_value + 1e-9)])  # scaled position exposure

        obs = np.concatenate([np.array(indicators, dtype=np.float32), window_returns.astype(np.float32), pos_flag.astype(np.float32)])
        return obs

    # ---------- Step ----------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step. Returns:
            obs, reward, terminated, truncated, info
        Action semantics:
            0 = hold
            1 = buy (increase exposure by a fraction)
            2 = sell (decrease exposure by a fraction)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        price = self._get_price()
        prev_portfolio = self.get_portfolio_value()

        invalid_action = False
        trade_info = None

        # define fractional trade size as fraction of max_position_value
        # e.g. buy_fraction = 0.1 => attempt to increase position by 10% of max position
        trade_fraction = 0.1
        

        if action == 1:  # BUY
            desired_increase = self.max_position_value * trade_fraction
            cash_needed = desired_increase / (1 + self.commission + self.slippage)
            if self.balance >= cash_needed:
                qty = cash_needed / price
                # apply slippage and commission as cost
                cost = qty * price * (1 + self.commission + self.slippage)
                self.balance -= cost
                self.shares_held += qty
                trade_info = ("buy", qty, price, cost)
            else:
                invalid_action = True

        elif action == 2:  # SELL
            desired_decrease_value = self.max_position_value * trade_fraction
            qty = min(self.shares_held, desired_decrease_value / price)
            if qty > 1e-12:
                proceeds = qty * price * (1 - (self.commission + self.slippage))
                self.balance += proceeds
                self.shares_held -= qty
                trade_info = ("sell", qty, price, proceeds)
            else:
                invalid_action = True
        # action == 0: hold

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1 or (self.current_step >= self.max_steps)
        truncated = False

        # New portfolio value
        current_price = self._get_price() if not terminated else float(self.df.loc[len(self.df) - 1, "Close"])
        portfolio_value = float(self.balance + self.shares_held * current_price)

        # Reward: use portfolio % change minus tiny penalty for trade frequency
        if abs(prev_portfolio) < 1e-9:
            reward = 0.0
        else:
            pnl_pct = (portfolio_value - prev_portfolio) / (abs(prev_portfolio) + 1e-9)
            # small trade penalty encourages selective trades
            trade_penalty = 0.0
            if trade_info is not None:
                trade_penalty = 1e-4
            reward = pnl_pct - trade_penalty

        # Penalize invalid actions
        if invalid_action:
            reward -= 0.005

        self.prev_portfolio_value = portfolio_value

        obs = self._get_obs()
        info = {
            "step": int(self.current_step),
            "price": float(current_price),
            "balance": float(self.balance),
            "shares_held": float(self.shares_held),
            "portfolio_value": float(portfolio_value),
            "trade": trade_info,
            "invalid_action": invalid_action,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------- Reset ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.current_step = 0
        self.prev_portfolio_value = float(self.initial_balance)
        return self._get_obs(), {}

    # ---------- Utilities ----------
    def render(self, mode="human"):
        pv = self.get_portfolio_value()
        print(f"Step: {self.current_step}, Price: {self._get_price():.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.6f}, Portfolio: {pv:.2f}")
