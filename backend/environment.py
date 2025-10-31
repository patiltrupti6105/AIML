# backend/environment.py
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import ta  # technical indicators


class StockTradingEnv(Env):
    """
    Custom Stock Trading Environment for Reinforcement Learning.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    Observation: [SMA_10, EMA_20, RSI, MACD, Volume, Balance, SharesHeld]
    Reward: Change in portfolio value (cash + shares*price) between timesteps
    """

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000):
        super(StockTradingEnv, self).__init__()

        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        # ---------- Data Preprocessing ----------
        df = df.copy()
        # Ensure expected columns
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' and 'Volume' columns")

        # Add technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()

        # Drop rows with NaNs created by indicators
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("DataFrame is empty after calculating indicators. Provide more data.")

        self.df = df

        # ---------- Environment attributes ----------
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.shares = 0
        self.prev_portfolio_value = self.initial_balance  # Track previous portfolio value

        self.current_step = 0

        # Action space: 0 = Hold, 1 = Buy (one share), 2 = Sell (sell all shares)
        self.action_space = Discrete(3)

        # Observation space: 7 floats (SMA_10, EMA_20, RSI, MACD, Volume, balance, shares_held)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)


    def _get_obs(self):
        """Return the observation for the current step as a numpy array."""
        row = self.df.loc[self.current_step]
        obs = np.array([
            row['SMA_10'],
            row['EMA_20'],
            row['RSI'],
            row['MACD'],
            row['Volume'],
            self.balance,
            float(self.shares_held)
        ], dtype=np.float32)
        return obs

    def _get_current_price(self):
        return float(self.df.loc[self.current_step, 'Close'])

    def _get_portfolio_value(self):
        current_price = self._get_current_price()
        return float(self.balance + self.shares_held * current_price)

    def step(self, action):
        """
        Execute one step in the environment.
        - action: 0 (Hold), 1 (Buy one share), 2 (Sell all shares)
        Adds transaction fee (commission %) and slippage (price impact).
        Returns: obs, reward, done, info
        """
        if action not in [0, 1, 2]:
            raise ValueError("Invalid action.")

        current_price = self._get_current_price()
        prev_value = self._get_portfolio_value()

        # Transaction fee and slippage
        transaction_fee_percent = 0.001  # 0.1% per trade
        slippage_percent = 0.0005        # 0.05% price impact

        # Adjust executed price for slippage
        executed_price = current_price * (1 + slippage_percent if action == 1 else 1 - slippage_percent)

        cost = 0  # Initialize cost for reward calculation

        # Execute action
        if action == 1 and self.balance >= executed_price:
            # Buy one share
            self.shares_held += 1
            cost = executed_price * transaction_fee_percent
            self.balance -= executed_price + cost

        elif action == 2 and self.shares_held > 0:
            # Sell all shares
            revenue = self.shares_held * executed_price
            cost = revenue * transaction_fee_percent
            self.balance += revenue - cost
            self.shares_held = 0

        # Advance to next time step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Calculate reward based on portfolio change
        portfolio_value = self._get_portfolio_value()
        reward = portfolio_value - self.prev_portfolio_value - cost
        self.prev_portfolio_value = portfolio_value

        # Small penalty for holding
        if action == 0:
            reward -= 0.1

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price,
            "portfolio_value": portfolio_value
        }

        return obs, float(reward), done, False, info

    def reset(self, seed=None, options=None):
        """
        Reset environment to the starting state and return initial observation.
        Args:
            seed: Optional random seed
            options: Optional configuration
        Returns:
            tuple (observation, info)
        """
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        self.current_step = 0
        
        # Initial info dict
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": self._get_current_price(),
            "portfolio_value": self._get_portfolio_value()
        }
        return self._get_obs(), info
