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

        # Simple trading constraints
        self.max_shares = 100  # Maximum number of shares to hold at once
        
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        # ---------- Data Preprocessing ----------
        df = df.copy()
        # Ensure expected columns
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' and 'Volume' columns")

        # Add technical indicators
        # Trend indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Momentum indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        
        # Price relative to moving averages (trend strength)
        df['Price_vs_MA'] = (df['Close'] - df['EMA_20']) / df['EMA_20'] * 100  # Percent from EMA20

        # Drop rows with NaNs created by indicators
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("DataFrame is empty after calculating indicators. Provide more data.")

        self.df = df

        # ---------- Environment attributes ----------
        # Use the constructor initial_balance and ensure consistent attribute names
        self.initial_balance = float(initial_balance)
        self.balance = float(self.initial_balance)
        self.shares_held = 0
        self.prev_portfolio_value = float(self.initial_balance)  # Track previous portfolio value
        # Track average buy price (cost basis per share, including fees)
        self.avg_buy_price = 0.0

        self.current_step = 0

        # Action space: 0 = Hold, 1 = Buy (one share), 2 = Sell (sell all shares)
        self.action_space = Discrete(3)

        # Observation space: 9 normalized features
        # [sma10_dist, ema20_dist, ema50_dist, rsi_norm, macd_norm, macd_hist_norm, 
        #  trend_strength, balance_norm, holdings_value_norm]
        self.observation_space = Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )


    def _get_obs(self):
        """Return the observation for the current step as a numpy array."""
        row = self.df.loc[self.current_step]
        
        # Get current price and normalize indicators
        current_price = self._get_current_price()
        
        # Portfolio state (normalized)
        balance_norm = float(self.balance) / float(self.initial_balance)
        holdings_value_norm = (float(self.shares_held) * current_price) / float(self.initial_balance)
        
        # Trend signals (normalized)
        sma10_dist = (row['SMA_10'] - current_price) / current_price
        ema20_dist = (row['EMA_20'] - current_price) / current_price
        ema50_dist = (row['EMA_50'] - current_price) / current_price
        
        # Momentum (already normalized)
        rsi_norm = row['RSI'] / 100.0
        
        # MACD (normalize by ATR to scale with volatility)
        macd_norm = row['MACD'] / row['ATR'] if row['ATR'] != 0 else 0
        macd_hist_norm = row['MACD_Hist'] / row['ATR'] if row['ATR'] != 0 else 0
        
        # Trend strength
        trend_strength = row['Price_vs_MA'] / 100.0  # Already in percent, normalize to -1 to 1 range
        
        obs = np.array([
            sma10_dist,          # Distance from price to SMA10
            ema20_dist,          # Distance from price to EMA20
            ema50_dist,          # Distance from price to EMA50
            rsi_norm,            # RSI (normalized 0-1)
            macd_norm,           # MACD (normalized by ATR)
            macd_hist_norm,      # MACD histogram (normalized by ATR)
            trend_strength,      # Price vs MA trend strength
            balance_norm,        # Normalized balance remaining
            holdings_value_norm  # Normalized position size
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

        # Adjust executed price for slippage depending on buy/sell
        if action == 1:  # buy
            executed_price = current_price * (1 + slippage_percent)
        elif action == 2:  # sell
            executed_price = current_price * (1 - slippage_percent)
        else:
            executed_price = current_price

        cost = 0  # Initialize cost for reward calculation

        # Execute action
        if action == 1:  # Buy checks
            # Can only buy if: have enough cash AND not at max shares
            can_buy = (self.balance >= executed_price * (1 + transaction_fee_percent) and 
                      self.shares_held < self.max_shares)
            if can_buy:
                # Buy one share
                fee = executed_price * transaction_fee_percent
                total_cost = executed_price + fee
                self.shares_held += 1
                self.balance -= total_cost
                cost = fee  # Only count fee in reward penalty

        elif action == 2:  # Sell checks
            # Can only sell if we have shares
            if self.shares_held > 0:
                # Sell one share
                fee = executed_price * transaction_fee_percent
                revenue_after_fee = executed_price - fee
                self.balance += revenue_after_fee
                self.shares_held -= 1
                cost = fee

        # Advance to next time step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Calculate reward based on indicators and trade results
        portfolio_value = self._get_portfolio_value()
        value_change = portfolio_value - self.prev_portfolio_value
        
        # Get current indicators
        row = self.df.loc[self.current_step]
        
        # 1. Trend Analysis (multiple timeframes)
        short_trend = row['SMA_10'] > row['EMA_20']  # Short-term trend
        medium_trend = row['EMA_20'] > row['EMA_50']  # Medium-term trend
        trend_strength = abs(row['Price_vs_MA'])      # Trend strength
        
        # 2. Momentum
        rsi = row['RSI']
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # 3. MACD Analysis
        macd_above_signal = row['MACD'] > row['MACD_Signal']
        macd_hist_increasing = row['MACD_Hist'] > 0
        
        # Calculate technical score (-1 to +1)
        tech_score = 0.0
        
        # Trend score (40% weight)
        trend_score = (0.6 * short_trend + 0.4 * medium_trend) * min(1.0, trend_strength/5.0)
        
        # Momentum score (30% weight)
        momentum_score = 0.0
        if rsi_oversold:
            momentum_score = 1.0  # Strong buy signal
        elif rsi_overbought:
            momentum_score = -1.0  # Strong sell signal
        else:
            momentum_score = (50 - rsi) / 50.0  # Scaled -1 to 1
        
        # MACD score (30% weight)
        macd_score = (0.6 * macd_above_signal + 0.4 * macd_hist_increasing) * 2 - 1
        
        # Combine scores
        tech_score = 0.4 * trend_score + 0.3 * momentum_score + 0.3 * macd_score
        
        # Reward or penalize based on action matching signals
        signal_reward = 0
        if action == 1:  # Buy
            # Strong buy signal: uptrend + not overbought + MACD bullish
            if tech_score > 0.3:  # Requiring strong conviction
                signal_reward = 1.0 * tech_score
            else:
                signal_reward = -0.5
        elif action == 2:  # Sell
            # Strong sell signal: downtrend + not oversold + MACD bearish
            if tech_score < -0.3:  # Requiring strong conviction
                signal_reward = -1.0 * tech_score  # Convert negative score to positive reward
            else:
                signal_reward = -0.5
        
        # Base reward from value change
        reward = value_change - cost + signal_reward
        
        # Invalid action penalty (trying to buy without cash or sell without shares)
        if (action == 1 and self.balance < executed_price * (1 + transaction_fee_percent)) or \
           (action == 2 and self.shares_held == 0):
            reward -= 1.0
        
        # Update for next step
        self.prev_portfolio_value = portfolio_value

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price,
            "portfolio_value": portfolio_value,
            "short_trend": short_trend,
            "medium_trend": medium_trend,
            "trend_strength": trend_strength,
            "rsi": rsi,
            "macd_signal": macd_above_signal,
            "tech_score": tech_score
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
