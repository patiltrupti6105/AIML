# trading_environment.py 
import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

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
# Trade tracking 
# -----------------------------------------------------------------------------
@dataclass
class Trade:
    """Represents a single trade/position lot with trailing stop support"""
    entry_price: float
    shares: int
    entry_step: int
    highest_price: float = None  # NEW: Track peak price for trailing stop
    trailing_stop_price: float = None  # NEW: Current trailing stop level
    
    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.shares
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price
    
    def update_trailing_stop(self, bar_high: float, trailing_pct: float, initial_stop_pct: float):
        """Update trailing stop based on new high price"""
        # Initialize highest price on first call
        if self.highest_price is None:
            self.highest_price = self.entry_price
        
        # Update highest price if new high reached
        if bar_high > self.highest_price:
            self.highest_price = bar_high
        
        # Calculate trailing stop (% below highest price)
        trailing_stop = self.highest_price * (1 - trailing_pct)
        
        # Calculate initial stop (% below entry)
        initial_stop = self.entry_price * (1 - initial_stop_pct)
        
        # Use the HIGHER of: initial stop OR trailing stop
        # (this ensures stop only moves UP, never down)
        self.trailing_stop_price = max(initial_stop, trailing_stop)
        
        return self.trailing_stop_price


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
# Config dataclass - MODIFIED
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
    bid_ask_spread_pct: float = 0.001  # 0.1% spread (10 bps)
    trailing_stop_pct: float = 0.05  #  5% trailing stop from peak
    use_trailing_stop: bool = True   #  Enable/disable trailing stop


# -----------------------------------------------------------------------------
# Trading Environment - WITH TRAILING STOP
# -----------------------------------------------------------------------------
class TradingEnvironment(gym.Env):
    """
    REALISTIC TRADING SIMULATION WITH TRAILING STOP:
    - Uses NEXT bar's Open price for fills (not current Close)
    - Tracks individual position lots with separate entry prices
    - Models bid-ask spread
    - Stop loss checks happen at Low, take profit at High
    - TRAILING STOP: Locks in profits as price rises
    - Proper realized vs unrealized P&L
    
    Actions:
        0 = Hold
        1 = Buy (risk 2% of capital on this trade)
        2 = Sell 50% of all positions (FIFO)
        3 = Sell 100% (close all positions)
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

        # Bollinger position
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
        self.action_space = Discrete(4)
        obs_len = self.cfg.lookback_window + 14
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
        info: Dict = {
            "trades_closed": [],
            "realized_pnl": 0.0,
        }

        current_bar = self.df.iloc[self.current_step]
        current_close = float(current_bar["Close"])
        
        # Check if we can advance (need next bar for execution)
        if self.current_step >= len(self.df) - 2:
            done = True
            obs = self._get_obs()
            self._mark_to_market(current_close)
            info.update(self._build_info(current_close))
            return obs, 0.0, done, False, info

        # ADVANCE TO NEXT BAR for execution
        self.current_step += 1
        next_bar = self.df.iloc[self.current_step]
        
        execution_price = float(next_bar["Open"])
        bar_high = float(next_bar["High"])
        bar_low = float(next_bar["Low"])
        bar_close = float(next_bar["Close"])

        start_nav = self.net_worth

        # Execute action at Open
        if action == 1:
            self._buy_risk_based(execution_price, info)
        elif action == 2:
            self._sell_pct_fifo(execution_price, 0.50, info)
        elif action == 3:
            self._close_all_positions(execution_price, info)

        # MODIFIED: Check risk management with trailing stop
        self._check_risk_management(bar_low, bar_high, info)

        # Mark all positions to market at bar close
        self._mark_to_market(bar_close)

        # Calculate reward based on step P&L
        step_pnl = self.net_worth - start_nav
        reward = (step_pnl / (start_nav + 1e-9)) * 100
        
        # Small penalty for trading
        if action != 0:
            reward -= 0.02

        # Check if episode done
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_obs()
        info.update(self._build_info(bar_close))
        info["action"] = int(action)
        
        return obs, float(reward), done, False, info

    # -------------------------------------------------------------------------
    # Internal State
    # -------------------------------------------------------------------------
    def reset_state(self):
        self.current_step = max(self.cfg.lookback_window - 1, 0)
        self.balance = float(self.cfg.initial_capital)
        self.positions: List[Trade] = []
        self.net_worth = self.balance
        self.total_realized_pnl = 0.0
        self.total_commission_paid = 0.0

    def _total_shares(self) -> int:
        return sum(p.shares for p in self.positions)

    def _total_position_value(self, price: float) -> float:
        return sum(p.shares * price for p in self.positions)
    
    def _unrealized_pnl(self, price: float) -> float:
        return sum(p.unrealized_pnl(price) for p in self.positions)

    def _mark_to_market(self, price: float):
        """Update net worth based on current market price"""
        position_value = self._total_position_value(price)
        self.net_worth = self.balance + position_value

    def _avg_entry_price(self) -> float:
        """For observation normalization"""
        if not self.positions:
            return 0.0
        total_cost = sum(p.entry_price * p.shares for p in self.positions)
        total_shares = self._total_shares()
        return total_cost / total_shares if total_shares > 0 else 0.0

    # -------------------------------------------------------------------------
    # Trading Logic
    # -------------------------------------------------------------------------
    def _get_fill_price(self, market_price: float, is_buy: bool) -> float:
        """Model bid-ask spread"""
        spread = market_price * self.cfg.bid_ask_spread_pct
        if is_buy:
            return market_price + spread / 2
        else:
            return market_price - spread / 2

    def _calculate_commission(self, shares: int, price: float) -> float:
        """Calculate commission on trade"""
        trade_value = shares * price
        return trade_value * self.cfg.commission

    def _buy_risk_based(self, market_price: float, info: Dict):
        """Risk-based position sizing"""
        if self.cfg.stop_loss_pct <= 0:
            risk_amount = self.balance * 0.25
        else:
            risk_amount = self.net_worth * 0.02
            
        stop_distance_pct = abs(self.cfg.stop_loss_pct)
        max_position_value = risk_amount / stop_distance_pct
        
        fill_price = self._get_fill_price(market_price, is_buy=True)
        fill_price = fill_price * (1 + self.cfg.slippage)
        
        shares_to_buy = int(max_position_value / fill_price)
        
        if shares_to_buy <= 0:
            info["buy_executed"] = False
            info["buy_reason"] = "insufficient_capital"
            return
        
        total_cost = shares_to_buy * fill_price
        commission = self._calculate_commission(shares_to_buy, fill_price)
        total_cost_with_commission = total_cost + commission
        
        new_total_value = self._total_position_value(market_price) + total_cost
        max_allowed = self.cfg.max_position_pct * self.cfg.initial_capital
        
        if total_cost_with_commission > self.balance:
            info["buy_executed"] = False
            info["buy_reason"] = "insufficient_balance"
            return
            
        if new_total_value > max_allowed:
            info["buy_executed"] = False
            info["buy_reason"] = "position_limit_exceeded"
            return
        
        # Execute trade
        self.balance -= total_cost_with_commission
        self.total_commission_paid += commission
        
        # Create new position lot (trailing stop initialized in update)
        new_trade = Trade(
            entry_price=fill_price,
            shares=shares_to_buy,
            entry_step=self.current_step
        )
        self.positions.append(new_trade)
        
        info["buy_executed"] = True
        info["shares_bought"] = shares_to_buy
        info["fill_price"] = fill_price
        info["commission"] = commission
        info["total_cost"] = total_cost_with_commission

    def _sell_pct_fifo(self, market_price: float, pct: float, info: Dict):
        """Sell percentage of position using FIFO"""
        if not self.positions:
            info["sell_executed"] = False
            return
        
        total_shares = self._total_shares()
        shares_to_sell = int(total_shares * pct)
        
        if shares_to_sell <= 0:
            info["sell_executed"] = False
            return
        
        fill_price = self._get_fill_price(market_price, is_buy=False)
        fill_price = fill_price * (1 - self.cfg.slippage)
        
        self._execute_sell(shares_to_sell, fill_price, info)

    def _close_all_positions(self, market_price: float, info: Dict):
        """Sell all positions"""
        if not self.positions:
            info["sell_executed"] = False
            return
        
        total_shares = self._total_shares()
        fill_price = self._get_fill_price(market_price, is_buy=False)
        fill_price = fill_price * (1 - self.cfg.slippage)
        
        self._execute_sell(total_shares, fill_price, info)

    def _execute_sell(self, shares_to_sell: int, fill_price: float, info: Dict):
        """Execute sale using FIFO accounting"""
        remaining = shares_to_sell
        total_proceeds = 0.0
        total_cost_basis = 0.0
        trades_closed = []
        
        while remaining > 0 and self.positions:
            position = self.positions[0]
            
            if position.shares <= remaining:
                # Close entire position
                proceeds = position.shares * fill_price
                cost_basis = position.shares * position.entry_price
                commission = self._calculate_commission(position.shares, fill_price)
                
                total_proceeds += proceeds - commission
                total_cost_basis += cost_basis
                
                trades_closed.append({
                    "shares": position.shares,
                    "entry_price": position.entry_price,
                    "exit_price": fill_price,
                    "pnl": proceeds - commission - cost_basis,
                    "hold_time": self.current_step - position.entry_step
                })
                
                remaining -= position.shares
                self.positions.pop(0)
            else:
                # Partial close
                proceeds = remaining * fill_price
                cost_basis = remaining * position.entry_price
                commission = self._calculate_commission(remaining, fill_price)
                
                total_proceeds += proceeds - commission
                total_cost_basis += cost_basis
                
                trades_closed.append({
                    "shares": remaining,
                    "entry_price": position.entry_price,
                    "exit_price": fill_price,
                    "pnl": proceeds - commission - cost_basis,
                    "hold_time": self.current_step - position.entry_step
                })
                
                position.shares -= remaining
                remaining = 0
        
        self.balance += total_proceeds
        self.total_commission_paid += sum(self._calculate_commission(t["shares"], t["exit_price"]) for t in trades_closed)
        
        realized_pnl = total_proceeds - total_cost_basis
        self.total_realized_pnl += realized_pnl
        
        info["sell_executed"] = True
        info["shares_sold"] = shares_to_sell
        info["fill_price"] = fill_price
        info["realized_pnl"] = realized_pnl
        info["trades_closed"] = trades_closed

    # -------------------------------------------------------------------------
    # Risk Management 
    # -------------------------------------------------------------------------
    def _check_risk_management(self, bar_low: float, bar_high: float, info: Dict):
        """
        Check stops at bar extremes with TRAILING STOP support
        - Updates trailing stop based on bar_high
        - Checks if bar_low triggered the stop
        - Also checks fixed take profit at bar_high
        """
        if not self.positions:
            return
        
        positions_to_close = []
        
        for position in self.positions:
            # UPDATE TRAILING STOP (moves up as price rises)
            if self.cfg.use_trailing_stop:
                trailing_stop_price = position.update_trailing_stop(
                    bar_high=bar_high,
                    trailing_pct=self.cfg.trailing_stop_pct,
                    initial_stop_pct=self.cfg.stop_loss_pct
                )
                
                # Check if trailing stop was hit at bar's LOW
                if bar_low <= trailing_stop_price:
                    positions_to_close.append(("trailing_stop", position, trailing_stop_price))
                    continue  # Skip other checks for this position
            else:
                # Original fixed stop loss logic
                pnl_at_low = position.unrealized_pnl_pct(bar_low)
                stop_loss = -abs(self.cfg.stop_loss_pct)
                
                if pnl_at_low <= stop_loss:
                    positions_to_close.append(("stop_loss", position, bar_low))
                    continue
            
            # Check take profit at HIGH (unchanged)
            pnl_at_high = position.unrealized_pnl_pct(bar_high)
            take_profit = abs(self.cfg.take_profit_pct)
            
            if pnl_at_high >= take_profit:
                positions_to_close.append(("take_profit", position, bar_high))
        
        # Execute risk exits
        for exit_type, position, exit_price in positions_to_close:
            fill_price = self._get_fill_price(exit_price, is_buy=False)
            
            proceeds = position.shares * fill_price
            cost_basis = position.shares * position.entry_price
            commission = self._calculate_commission(position.shares, fill_price)
            
            self.balance += proceeds - commission
            self.total_commission_paid += commission
            
            realized_pnl = proceeds - commission - cost_basis
            self.total_realized_pnl += realized_pnl
            
            # NEW: Include trailing stop info
            exit_info = {
                "type": exit_type,
                "shares": position.shares,
                "entry_price": position.entry_price,
                "exit_price": fill_price,
                "pnl": realized_pnl
            }
            
            # Add trailing stop details if applicable
            if exit_type == "trailing_stop":
                exit_info["highest_price"] = position.highest_price
                exit_info["trailing_stop_price"] = position.trailing_stop_price
                exit_info["profit_from_entry_pct"] = ((fill_price - position.entry_price) / position.entry_price) * 100
            
            info.setdefault("risk_exits", []).append(exit_info)
            
            self.positions.remove(position)

    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        start = max(0, self.current_step - self.cfg.lookback_window + 1)
        window = self.df.iloc[start : self.current_step + 1]
        rets = window["ret"].to_numpy(dtype=np.float32)
        if len(rets) < self.cfg.lookback_window:
            pad = np.zeros(self.cfg.lookback_window - len(rets), dtype=np.float32)
            rets = np.concatenate([pad, rets], axis=0)

        row = self.df.iloc[self.current_step]
        price = float(row["Close"])

        ma_start = max(0, self.current_step - 49)
        ma50 = float(self.df["Close"].iloc[ma_start : self.current_step + 1].mean())
        if not np.isfinite(ma50) or ma50 <= 0:
            ma50 = price
            
        price_norm = np.clip((price / ma50 - 1.0) * 10, -3, 3)
        balance_norm = np.clip(self.balance / self.cfg.initial_capital - 1.0, -1, 2)
        
        total_shares = self._total_shares()
        shares_norm = np.clip(total_shares / 100.0, 0, 5)
        position_value = self._total_position_value(price)
        position_pct = position_value / (self.cfg.initial_capital + 1e-9)
        position_value_norm = np.clip(position_pct, 0, 2)
        
        unrealized_pnl = self._unrealized_pnl(price)
        unrealized_pnl_norm = np.clip(unrealized_pnl / (self.cfg.initial_capital + 1e-9) * 10, -3, 3)

        rsi = float(row.get("rsi", 50.0)) / 100.0
        sma10_z = float(row.get("sma10_z", 0.0))
        sma20_z = float(row.get("sma20_z", 0.0))
        ema12_z = float(row.get("ema12_z", 0.0))
        ema26_z = float(row.get("ema26_z", 0.0))
        macd = np.clip(float(row.get("macd", 0.0)) / 10.0, -3, 3)
        macd_signal = np.clip(float(row.get("macd_signal", 0.0)) / 10.0, -3, 3)
        bb_pos = float(row.get("bb_pos", 0.0))
        atr_z = float(row.get("atr_z", 0.0))

        scalars = np.array(
            [
                balance_norm,
                shares_norm,
                price_norm,
                position_value_norm,
                unrealized_pnl_norm,
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

    def _build_info(self, price: float) -> Dict:
        """Build info dict with current state"""
        return {
            "step": self.current_step,
            "price": price,
            "balance": self.balance,
            "total_shares": self._total_shares(),
            "num_positions": len(self.positions),
            "position_value": self._total_position_value(price),
            "unrealized_pnl": self._unrealized_pnl(price),
            "net_worth": self.net_worth,
            "total_realized_pnl": self.total_realized_pnl,
            "total_commission_paid": self.total_commission_paid,
            "symbol": self.symbol,
        }
    
    @property
    def shares_held(self) -> int:
        """Backward-compatible property"""
        return self._total_shares()
    
    def _cost_multiplier(self) -> float:
        """Backward-compatible cost multiplier"""
        return self.cfg.commission + self.cfg.slippage

    # -------------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------------
    def render(self):
        if self.current_step >= len(self.df):
            return
        price = float(self.df.iloc[self.current_step]["Close"])
        return_pct = ((self.net_worth / self.cfg.initial_capital) - 1.0) * 100
        unrealized = self._unrealized_pnl(price)
        
        # NEW: Show trailing stop info
        trailing_info = ""
        if self.positions and self.cfg.use_trailing_stop:
            avg_stop = sum(p.trailing_stop_price or 0 for p in self.positions) / len(self.positions)
            trailing_info = f" avg_trail_stop={avg_stop:.2f}"
        
        print(
            f"[{self.symbol}] step={self.current_step} price={price:.2f} "
            f"bal={self.balance:.2f} positions={len(self.positions)} "
            f"shares={self._total_shares()} unrealized_pnl={unrealized:.2f} "
            f"nav={self.net_worth:.2f} return={return_pct:.2f}%{trailing_info}"
        )