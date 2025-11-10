# realtime_trader.py
"""
Real-time paper trading with trained model.
Writes streaming CSV logs with headers that the dashboard reads.
No dependency on TradingEnvironment constructor kwargs (works with either signature).
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data_fetcher import DataFetcher
from trading_environment import TradingEnvironment

# optional: load model if available
try:
    from stable_baselines3 import DQN
except Exception:
    DQN = None

# ------------------------------ CSV Schemas -----------------------------------
PORTFOLIO_COLS = [
    "timestamp", "symbol", "price",
    "balance", "shares", "position_value", "net_worth",
    "action"
]

TRADES_COLS = [
    "timestamp", "symbol", "type",
    "price", "shares", "balance", "net_worth", "reason"
]


def _append_csv(path: str, row: Dict, header_cols: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_cols)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header_cols})


class RealTimeTrader:
    """Lightweight realtime trading loop using a trained DQN (if present)."""

    def __init__(self, model_path: str, symbols: List[str], initial_capital: Optional[float] = None):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.initial_capital = float(initial_capital or config.INITIAL_CAPITAL)

        # per-symbol cash allocation
        self.capital_per_symbol = self.initial_capital / max(len(self.symbols), 1)

        # simple portfolio state we maintain ourselves
        self.portfolios = {
            sym: {"balance": self.capital_per_symbol, "shares": 0.0, "entry_price": 0.0}
            for sym in self.symbols
        }

        self.results_dir = config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        self.portfolio_file = os.path.join(self.results_dir, "realtime_portfolio.csv")
        self.trades_file = os.path.join(self.results_dir, "realtime_trades.csv")

        # data fetchers and buffers
        self.fetchers = {sym: DataFetcher(sym) for sym in self.symbols}
        self.buffers = {sym: pd.DataFrame() for sym in self.symbols}

        # load model if present
        self.model = None
        if DQN is not None and os.path.exists(model_path):
            # create a dummy env with any valid dataframe (no special kwargs!)
            df0 = self._fetch_initial(sym=self.symbols[0])
            dummy_env = TradingEnvironment(df0)
            try:
                self.model = DQN.load(model_path, env=dummy_env, device="cpu")
            except Exception:
                self.model = None  # fall back to heuristic

    # --------------------------------- Data -----------------------------------
    def _fetch_initial(self, sym: str) -> pd.DataFrame:
        df = self.fetchers[sym].fetch_historical(period=f"{config.LOOKBACK_DAYS}d", interval="1d")
        if df is None or df.empty:
            raise ValueError(f"No historical data for {sym}")
        return df

    def _update_buffer(self, sym: str, tick: dict):
        if tick is None:
            return
        row = {
            "Date": pd.Timestamp(tick["timestamp"]),
            "Open": float(tick["Open"]),
            "High": float(tick["High"]),
            "Low": float(tick["Low"]),
            "Close": float(tick["Close"]),
            "Volume": float(tick.get("Volume", 0.0)),
        }
        if self.buffers[sym].empty:
            self.buffers[sym] = self._fetch_initial(sym)
        self.buffers[sym] = pd.concat([self.buffers[sym], pd.DataFrame([row])], ignore_index=True).tail(
            max(300, config.LOOKBACK_DAYS)  # keep last window
        )

    # ------------------------------ Policy ------------------------------------
    def _decide(self, sym: str) -> int:
        """
        Return an action: 0 hold, 1 buy, 2 sell.
        Uses model if available; else a tiny momentum heuristic on last return.
        """
        df = self.buffers[sym]
        if df is None or len(df) < 60:
            return 0

        # Build a temp env without passing unexpected kwargs
        env = TradingEnvironment(df)
        obs, _ = env.reset()

        if self.model is not None:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                return int(action)
            except Exception:
                pass

        # Heuristic fallback: buy if last return strongly positive, sell on negative
        rets = df["Close"].pct_change().fillna(0.0).to_numpy()
        last_ret = float(rets[-1]) if len(rets) else 0.0
        if last_ret > 0.003:
            return 1
        if last_ret < -0.003 and self.portfolios[sym]["shares"] > 0:
            return 2
        return 0

    # ------------------------------ Logging -----------------------------------
    def _log_portfolio(self, sym: str, price: float, action: str):
        p = self.portfolios[sym]
        row = {
            "timestamp": datetime.now(),
            "symbol": sym,
            "price": price,
            "balance": p["balance"],
            "shares": p["shares"],
            "position_value": p["shares"] * price,
            "net_worth": p["balance"] + p["shares"] * price,
            "action": action,
        }
        _append_csv(self.portfolio_file, row, PORTFOLIO_COLS)

    def _log_trade(self, sym: str, typ: str, price: float, shares: float, reason: str = ""):
        p = self.portfolios[sym]
        row = {
            "timestamp": datetime.now(),
            "symbol": sym,
            "type": typ,
            "price": price,
            "shares": shares,
            "balance": p["balance"],
            "net_worth": p["balance"] + p["shares"] * price,
            "reason": reason,
        }
        _append_csv(self.trades_file, row, TRADES_COLS)

    # ------------------------------ Execution ---------------------------------
    def _execute(self, sym: str, action: int, price: float):
        p = self.portfolios[sym]
        fees = config.COMMISSION + config.SLIPPAGE

        if action == 1:  # BUY ~30% of cash
            cash_to_use = p["balance"] * 0.30
            total_cost = cash_to_use
            if total_cost > price * (1 + fees):
                shares = cash_to_use / (price * (1 + fees))
                p["balance"] -= shares * price * (1 + fees)
                # update entry price (weighted)
                if p["shares"] > 0:
                    total_shares_value = p["shares"] * p["entry_price"] + shares * price
                    p["shares"] += shares
                    p["entry_price"] = total_shares_value / p["shares"]
                else:
                    p["shares"] = shares
                    p["entry_price"] = price
                self._log_trade(sym, "BUY", price, shares, "policy")
                self._log_portfolio(sym, price, "BUY")
                return

        elif action == 2 and p["shares"] > 0:  # SELL all
            proceeds = p["shares"] * price * (1 - fees)
            self._log_trade(sym, "SELL", price, p["shares"], "policy")
            p["balance"] += proceeds
            p["shares"] = 0.0
            p["entry_price"] = 0.0
            self._log_portfolio(sym, price, "SELL")
            return

        # Hold (or invalid buy)
        self._log_portfolio(sym, price, "HOLD")

    # --------------------------------- Loop -----------------------------------
    def run(self, duration_minutes: int = config.DEFAULT_DURATION):
        end_time = time.time() + duration_minutes * 60

        # seed buffers
        for sym in self.symbols:
            self.buffers[sym] = self._fetch_initial(sym)

        while time.time() < end_time:
            for sym in self.symbols:
                tick = self.fetchers[sym].fetch_realtime(interval="1m")
                if tick is not None and float(tick["Close"]) > 0:
                    self._update_buffer(sym, tick)
                    price = float(tick["Close"])
                else:
                    # fallback to last known close
                    price = float(self.buffers[sym]["Close"].iloc[-1])

                action = self._decide(sym)
                self._execute(sym, action, price)

            time.sleep(max(1, int(config.UPDATE_INTERVAL)))
