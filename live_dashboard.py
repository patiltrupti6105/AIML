# live_dashboard.py
# Streamlit live dashboard for the trading bot (safe against empty/missing CSVs)

from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

import config

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


# ----------------------------- Safe CSV Reader --------------------------------
def safe_read_csv(path: str, columns: list[str]) -> pd.DataFrame:
    """
    Read a CSV defensively:
      • If file is missing/empty → empty DF with expected columns.
      • If parsing fails → empty DF with expected columns.
      • Ensures expected columns exist (adds missing as NA).
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame(columns=columns)

        df = pd.read_csv(path)
        if df is None or df.empty or len(df.columns) == 0:
            return pd.DataFrame(columns=columns)

        for c in columns:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[columns]

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=columns)


# ------------------------------- Dashboard ------------------------------------
class LiveDashboard:
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        self.portfolio_file = os.path.join(self.results_dir, "realtime_portfolio.csv")
        self.trades_file = os.path.join(self.results_dir, "realtime_trades.csv")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        portfolio_df = safe_read_csv(self.portfolio_file, PORTFOLIO_COLS)
        trades_df = safe_read_csv(self.trades_file, TRADES_COLS)
        return portfolio_df, trades_df

    @staticmethod
    def get_current_status(portfolio_df: pd.DataFrame) -> dict:
        if portfolio_df.empty:
            return {
                "last_time": "-",
                "symbol": "-",
                "price": 0.0,
                "balance": config.INITIAL_CAPITAL,
                "shares": 0,
                "pos_val": 0.0,
                "net_worth": config.INITIAL_CAPITAL,
                "action": "N/A",
            }
        last = portfolio_df.iloc[-1]
        return {
            "last_time": last["timestamp"],
            "symbol": last.get("symbol", "-"),
            "price": float(last.get("price", 0.0) or 0.0),
            "balance": float(last.get("balance", 0.0) or 0.0),
            "shares": int(last.get("shares", 0) or 0),
            "pos_val": float(last.get("position_value", 0.0) or 0.0),
            "net_worth": float(last.get("net_worth", 0.0) or 0.0),
            "action": str(last.get("action", "N/A")),
        }

    # --------------------------- Rendering helpers ----------------------------
    def render_header(self, status: dict):
        st.title("📈 Real-Time RL Trading Bot — Live Dashboard")
        st.caption("Robust dashboard: handles empty/missing files without crashing.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbol", status["symbol"])
        col2.metric("Last Price", f"{status['price']:.2f}")
        col3.metric("Net Worth", f"{status['net_worth']:.2f}")
        col4.metric("Shares", f"{status['shares']}")
        st.write(f"**Last Update:** {status['last_time']} — **Last Action:** {status['action']}")

    def render_portfolio_chart(self, portfolio_df: pd.DataFrame):
        st.subheader("Net Worth Over Time")
        if portfolio_df.empty:
            st.info("Waiting for trading data…")
            return
        chart_df = portfolio_df[["timestamp", "net_worth"]].rename(columns={"timestamp": "time"})
        st.line_chart(chart_df, x="time", y="net_worth")

    def render_price_chart(self, portfolio_df: pd.DataFrame):
        st.subheader("Price Over Time")
        if portfolio_df.empty:
            st.info("Waiting for price data…")
            return
        chart_df = portfolio_df[["timestamp", "price"]].rename(columns={"timestamp": "time"})
        st.line_chart(chart_df, x="time", y="price")

    def render_tables(self, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame):
        st.subheader("Recent Portfolio Rows")
        st.dataframe(portfolio_df.tail(50), use_container_width=True)
        st.subheader("Recent Trades")
        st.dataframe(trades_df.tail(50), use_container_width=True)


# --------------------------------- App ----------------------------------------
def main():
    st.set_page_config(page_title="Live Trading Dashboard", layout="wide")

    # Sidebar
    st.sidebar.header("Settings")
    refresh_sec = st.sidebar.slider("Auto-refresh (seconds)", 2, 20, 5)
    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared.")

    dashboard = LiveDashboard()

    # Load
    portfolio_df, trades_df = dashboard.load_data()
    status = dashboard.get_current_status(portfolio_df)

    # Render
    dashboard.render_header(status)
    left, right = st.columns(2)
    with left:
        dashboard.render_portfolio_chart(portfolio_df)
    with right:
        dashboard.render_price_chart(portfolio_df)
    dashboard.render_tables(portfolio_df, trades_df)

    st.sidebar.write(f"Data dir: `{config.RESULTS_DIR}`")
    st.sidebar.write("Tip: start trading from the launcher to populate data.")

    # simple time-based refresh (no deprecated experimental APIs)
    time.sleep(refresh_sec)
    st.rerun()


if __name__ == "__main__":
    main()
