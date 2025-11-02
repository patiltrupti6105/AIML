# frontend/app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(ROOT, "backend", "model", "results.csv")

st.set_page_config(page_title="Self-Learning Trading Bot", layout="wide")
st.title("🤖 Self-Learning Stock Trading Bot — Demo")

def compute_metrics(df):
    equity = df["portfolio"].values
    returns = np.diff(equity) / (equity[:-1] + 1e-9)
    cum_return = equity[-1] / equity[0] - 1
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-9)
    max_dd = drawdown.min()
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9) if returns.std() > 0 else 0.0
    return {"cum_return": cum_return, "max_drawdown": max_dd, "sharpe": sharpe}

if not os.path.exists(RESULTS_CSV):
    st.warning("Results not found. Run backend/test.py to generate results.csv first.")
else:
    df = pd.read_csv(RESULTS_CSV)

    st.subheader("📊 Basic Metrics")
    st.write(f"Data points: {len(df)}")
    total_return = (df["portfolio"].iloc[-1] - df["portfolio"].iloc[0]) / max(df["portfolio"].iloc[0], 1)
    st.metric("Total Return (approx)", f"{total_return:.2%}")

    metrics = compute_metrics(df)
    st.metric("Sharpe (approx)", f"{metrics['sharpe']:.2f}")
    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    # ----- Stock Price with Buy/Sell -----
    st.subheader("📈 Stock Price with Buy/Sell Markers")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["price"], label="Stock Price", linewidth=1.2)
    ax.scatter(df.index[df["action"] == 1], df.loc[df["action"] == 1, "price"],
               marker="^", color="green", label="Buy", zorder=3)
    ax.scatter(df.index[df["action"] == 2], df.loc[df["action"] == 2, "price"],
               marker="v", color="red", label="Sell", zorder=3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

    # ----- Portfolio, Balance, Holdings -----
    st.subheader("💰 Portfolio Breakdown Over Time")
    if {"balance", "shares_held"}.issubset(df.columns):
        df["holdings_value"] = df["shares_held"] * df["price"]

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df["portfolio"], label="Portfolio Value", color="orange", linewidth=1.3)
        ax2.plot(df["balance"], label="Cash Balance", color="green", linestyle="--", alpha=0.8)
        ax2.plot(df["holdings_value"], label="Stock Holdings Value", color="blue", linestyle="--", alpha=0.8)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Value ($)")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Balance/holdings not found in results.csv (update test.py to log them).")

   
EVAL_FILE = os.path.join(ROOT, "backend", "model", "evaluations.npz")

if os.path.exists(EVAL_FILE):
    st.subheader("📈 Training Evaluation Performance")
    data = np.load(EVAL_FILE)
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)

    fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
    ax_eval.plot(timesteps, mean_rewards, color="purple", linewidth=2)
    ax_eval.set_title("Evaluation Mean Reward Over Time")
    ax_eval.set_xlabel("Timesteps")
    ax_eval.set_ylabel("Mean Reward")
    st.pyplot(fig_eval)
else:
    st.info("No evaluation file found (evaluations.npz). Train model with EvalCallback to generate it.")