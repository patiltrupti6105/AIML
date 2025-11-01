# frontend/app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(ROOT, "backend", "model", "results.csv")

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="🤖 Self-Learning Trading Bot", layout="wide")
st.title("🤖 Self-Learning Stock Trading Bot — Demo")

# ---------- Load Results ----------
if not os.path.exists(RESULTS_CSV):
    st.warning("⚠️ Results not found. Run backend/test.py to generate results.csv first.")
    st.write("Expected results file:", RESULTS_CSV)
else:
    df = pd.read_csv(RESULTS_CSV)

    # Validate and compute portfolio value properly
    if "portfolio" not in df.columns:
        if all(col in df.columns for col in ["balance", "shares", "price"]):
            df["portfolio"] = df["balance"] + df["shares"] * df["price"]
        else:
            st.error("❌ Missing portfolio data in CSV.")
            st.stop()

    # ---------- Basic Metrics ----------
    st.subheader("📊 Basic Metrics")
    st.write(f"Data points: {len(df)}")

    total_return = (df["portfolio"].iloc[-1] - df["portfolio"].iloc[0]) / max(df["portfolio"].iloc[0], 1)
    st.metric("Total Return (approx)", f"{total_return:.2%}")

    # ---------- Chart 1: Stock Price with Buy/Sell Markers ----------
    st.subheader("📈 Stock Price with Buy/Sell Markers")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["price"], label="Stock Price", color="blue", linewidth=1.5)

    buy_indices = df.index[df["action"] == 1].tolist()
    sell_indices = df.index[df["action"] == 2].tolist()

    ax.scatter(buy_indices, df.loc[buy_indices, "price"], marker="^", color="green", label="Buy", zorder=3)
    ax.scatter(sell_indices, df.loc[sell_indices, "price"], marker="v", color="red", label="Sell", zorder=3)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ---------- Chart 2: Portfolio Value vs Stock Price ----------
    st.subheader("💰 Portfolio Performance vs Stock Price")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df["portfolio"], label="Portfolio Value", color="orange", linewidth=1.8)
    ax2.plot(df["price"] / df["price"].iloc[0] * df["portfolio"].iloc[0],
             label="Stock Price (normalized)", color="gray", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Value ($)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    # ---------- Chart 3: Action Distribution ----------
    st.subheader("⚙️ Action Distribution")
    action_counts = df["action"].value_counts().sort_index()
    st.bar_chart(action_counts.rename({0: "Hold", 1: "Buy", 2: "Sell"}))

    # ---------- Raw Results ----------
    st.subheader("📄 Raw Results (first 200 rows)")
    st.dataframe(df.head(200))
