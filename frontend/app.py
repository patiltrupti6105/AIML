# frontend/app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(ROOT, "backend", "model", "results.csv")

st.set_page_config(page_title="Self-Learning Trading Bot", layout="wide")
st.title("🤖 Self-Learning Stock Trading Bot — Demo")

if not os.path.exists(RESULTS_CSV):
    st.warning("Results not found. Run backend/test.py to generate results.csv first.")
else:
    df = pd.read_csv(RESULTS_CSV)

    st.subheader("📊 Basic Metrics")
    st.write(f"Data points: {len(df)}")
    total_return = (df["portfolio"].iloc[-1] - df["portfolio"].iloc[0]) / max(df["portfolio"].iloc[0], 1)
    st.metric("Total Return (approx)", f"{total_return:.2%}")

    # ----- Stock Price with Buy/Sell -----
    st.subheader("📈 Stock Price with Buy/Sell Markers")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["price"], label="Stock Price", color="blue", linewidth=1.5)
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
    if "balance" in df.columns and "shares_held" in df.columns:
        df["holdings_value"] = df["shares_held"] * df["price"]

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df["portfolio"], label="Portfolio Value", color="orange", linewidth=1.5)
        ax2.plot(df["balance"], label="Cash Balance", color="green", linestyle="--", alpha=0.7)
        ax2.plot(df["holdings_value"], label="Stock Holdings Value", color="blue", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Value ($)")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Balance/holdings not found in results.csv (update test.py to log them).")

    # ----- Action Distribution -----
    st.subheader("⚙️ Action Distribution")
    action_counts = df["action"].value_counts().sort_index()
    st.bar_chart(pd.DataFrame({
        "Buy": [action_counts.get(1, 0)],
        "Hold": [action_counts.get(0, 0)],
        "Sell": [action_counts.get(2, 0)],
    }).T)

    st.subheader("🧾 Raw Results (first 200 rows)")
    st.dataframe(df.head(200))
