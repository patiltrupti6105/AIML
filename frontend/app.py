# frontend/app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(ROOT, "backend", "model", "results.csv")

st.write("Resolved absolute path:", RESULTS_CSV)


st.set_page_config(page_title="Self-Learning Trading Bot", layout="wide")
st.title("🤖 Self-Learning Stock Trading Bot — Demo")

if not os.path.exists(RESULTS_CSV):
    st.warning("Results not found. Run backend/test.py to generate results.csv first.")
    st.write("Expected results file:", RESULTS_CSV)
else:
    df = pd.read_csv(RESULTS_CSV)
    st.subheader("Basic metrics")
    st.write(f"Data points: {len(df)}")
    total_return = (df['portfolio'].iloc[-1] - df['portfolio'].iloc[0]) / max(df['portfolio'].iloc[0], 1)
    st.metric("Total Return (approx)", f"{total_return:.2%}")

    # Price chart with Buy/Sell markers
    st.subheader("Stock Price with Buy/Sell Markers")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['price'], label='Price', linewidth=1.5)
    buy_indices = df.index[df['action'] == 1].tolist()
    sell_indices = df.index[df['action'] == 2].tolist()
    ax.scatter(buy_indices, df.loc[buy_indices, 'price'], marker='^', color='green', label='Buy', zorder=3)
    ax.scatter(sell_indices, df.loc[sell_indices, 'price'], marker='v', color='red', label='Sell', zorder=3)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Portfolio Value Over Time")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df['portfolio'], label='Portfolio Value', linewidth=1.5)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Portfolio ($)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Raw results (first 200 rows)")
    st.dataframe(df.head(200))
