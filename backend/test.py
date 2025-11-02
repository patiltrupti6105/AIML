import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from environment import StockTradingEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_metrics(equity):
    returns = np.diff(equity) / (equity[:-1] + 1e-9)
    cum_return = equity[-1] / equity[0] - 1
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-9)
    max_dd = drawdown.min()
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
    return {
        "cum_return": float(cum_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
    }


def test(data_file: str, model_file: str, results_file: str = None):
    """
    Load trained DQN model and run full simulation.
    Saves results and plots to backend/model/ only.
    """
    print(f"📂 Loading data from {data_file} ...")
    df = pd.read_csv(data_file)
    env = StockTradingEnv(df)

    # --- Model Path Handling (no new folders) ---
    model_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"🤖 Loading model from {model_path} ...")
    model = DQN.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    log_data = []
    step = 0
    equity = [env.get_portfolio_value()]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        price = info.get("price", env._get_price())
        balance = info.get("balance", env.balance)
        shares_held = info.get("shares_held", env.shares_held)
        portfolio_value = info.get("portfolio_value", balance + shares_held * price)

        log_data.append({
            "step": step,
            "price": price,
            "balance": balance,
            "shares_held": shares_held,
            "portfolio": portfolio_value,
            "action": int(action),
            "reward": reward
        })

        equity.append(portfolio_value)
        step += 1
        if step > len(env.df) + 5:
            print("⚠️ Early stop (safety limit reached)")
            break

    # --- Save Results ---
    results_df = pd.DataFrame(log_data)
    if results_file is None:
        results_file = os.path.join(MODEL_DIR, "results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✅ Results saved to {results_file} ({len(results_df)} steps)")

    # --- Compute Metrics ---
    metrics = compute_metrics(np.array(equity))
    print(f"📊 Metrics: {metrics}")

    # --- Save Plots (in model folder only) ---
    plt.figure(figsize=(10, 4))
    plt.plot(results_df["price"].values, label="Price")
    buy_idx = results_df.loc[results_df["action"] == 1].index
    sell_idx = results_df.loc[results_df["action"] == 2].index
    plt.scatter(buy_idx, results_df.loc[buy_idx, "price"], marker="^", color="green", label="Buy")
    plt.scatter(sell_idx, results_df.loc[sell_idx, "price"], marker="v", color="red", label="Sell")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "price_actions.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(equity, label="Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "equity_curve.png"))
    plt.close()

    print("📈 Plots saved to backend/model/")
    return results_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=os.path.join(DATA_DIR, "data_AAPL.csv"),
                        help="Path to input CSV file (price history)")
    parser.add_argument("--model", type=str, default="dqn_trading_agent.zip",
                        help="Model filename in backend/model/")
    parser.add_argument("--out", type=str, default=os.path.join(MODEL_DIR, "results.csv"),
                        help="Output CSV for simulation results")
    args = parser.parse_args()

    test(args.csv, args.model, args.out)
