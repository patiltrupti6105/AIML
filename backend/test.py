# backend/test.py
import os
import argparse
import pandas as pd
from stable_baselines3 import DQN
from environment import StockTradingEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODEL_DIR, exist_ok=True)


def test(data_file: str, model_file: str, results_file: str = None):
    """
    Load trained model and run a full simulation.
    Saves (price, balance, shares_held, portfolio_value, action) to results.csv
    """
    print(f"Loading data from {data_file} ...")
    df = pd.read_csv(data_file)
    env = StockTradingEnv(df)

    model_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path} ...")
    model = DQN.load(model_path, env=env)

    obs, _ = env.reset()
    done = False

    log_data = []
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        # Get info dict safely
        price = info.get("price", float(env.df.loc[env.current_step, "Close"]))
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

        step += 1
        if step > len(env.df) + 5:
            break  # safety limit

    results_df = pd.DataFrame(log_data)

    if results_file is None:
        results_file = os.path.join(MODEL_DIR, "results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✅ Results saved to {results_file} (total {len(results_df)} steps)")
    return results_df


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
