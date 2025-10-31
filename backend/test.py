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
    Load model and run simulation on data_file. Save results (price, portfolio, action) to CSV.
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
    terminated = truncated = False
    done = False

    prices, portfolio_values, actions, balances, shares = [], [], [], [], []
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        # Save info (we append value AFTER the step moved ahead)
        # If env moved to next step, current_step points to the new step's index
        # Use info where possible or recompute
        cur_price = float(env.df.loc[env.current_step, 'Close'])
        portfolio_val = env.balance + env.shares_held * cur_price

        prices.append(cur_price)
        portfolio_values.append(portfolio_val)
        actions.append(int(action))
        balances.append(env.balance)
        shares.append(env.shares_held)

        step_idx += 1
        if step_idx > len(env.df) + 5:
            break  # safety

    results_df = pd.DataFrame({
        "price": prices,
        "portfolio": portfolio_values,
        "action": actions,
        "balance": balances,
        "shares": shares
    })

    if results_file is None:
        results_file = os.path.join(MODEL_DIR, "results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=os.path.join(DATA_DIR, "data_AAPL.csv"),
                        help="Path to CSV file")
    parser.add_argument("--model", type=str, default="dqn_trading_agent.zip",
                        help="Model filename in backend/model/")
    parser.add_argument("--out", type=str, default=os.path.join(MODEL_DIR, "results.csv"),
                        help="Output CSV for results")
    args = parser.parse_args()
    test(args.csv, args.model, args.out)
