# backend/train.py
import os
import argparse
import pandas as pd
from stable_baselines3 import DQN
from environment import StockTradingEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODEL_DIR, exist_ok=True)


def train(data_file: str, total_timesteps: int = 50000, save_name: str = "dqn_trading_agent"):
    """
    Train a DQN agent on data_file (CSV) and save model to backend/model/.
    """
    print(f"Loading data from {data_file} ...")
    df = pd.read_csv(data_file)
    env = StockTradingEnv(df)

    print("Initializing DQN model...")
    model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=1e-3,
    exploration_final_eps=0.01,
    exploration_fraction=0.3,
)



    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = os.path.join(MODEL_DIR, f"{save_name}.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=os.path.join(DATA_DIR, "data_AAPL.csv"),
                        help="Path to CSV file containing stock data")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps for training")
    parser.add_argument("--name", type=str, default="dqn_trading_agent", help="Model save name (without extension)")
    args = parser.parse_args()
    train(args.csv, args.timesteps, args.name)
