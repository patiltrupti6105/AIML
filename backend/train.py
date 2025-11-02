# backend/train.py
import os
import argparse
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from environment import StockTradingEnv
import shutil
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(df):
    def _init():
        env = StockTradingEnv(df)
        env = Monitor(env)
        return env
    return _init


def train(data_file: str, total_timesteps: int = 50000, save_name: str = "dqn_trading_agent"):
    """
    Train a DQN agent on data_file (CSV) and save model to backend/model/.
    Uses an evaluation callback to save best model.
    """
    print(f"Loading data from {data_file} ...")
    df = pd.read_csv(data_file)
    # Ensure Date sorted
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    # Split train / eval (time-series split)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)

    env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(eval_df)])
    print("Environments created: train len", len(train_df), "eval len", len(eval_df))

 
    print("Initializing DQN model...")
    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=1e-3,
        exploration_final_eps=0.01,
        exploration_fraction=0.3,
        tensorboard_log=None,
        buffer_size=10000,
        train_freq=4,
        batch_size=64,
        target_update_interval=1000,
        learning_starts=1000,
    )

    # Eval callback to save best model
    save_path = os.path.join(MODEL_DIR)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=MODEL_DIR,
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Rename best_model.zip to dqn_trading_agent.zip (overwrite each time)
    best_model_path = os.path.join(MODEL_DIR, "best_model.zip")
    final_model_path = os.path.join(MODEL_DIR, f"{save_name}.zip")

    if os.path.exists(best_model_path):
        shutil.move(best_model_path, final_model_path)
        print(f"✅ Best model saved as {final_model_path}")
    else:
        # fallback if callback didn’t save best_model
        model.save(final_model_path)
        print(f"✅ Model saved directly to {final_model_path}")

    # evaluations.npz will already be in MODEL_DIR
    eval_file = os.path.join(MODEL_DIR, "evaluations.npz")
    if os.path.exists(eval_file):
        print(f"📊 Evaluations saved to {eval_file}")
    else:
        print("⚠️ No evaluations file found (callback may not have run yet).")

    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=os.path.join(DATA_DIR, "data_AAPL.csv"),
                        help="Path to CSV file containing stock data")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps for training")
    parser.add_argument("--name", type=str, default="dqn_trading_agent", help="Model save name (without extension)")
    args = parser.parse_args()
    train(args.csv, args.timesteps, args.name)
