"""Debug script: run a saved DQN policy step-by-step and print detailed per-step info.

Usage (from repo root):
    python backend/debug_trades.py --csv backend/data/data_AAPL.csv --model dqn_trading_agent.zip --steps 200

This prints: step, action, obs, balance_before, balance_after, shares_before, shares_after,
current_price, executed_price, portfolio_value.
"""
import argparse
import os
import pandas as pd
from stable_baselines3 import DQN
from environment import StockTradingEnv


def debug_run(data_file: str, model_file: str, steps: int = 200):
    model_path = os.path.join(os.path.dirname(__file__), "model", model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading data from {data_file} ...")
    df = pd.read_csv(data_file)
    env = StockTradingEnv(df)

    print(f"Loading model from {model_path} ...")
    model = DQN.load(model_path, env=env)

    obs, info = env.reset()

    transaction_fee_percent = 0.001
    slippage_percent = 0.0005

    print("step,action,trends,rsi,macd,signal,balance_before,balance_after,shares_before,shares_after,current_price,executed_price,portfolio_value")
    for i in range(steps):
        # model.predict expects an observation array (not a tuple)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        balance_before = float(env.balance)
        shares_before = int(env.shares_held)
        current_price = env._get_current_price()

        # Get technical indicators
        row = env.df.loc[env.current_step]
        
        # Trend analysis
        short_trend = "UP" if row['SMA_10'] > row['EMA_20'] else "DOWN"
        medium_trend = "UP" if row['EMA_20'] > row['EMA_50'] else "DOWN"
        trend_str = f"{short_trend}/{medium_trend}"
        
        # RSI
        rsi = row['RSI']
        rsi_signal = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        
        # MACD
        macd_signal = "BUY" if row['MACD'] > row['MACD_Signal'] else "SELL"
        macd_hist = "+" if row['MACD_Hist'] > 0 else "-"
        
        # Technical score from environment
        tech_score = info.get('tech_score', 0)
        signal_str = f"Strong {'BUY' if tech_score > 0.3 else 'SELL' if tech_score < -0.3 else 'NEUTRAL'}"

        if action == 1:
            executed_price = current_price * (1 + slippage_percent)
        elif action == 2:
            executed_price = current_price * (1 - slippage_percent)
        else:
            executed_price = current_price

        # print pre-step snapshot with indicators
        can_buy = balance_before >= executed_price * (1 + transaction_fee_percent) and shares_before < env.max_shares
        can_sell = shares_before > 0

        print(f"{i},{action},{trend_str},{rsi_signal},{macd_signal}{macd_hist},{signal_str},{balance_before:.2f},{'?',},{shares_before},{'?',},{current_price:.2f},{executed_price:.2f},{env._get_portfolio_value():.2f}")
        if not can_buy and action == 1:
            print(f"WARNING: Buy attempted but not allowed! Cash:{balance_before:.2f} Needed:{executed_price*(1+transaction_fee_percent):.2f}")
        if not can_sell and action == 2:
            print(f"WARNING: Sell attempted but no shares held!")

        # step environment
        obs, reward, terminated, truncated, info = env.step(action)

        balance_after = float(env.balance)
        shares_after = int(env.shares_held)

        # After-step summary (print second CSV row for the step for easier inspection)
        print(f"after_{i},{action},,,{balance_after:.2f},{shares_after},{env.current_step},{current_price:.2f},{executed_price:.2f},{env._get_portfolio_value():.2f}")

        if terminated or truncated:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV data file (relative to repo root)')
    parser.add_argument('--model', required=True, help='Model filename under backend/model, e.g. dqn_trading_agent.zip')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps to run')
    args = parser.parse_args()

    data_file = args.csv
    debug_run(data_file, args.model, steps=args.steps)
