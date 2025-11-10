# train.py
"""
Train DQN agent for real-time trading
"""
import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from trading_environment import TradingEnvironment
import logging
import config
import shutil

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """Custom callback for trading metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_trades = []
        self.episode_win_rates = []
    
    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            if 'episode' in self.locals:
                ep_info = self.locals['episode']
                if ep_info:
                    self.episode_rewards.append(ep_info['r'])
                    
                    if 'total_trades' in info:
                        self.episode_trades.append(info['total_trades'])
                    if 'win_rate' in info:
                        self.episode_win_rates.append(info['win_rate'])
        
        return True


def make_env(df):
    """Create monitored environment"""
    def _init():
        env = TradingEnvironment(df)
        env = Monitor(env)
        return env
    return _init


def train(symbol='AAPL', timesteps=None):
    """
    Train DQN agent
    
    Args:
        symbol: Stock symbol
        timesteps: Training timesteps (uses config if None)
    """
    timesteps = timesteps or config.TRAIN_TIMESTEPS
    
    logger.info("="*70)
    logger.info("TRAINING DQN AGENT")
    logger.info("="*70)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timesteps: {timesteps:,}")
    logger.info(f"Initial Balance: ${config.INITIAL_CAPITAL:,}")
    
    # Load data
    data_file = os.path.join(config.DATA_DIR, f'data_{symbol}.csv')
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.info("Run: python data_fetcher.py --symbols {symbol}")
        return None
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} data points")
    
    # Sort by date if available
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Train/val split
    split_idx = int(len(df) * config.TRAIN_SPLIT)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Create environments
    env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(val_df)])
    
    # Initialize DQN
    logger.info("Initializing DQN model...")
    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        learning_starts=config.LEARNING_STARTS,
        batch_size=config.BATCH_SIZE,
        tau=1.0,
        gamma=config.GAMMA,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config.TARGET_UPDATE_INTERVAL,
        exploration_fraction=config.EXPLORATION_FRACTION,
        exploration_initial_eps=1.0,
        exploration_final_eps=config.EXPLORATION_FINAL_EPS,
        max_grad_norm=10,
        policy_kwargs=dict(net_arch=config.HIDDEN_LAYERS),
        device='auto',
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_DIR,
        log_path=config.MODEL_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    trading_callback = TradingCallback(verbose=1)
    callbacks = CallbackList([eval_callback, trading_callback])
    
    # Train
    logger.info("\nStarting training...\n")
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        log_interval=100,
        progress_bar=True
    )
    
    # Save model
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.zip')
    
    if os.path.exists(best_model_path):
        shutil.move(best_model_path, config.MODEL_PATH)
        logger.info(f"\nBest model saved to {config.MODEL_PATH}")
    else:
        model.save(config.MODEL_PATH)
        logger.info(f"\nModel saved to {config.MODEL_PATH}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    
    return config.MODEL_PATH


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol')
    parser.add_argument('--timesteps', type=int, help='Training timesteps')
    
    args = parser.parse_args()
    
    train(args.symbol, args.timesteps)