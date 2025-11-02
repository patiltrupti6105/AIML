# backend/utils.py
import numpy as np
import torch
import random
from stable_baselines3.common.vec_env import DummyVecEnv

def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception:
        pass

def make_vec_env_from_df(df, n_envs=1, env_class=None):
    if env_class is None:
        raise ValueError("env_class must be provided")
    def _thunk():
        return env_class(df)
    return DummyVecEnv([_thunk for _ in range(n_envs)])
