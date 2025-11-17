# web_simulator.py
"""
FastAPI backend for web-based trading simulation with WebSocket streaming
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import  FileResponse
from fastapi.staticfiles import StaticFiles

import config
from trading_environment import TradingEnvironment

# Optional: load model if available
try:
    from stable_baselines3 import DQN
except Exception:
    DQN = None


app = FastAPI(title="Trading Simulator API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
static_dir = Path(__file__).parent / "web"
print(f"Static directory: {static_dir}")
print(f"Static directory exists: {static_dir.exists()}")

if static_dir.exists():
    # List files in static directory
    print("Files in static directory:")
    for file in static_dir.iterdir():
        print(f"  - {file.name}")
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    print("WARNING: Static directory not found!")


# ================================ Data Loading ================================
class DataManager:
    """Manages historical data files"""
    
    def __init__(self):
        self.data_dir = Path(config.DATA_DIR)
        self._cache: Dict[str, pd.DataFrame] = {}
        print(f"Data directory: {self.data_dir}")
        print(f"Data directory exists: {self.data_dir.exists()}")
    
    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from data directory"""
        if not self.data_dir.exists():
            print(f"Data directory does not exist: {self.data_dir}")
            return []
        
        symbols = []
        for file in self.data_dir.glob("data_test*.csv"):
            symbol = file.stem.replace("data_test_", "")
            symbols.append(symbol)
        
        print(f"Found symbols: {symbols}")
        return sorted(symbols)
    
    def get_available_dates(self, symbol: str) -> list[str]:
        """Get available date ranges for a symbol"""
        df = self.load_symbol_data(symbol)
        if df is None or df.empty:
            return []
        
        # Return unique years or year-month combinations
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            dates = df['Date'].dt.to_period('M').unique()
            return sorted([str(d) for d in dates])
        return []
    
    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a symbol"""
        if symbol in self._cache:
            return self._cache[symbol]
        
        file_path = self.data_dir / f"data_test_{symbol}.csv"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows for {symbol}")
            print(f"Columns: {df.columns.tolist()}")
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            
            self._cache[symbol] = df
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_date_range(self, symbol: str, date_str: str = "all") -> Optional[pd.DataFrame]:
        """Load data for a specific date range (year-month) or all data if date_str is 'all'"""
        df = self.load_symbol_data(symbol)
        if df is None or df.empty:
            print(f"No data loaded for {symbol}")
            return None
        
        # If "all" is specified, return entire dataset
        if date_str == "all" or date_str == "":
            print(f"Loading entire dataset for {symbol}: {len(df)} rows")
            return df
        
        try:
            # Ensure Date column exists
            if 'Date' not in df.columns:
                print(f"Warning: No Date column in {symbol} data")
                return df.tail(252)  # Return last year of data
            
            # Parse date_str (format: "2024-01" or "2024")
            if '-' in date_str:
                year, month = map(int, date_str.split('-'))
                mask = (df['Date'].dt.year == year) & (df['Date'].dt.month == month)
            else:
                year = int(date_str)
                mask = df['Date'].dt.year == year
            
            filtered = df[mask].copy()
            
            # Ensure we have enough data (need at least lookback + some trading days)
            min_required = config.LOOKBACK_WINDOW + 20
            if len(filtered) < min_required:
                print(f"Warning: Only {len(filtered)} rows for {symbol} {date_str}, using more data")
                # Expand to include more data
                if len(df) >= min_required:
                    # Get the last min_required rows that end with the requested date
                    if not filtered.empty:
                        end_date = filtered['Date'].max()
                        # Get rows up to and including end_date
                        mask = df['Date'] <= end_date
                        filtered = df[mask].tail(min_required).copy()
                    else:
                        # Just use recent data
                        filtered = df.tail(min_required).copy()
                else:
                    # Use all available data
                    filtered = df.copy()
            
            if filtered.empty:
                print(f"Warning: No data found for {symbol} {date_str}, using last year")
                return df.tail(252)
            
            print(f"Loaded {len(filtered)} rows for {symbol} {date_str}")
            return filtered
            
        except Exception as e:
            print(f"Error filtering date range: {e}")
            import traceback
            traceback.print_exc()
            return df.tail(252)  # fallback: last year


data_manager = DataManager()


# ================================ Model Loading ================================
def load_model(model_path: str = None) -> Optional[DQN]:
    """Load trained DQN model if available"""
    if DQN is None:
        return None
    
    model_path = model_path or config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    try:
        # Create dummy env for loading
        dummy_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Open': 100, 'High': 101, 'Low': 99, 'Close': 100, 'Volume': 1000000
        })
        dummy_env = TradingEnvironment(dummy_df)
        model = DQN.load(model_path, env=dummy_env, device="cpu")
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Could not load model: {e}")
        return None


# ================================ Simulator ================================
class TradingSimulator:
    """Handles WebSocket-based simulation streaming"""
    
    def __init__(self, df: pd.DataFrame, model: Optional[DQN] = None):
        self.df = df
        self.model = model
        self.env = TradingEnvironment(df)
        self.paused = False
        self.step_mode = False
        
    def _decide_action(self, obs) -> int:
        """Get action from model or heuristic"""
        if self.model is not None:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                return int(action)
            except Exception as e:
                print(f"Model prediction error: {e}")
        
        # Heuristic fallback
        lookback = config.LOOKBACK_WINDOW
        if self.env.current_step < lookback:
            return 0
        
        try:
            recent_closes = self.df['Close'].iloc[max(0, self.env.current_step-lookback):self.env.current_step]
            if len(recent_closes) > 1:
                last_ret = float((recent_closes.iloc[-1] - recent_closes.iloc[-2]) / recent_closes.iloc[-2])
                
                if last_ret > 0.003:
                    return 1  # Buy
                elif last_ret < -0.003 and self.env.shares_held > 0:
                    return 2  # Sell
        except Exception as e:
            print(f"Heuristic error: {e}")
        
        return 0  # Hold
    
    async def stream_simulation(
        self,
        websocket: WebSocket,
        speed: float = 0.5,
        lookback_bars: int = 60
    ):
        """Stream simulation results via WebSocket"""
        
        try:
            print("Starting simulation stream...")
            
            # Reset environment
            obs, _ = self.env.reset()
            print(f"Environment reset. Starting at step 0, total bars: {len(self.df)}")
            
            # Send initial candle batch (for chart pre-fill)
            initial_bars = min(lookback_bars, len(self.df))
            initial_data = []
            
            for i in range(initial_bars):
                row = self.df.iloc[i]
                initial_data.append({
                    'timestamp': row['Date'].isoformat() if 'Date' in row else str(i),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            await websocket.send_json({
                'type': 'initial',
                'data': initial_data,
                'total_bars': len(self.df),
                'symbol': getattr(self.env, 'symbol', 'Unknown')
            })
            print(f"Sent {len(initial_data)} initial bars")
            
            # Main simulation loop
            done = False
            truncated = False
            step_count = 0
            cumulative_reward = 0.0
            
            while not done and not truncated:
                # Check for control messages (non-blocking)
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=0.01
                    )
                    await self._handle_control(msg)
                except asyncio.TimeoutError:
                    pass
                
                # Handle pause
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if we're at the end
                if self.env.current_step >= len(self.df) - 2:
                    print("Reached end of data")
                    done = True
                    break
                
                # Get action
                action = self._decide_action(obs)
                
                # Validate action before executing
                price = float(self.df.iloc[self.env.current_step]["Close"])
                if action == 1:  # Buy
                    # Check if we can actually buy
                    cost_mult = 1.0 + self.env._cost_multiplier()
                    total_cost = price * cost_mult
                    if total_cost > self.env.balance:
                        action = 0  # Can't buy, hold instead
                elif action == 2:  # Sell
                    # Check if we have shares to sell
                    if self.env.shares_held <= 0:
                        action = 0  # Can't sell, hold instead
                
                # Execute step with error handling
                try:
                    obs, reward, done, truncated, info = self.env.step(action)
                    cumulative_reward += reward
                    step_count += 1
                    
                    # Only send action if it was actually executed
                    if action == 1 and not info.get("buy_executed", False):
                        action = 0  # Buy failed, show as hold
                    elif action == 2 and not info.get("sell_executed", False):
                        action = 0  # Sell failed, show as hold
                except IndexError as e:
                    print(f"Index error at step {step_count}: {e}")
                    done = True
                    break
                except Exception as e:
                    print(f"Step error: {e}")
                    import traceback
                    traceback.print_exc()
                    done = True
                    break
                
                # Get current candle - with bounds checking
                current_idx = self.env.current_step
                if current_idx >= len(self.df):
                    print(f"Warning: current_step {current_idx} >= len(df) {len(self.df)}")
                    done = True
                    break
                
                row = self.df.iloc[current_idx]
                current_price = float(row['Close'])
                # Use position_value from info (already calculated with correct price) for consistency
                shares_held = float(info.get('shares', self.env.shares_held))
                # Use position_value directly from info - it's already calculated correctly
                holdings_value = float(info.get('position_value', shares_held * current_price))
                shares_sold = int(info.get('shares_sold', 0))
                
                # Get indicator values from the environment's dataframe
                indicators = {}
                if 'rsi' in self.env.df.columns:
                    indicators['rsi'] = float(self.env.df.iloc[current_idx]['rsi'])
                if 'macd' in self.env.df.columns:
                    indicators['macd'] = float(self.env.df.iloc[current_idx]['macd'])
                if 'macd_signal' in self.env.df.columns:
                    indicators['macd_signal'] = float(self.env.df.iloc[current_idx]['macd_signal'])
                if 'sma10' in self.env.df.columns:
                    indicators['sma10'] = float(self.env.df.iloc[current_idx]['sma10'])
                if 'sma20' in self.env.df.columns:
                    indicators['sma20'] = float(self.env.df.iloc[current_idx]['sma20'])
                
                # Determine if action was executed and what type
                # If risk management sold shares, show as sell action
                if info.get("risk_exit", False) and shares_sold > 0:
                    executed_action = 2  # Show as sell
                elif action == 1 and info.get("buy_executed", False):
                    executed_action = 1  # Buy executed
                elif action == 2 and info.get("sell_executed", False):
                    executed_action = 2  # Sell executed
                else:
                    executed_action = 0  # Hold or no action
                
                update_msg = {
                    'type': 'update',
                    'timestamp': row['Date'].isoformat() if 'Date' in row else str(current_idx),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': current_price,
                    'volume': int(row['Volume']),
                    'action': executed_action,  # Action type (0=hold, 1=buy, 2=sell)
                    'shares_sold': shares_sold,  # Number of shares sold (0 if no sell)
                    'shares_bought': int(info.get('shares_bought', 0)),  # Shares bought on this step
                    'shares_held': int(shares_held),  # Current number of shares held
                    'reward': float(reward),
                    'cumulative_reward': float(cumulative_reward),
                    'portfolio': {
                        'cash': float(info.get('balance', self.env.balance)),
                        'holdings': holdings_value,  # Holdings value (shares * price)
                        'shares': shares_held,  # Number of shares
                        'portfolio_value': float(info.get('net_worth', self.env.net_worth)),
                        'unrealized_pl': float(info.get('net_worth', self.env.net_worth) - config.INITIAL_CAPITAL)
                    },
                    'indicators': indicators,
                    'step': step_count
                }
                
                await websocket.send_json(update_msg)
                
                # Speed control - ensure minimum 2 seconds per action
                if self.step_mode:
                    self.paused = True
                    self.step_mode = False
                else:
                    # Ensure at least 2 seconds between actions
                    actual_speed = max(speed, 2.0)
                    await asyncio.sleep(actual_speed)
            
            # Send completion
            final_value = self.env.net_worth
            total_return = ((final_value / config.INITIAL_CAPITAL) - 1) * 100
            
            await websocket.send_json({
                'type': 'complete',
                'final_portfolio_value': float(final_value),
                'total_return': float(total_return),
                'cumulative_reward': float(cumulative_reward),
                'total_steps': step_count
            })
            print(f"Simulation complete. Final value: ${final_value:.2f}, Return: {total_return:.2f}%")
            
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await websocket.send_json({
                    'type': 'error',
                    'message': str(e)
                })
            except:
                pass
    
    async def _handle_control(self, msg: dict):
        """Handle control messages from client"""
        cmd = msg.get('cmd', '')
        
        if cmd == 'pause':
            self.paused = True
            print("Paused")
        elif cmd == 'resume':
            self.paused = False
            print("Resumed")
        elif cmd == 'step':
            self.step_mode = True
            self.paused = False
            print("Step forward")


# ================================ API Endpoints ================================
@app.get("/")
async def root():
    """Serve the main HTML page"""
    html_path = static_dir / "index.html"
    print(f"Looking for HTML at: {html_path}")
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "Trading Simulator API - HTML not found", "version": "1.0"}


@app.get("/api/data")
async def get_available_data():
    """Get available symbols and dates"""
    try:
        symbols = data_manager.get_available_symbols()
        
        data_info = {}
        for symbol in symbols:
            dates = data_manager.get_available_dates(symbol)
            data_info[symbol] = dates
        
        return {
            "symbols": symbols,
            "data": data_info
        }
    except Exception as e:
        print(f"Error in get_available_data: {e}")
        import traceback
        traceback.print_exc()
        return {"symbols": [], "data": {}}


@app.get("/api/symbols")
async def get_symbols():
    """Get list of available symbols"""
    return {"symbols": data_manager.get_available_symbols()}


@app.websocket("/ws/simulate/{symbol}/{date}")
async def websocket_simulate(
    websocket: WebSocket,
    symbol: str,
    date: str = "all",
    speed: float = 0.5,
    use_model: int = 1
):
    """
    WebSocket endpoint for simulation streaming
    
    Args:
        symbol: Stock symbol
        date: Date range (YYYY-MM or YYYY) or "all" for entire dataset
        speed: Seconds between updates (default: 0.5)
        use_model: 1 to use trained model, 0 for heuristic
    """
    await websocket.accept()
    print(f"WebSocket connection accepted for {symbol} {date}")
    
    try:
        # Load data (use "all" if date is empty or not provided)
        date_param = date if date else "all"
        df = data_manager.load_date_range(symbol, date_param)
        if df is None or df.empty:
            error_msg = f'No data found for {symbol}'
            print(error_msg)
            await websocket.send_json({
                'type': 'error',
                'message': error_msg
            })
            await websocket.close()
            return
        
        # Load model if requested
        model = None
        if use_model:
            model = load_model()
        
        print(f"Starting simulation for {symbol} (entire dataset) with {len(df)} bars")
        
        # Create and run simulator
        simulator = TradingSimulator(df, model)
        await simulator.stream_simulation(websocket, speed=speed)
        
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "data_dir": str(config.DATA_DIR),
        "model_path": str(config.MODEL_PATH),
        "model_exists": os.path.exists(config.MODEL_PATH),
        "static_dir": str(static_dir),
        "static_exists": static_dir.exists()
    }


@app.get("/api/data/{symbol}/{date}")
async def get_data_info(symbol: str, date: str):
    """Get information about a specific data range"""
    try:
        df = data_manager.load_date_range(symbol, date)
        if df is None or df.empty:
            return {"error": "No data found", "symbol": symbol, "date": date}
        
        return {
            "symbol": symbol,
            "date": date,
            "rows": len(df),
            "start": df['Date'].min().isoformat() if 'Date' in df.columns else None,
            "end": df['Date'].max().isoformat() if 'Date' in df.columns else None,
            "columns": list(df.columns)
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "date": date}


@app.get("/api/learning-curve")
async def get_learning_curve():
    """Load and return learning curve data from evaluations.npz with smoothing"""
    try:
        eval_path = Path(config.MODEL_DIR) / "evaluations.npz"
        
        if not eval_path.exists():
            return {"error": "evaluations.npz not found", "path": str(eval_path)}
        
        # Load the npz file
        data = np.load(eval_path)
        
        # Extract data
        timesteps = data['timesteps']
        results = data['results']  # Shape: (n_evaluations, n_episodes)
        
        # Compute mean and std across evaluation episodes
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        
        # Apply Gaussian smoothing (using scipy if available, otherwise simple smoothing)
        try:
            from scipy.ndimage import gaussian_filter1d
            smooth_mean = gaussian_filter1d(mean_rewards, sigma=2)
            smooth_std = gaussian_filter1d(std_rewards, sigma=2)
        except ImportError:
            # Fallback: simple moving average smoothing
            def simple_smooth(data, window=3):
                smoothed = np.zeros_like(data)
                for i in range(len(data)):
                    start = max(0, i - window // 2)
                    end = min(len(data), i + window // 2 + 1)
                    smoothed[i] = np.mean(data[start:end])
                return smoothed
            smooth_mean = simple_smooth(mean_rewards, window=5)
            smooth_std = simple_smooth(std_rewards, window=5)
        
        return {
            "timesteps": timesteps.tolist(),
            "mean_rewards": smooth_mean.tolist(),
            "std_rewards": smooth_std.tolist(),
            "raw_mean": mean_rewards.tolist(),
            "raw_std": std_rewards.tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ================================ Main ================================
if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("TRADING SIMULATOR WEB SERVER")
    print("="*70)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Available symbols: {data_manager.get_available_symbols()}")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Model available: {os.path.exists(config.MODEL_PATH)}")
    print("="*70)
    print("\nStarting server on http://localhost:8000")
    print("Open browser to http://localhost:8000 to access the simulator\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")