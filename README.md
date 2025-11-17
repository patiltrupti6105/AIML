# 📈 Self-Learning Stock Trading Bot

**AIML Lab Project**  
**Team Members:**
- UEC2023163 - Trupti Patil
- UCE2023566 - Sanika Tavate
- UCE2023565 - Divyanshi Singh
- UCE2023563 - Ananya Shroff

---

## 🎯 Project Overview

This project implements a complete, end-to-end reinforcement learning-based trading system that learns to make profitable trading decisions through experience. Unlike typical "toy trading scripts," this provides a **modular pipeline** with clean separation of concerns and professional-grade simulation capabilities.

### Key Features

- **Deep Q-Network (DQN)** agent trained via Stable-Baselines3
- **Realistic trading simulation** with proper position tracking, bid-ask spreads, commission, and slippage
- **Web-based visualization** with real-time candlestick charts, technical indicators, and portfolio metrics
- **Risk management** including stop-loss, take-profit, and position sizing
- **Technical indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, and more
- **Complete data pipeline**: automated fetching, preprocessing, and validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Yahoo Finance│─────▶│ Data Fetcher │─────▶ CSV Files    │
│  └──────────────┘      └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ENVIRONMENT LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TradingEnvironment (Gymnasium)                      │  │
│  │  • OHLCV data processing                             │  │
│  │  • Technical indicators                              │  │
│  │  • Realistic execution (next bar open price)         │  │
│  │  • Position tracking (FIFO)                          │  │
│  │  • Risk management (stop-loss, take-profit)          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     AGENT LAYER                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DQN Agent (Stable-Baselines3)                       │  │
│  │  • Neural network policy                             │  │
│  │  • Experience replay                                 │  │
│  │  • Target network                                    │  │
│  │  • Epsilon-greedy exploration                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  INTERFACE LAYER                            │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   FastAPI    │◀────▶│  WebSocket   │                    │
│  │   Backend    │      │  Streaming   │                    │
│  └──────────────┘      └──────────────┘                    │
│         │                      │                            │
│         ▼                      ▼                            │
│  ┌─────────────────────────────────────────────────┐       │
│  │  Web UI (HTML/CSS/JS + Plotly)                  │       │
│  │  • Real-time candlestick charts                 │       │
│  │  • Portfolio metrics                             │       │
│  │  • Technical indicators visualization            │       │
│  │  • Learning curve display                        │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd trading-bot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Quick Start

Run the complete pipeline in sequence:

1. **Fetch data:**
```bash
python run.py --mode setup
```

2. **Train the model:**
```bash
python run.py --mode train --timesteps 50000
```

3. **Launch web simulator:**
```bash
python start_web_simulator.py
```

The web interface will open automatically at `http://localhost:8000`

---

## 📁 Project Structure

```
trading-bot/
├── config.py                    # Central configuration
├── data_fetcher.py              # Yahoo Finance data downloader
├── trading_environment.py       # Gymnasium trading environment
├── train.py                     # DQN training script
├── web_simulator.py             # FastAPI backend + WebSocket
├── start_web_simulator.py       # Startup script for web UI
├── run.py                       # Main CLI orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/                        # Historical market data (CSV)
│   └── data_test_AAPL.csv
│
├── models/                      # Trained models
│   ├── dqn_model.zip
│   └── evaluations.npz
│
├── results/                     # Training logs & metrics
│
└── web/                         # Frontend assets
    ├── index.html
    ├── styles.css
    └── script.js
```

---

## 🎮 Usage Guide

### Data Fetching

Fetch historical data for specific symbols:

```bash
python data_fetcher.py --ticker AAPL --start 2021-01-01 --end 2024-12-31
```

Or use the orchestrator:

```bash
python run.py --mode setup --symbols AAPL MSFT GOOGL
```

### Training

Train a DQN agent:

```bash
python run.py --mode train --symbol AAPL --timesteps 100000
```

**Training parameters** (in `config.py`):
- `TRAIN_TIMESTEPS`: Total training steps (default: 50,000)
- `LEARNING_RATE`: Adam optimizer learning rate (2.5e-4)
- `BUFFER_SIZE`: Replay buffer size (100,000)
- `BATCH_SIZE`: Mini-batch size (256)
- `GAMMA`: Discount factor (0.99)
- `EXPLORATION_FRACTION`: Fraction of training for exploration decay (0.1)

### Web Simulator

Launch the interactive web interface:

```bash
python start_web_simulator.py
```

**Features:**
- ✅ Select any available symbol
- ✅ Adjust playback speed (2-10 seconds per step)
- ✅ Toggle between model and heuristic strategy
- ✅ Play/Pause/Step controls
- ✅ Real-time charts with buy/sell markers
- ✅ Portfolio tracking and statistics
- ✅ Technical indicators visualization
- ✅ Learning curve display

---

## 🧠 How It Works

### 1. Trading Environment

The environment simulates realistic trading conditions:

**Observation Space:**
- Historical returns (lookback window)
- Current price (normalized)
- Portfolio state (balance, shares, position value)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)

**Action Space:**
- 0: Hold (do nothing)
- 1: Buy (risk-based position sizing)
- 2: Sell 50% of positions (FIFO)
- 3: Sell 100% (close all positions)

**Realistic Execution:**
- Actions execute at **next bar's open price** (not current close)
- Bid-ask spread modeling (0.1% spread)
- Commission (0.05% per trade)
- Slippage (0.05%)
- FIFO position tracking for proper P&L calculation
- Stop-loss checks at bar low, take-profit at bar high

**Reward Function:**
```python
step_pnl = net_worth_after - net_worth_before
reward = (step_pnl / net_worth_before) * 100  # Percentage points
if action != 0:  # Penalize unnecessary trading
    reward -= 0.02
```

### 2. DQN Agent

The agent uses Deep Q-Learning to learn optimal trading policies:

**Architecture:**
- Input: Observation vector (lookback returns + scalars)
- Hidden layers: [256, 256] (configurable)
- Output: Q-values for each action

**Training Process:**
1. Experience collection via epsilon-greedy exploration
2. Store transitions in replay buffer
3. Sample mini-batches for training
4. Update Q-network with temporal difference learning
5. Periodically update target network

**Key Techniques:**
- Experience replay for stable learning
- Target network to reduce overestimation
- Epsilon decay for exploration/exploitation balance
- Gradient clipping for training stability

### 3. Technical Indicators

The environment computes these indicators for decision-making:

| Indicator | Purpose | Period |
|-----------|---------|--------|
| SMA | Trend following | 10, 20, 50, 200 |
| EMA | Responsive trend | 12, 26 |
| RSI | Overbought/oversold | 14 |
| MACD | Momentum | 12/26/9 |
| Bollinger Bands | Volatility | 20 (±2σ) |
| ATR | Volatility measurement | 14 |
| ROC | Rate of change | 12 |

All indicators are z-score normalized for stable learning.

---

## 📊 Web Interface Features

### Main Dashboard

1. **Price Chart**
   - Candlestick visualization
   - Buy/sell markers (green triangles up, red triangles down)
   - Auto-scrolling to show latest data
   - Interactive pan/zoom

2. **Portfolio Metrics**
   - Net worth (real-time)
   - Cash balance
   - Unrealized P&L
   - Portfolio value chart

3. **Recent Actions**
   - Trade log with timestamps
   - Buy/sell/hold indicators
   - Number of shares traded

4. **Statistics**
   - Total return percentage
   - Current step
   - Cumulative reward

5. **Technical Indicators**
   - RSI chart (0-100 range)
   - MACD + signal line
   - Auto-scrolling synchronized with price chart

6. **Learning Curve**
   - Mean reward over training
   - ±1 standard deviation band
   - Shows model learning progress

### Controls

- **Connect**: Establish WebSocket connection
- **Play**: Start/resume simulation
- **Pause**: Pause simulation
- **Step**: Advance one bar at a time
- **Speed Slider**: Adjust playback speed (2-10 seconds)
- **Use Model Checkbox**: Toggle between trained model and heuristic

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Trading Parameters
```python
INITIAL_CAPITAL = 100_000.0      # Starting balance
COMMISSION = 0.0005               # 0.05% per trade
SLIPPAGE = 0.0005                 # 0.05%
MAX_POSITION_PCT = 0.9            # Max 90% capital in positions
STOP_LOSS_PCT = 0.05              # 5% stop loss
TAKE_PROFIT_PCT = 0.08            # 8% take profit
```

### Training Parameters
```python
TRAIN_TIMESTEPS = 50_000
LEARNING_RATE = 2.5e-4
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
GAMMA = 0.99
HIDDEN_LAYERS = [256, 256]
```

### Data Parameters
```python
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
LOOKBACK_WINDOW = 30              # Steps of returns in observation
LOOKBACK_DAYS = 60                # Days for initialization
```

---

## 🔬 Results & Performance

### Training Metrics

The model is evaluated periodically during training:
- **Evaluation frequency**: Every 5,000 steps
- **Episodes per evaluation**: 5
- **Best model saving**: Automatically saves best performing model

### Performance Indicators

Monitor these metrics in the web simulator:
- **Total Return**: Portfolio return vs. buy-and-hold
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Holding period

### Example Results

```
Symbol: AAPL
Training Steps: 50,000
Final Return: +23.5%
Max Drawdown: -8.2%
Win Rate: 58%
Total Trades: 127
```

---

## 🛠️ Development

### Adding New Indicators

1. Add indicator calculation in `trading_environment.py`:
```python
def _calculate_custom_indicator(self):
    # Your indicator logic
    return indicator_values
```

2. Add to observation space in `_get_obs()`:
```python
custom_ind = float(row.get("custom_indicator", 0.0))
scalars = np.array([..., custom_ind])
```

3. Update observation space dimensions

### Customizing Actions

Modify action space in `TradingEnvironment.__init__()`:
```python
self.action_space = Discrete(5)  # Add more actions
```

Implement action logic in `step()`:
```python
elif action == 4:
    # Your custom action
    self._execute_custom_action()
```

### Extending the Web UI

Add new charts in `web/script.js`:
```javascript
Plotly.newPlot("myChart", [{
    x: timestamps,
    y: values,
    type: "scatter"
}], layout);
```

---

## 🐛 Troubleshooting

### Common Issues

**No data files found:**
```bash
python run.py --mode setup --symbols AAPL
```

**Model not found:**
```bash
python run.py --mode train --symbol AAPL --timesteps 50000
```

**Web UI not loading:**
- Check that `web/` directory contains `index.html`, `styles.css`, `script.js`
- Verify port 8000 is not in use
- Check browser console for errors

**Training crashes:**
- Reduce `BATCH_SIZE` in `config.py`
- Ensure sufficient data (at least 252 bars)
- Check GPU memory if using CUDA

**WebSocket disconnects:**
- Increase `speed` parameter (slower playback)
- Check network stability
- Review browser console logs

---

## 📚 Technical Details

### Position Tracking

The environment tracks individual position lots with FIFO accounting:

```python
@dataclass
class Trade:
    entry_price: float
    shares: int
    entry_step: int
```

This enables:
- Accurate realized P&L calculation
- Proper tax lot tracking
- Individual position risk management

### Execution Realism

Unlike naive implementations, this system:
- ✅ Executes at **next bar's open** (not current close)
- ✅ Models bid-ask spread
- ✅ Applies realistic commission and slippage
- ✅ Checks stops at bar extremes (low/high)
- ✅ Respects position size limits

### Risk Management

Automated risk controls:
- **Stop Loss**: Exits at bar low if threshold breached
- **Take Profit**: Exits at bar high if target reached
- **Position Sizing**: Risk 2% of capital per trade
- **Max Position**: Limits total exposure to 90% of capital

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more RL algorithms (A3C, PPO, SAC)
- [ ] Implement multi-asset portfolio optimization
- [ ] Add sentiment analysis features
- [ ] Improve reward shaping
- [ ] Add backtesting with transaction costs
- [ ] Implement paper trading mode
- [ ] Add more technical indicators
- [ ] Improve web UI with more charts

---

## 📄 License

This project is for educational purposes as part of an AIML Lab project.

---

## 🙏 Acknowledgments

- **Stable-Baselines3** for RL implementations
- **Gymnasium** for environment framework
- **Yahoo Finance** for market data
- **Plotly** for interactive charts
- **FastAPI** for modern web framework

---

## 📞 Contact

For questions or issues, contact the project team:
- Trupti Patil (UEC2023163)
- Sanika Tavate (UCE2023156)
- Divyanshi Singh (UCE2023156)
- Ananya Shroff (UCE2023563)

---

## 🔮 Future Enhancements

### Planned Features
- Multi-timeframe analysis
- Alternative data integration
- Portfolio optimization
- Options trading support
- Crypto market support
- Live trading connector
- Mobile app interface
- Cloud deployment

### Research Directions
- Meta-learning for fast adaptation
- Multi-agent market simulation
- Attention mechanisms for time series
- Transformer-based models
- Interpretable RL decisions

---

**Happy Trading! 📈🚀**