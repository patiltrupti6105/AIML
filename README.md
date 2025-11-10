# Real-Time RL Trading Bot

Advanced reinforcement learning trading bot with real-time capabilities.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Everything
```bash
# Complete pipeline (fetch data + train + trade)
python run.py --mode all

# Or run steps individually:
python run.py --mode setup              # Fetch data
python run.py --mode train              # Train model
python run.py --mode trade --duration 60  # Trade for 60 minutes
```

## 📁 Project Structure