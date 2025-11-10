# run.py
"""
Main entry point - Run everything from here
"""
import os
import sys
import argparse
from data_fetcher import fetch_multiple_symbols
import train
from realtime_trader import RealTimeTrader
import config


def setup():
    """Initial setup - fetch data"""
    print("="*70)
    print("SETUP - Fetching Data")
    print("="*70)
    
    results = fetch_multiple_symbols(config.SYMBOLS, period='1y')
    
    print(f"\nFetched data for {len(results)}/{len(config.SYMBOLS)} symbols")
    
    for symbol, df in results.items():
        print(f"  {symbol}: {len(df)} rows")
    
    return len(results) > 0


def train_model(symbol='AAPL', timesteps=None):
    """Train model"""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    model_path = train.train(symbol, timesteps)
    
    if model_path:
        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {model_path}")
        return True
    
    return False


def run_trading(symbols=None, duration=60):
    """Run real-time trading"""
    symbols = symbols or config.SYMBOLS[:1]  # Default to first symbol
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n❌ Model not found: {config.MODEL_PATH}")
        print("   Run training first: python run.py --mode train")
        return False
    
    print("\n" + "="*70)
    print("REAL-TIME TRADING")
    print("="*70)
    
    trader = RealTimeTrader(config.MODEL_PATH, symbols)
    trader.run(duration_minutes=duration)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Real-Time RL Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode setup                    # Fetch data
  python run.py --mode train                    # Train model
  python run.py --mode trade --duration 60      # Run trading
  python run.py --mode all                      # Do everything
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['setup', 'train', 'trade', 'all'],
                       help='Operation mode')
    parser.add_argument('--symbols', nargs='+',
                       help='Stock symbols')
    parser.add_argument('--timesteps', type=int,
                       help='Training timesteps')
    parser.add_argument('--duration', type=int, default=60,
                       help='Trading duration (minutes)')
    
    args = parser.parse_args()
    
    # Print config
    config.print_config()
    
    if args.mode == 'setup':
        success = setup()
    
    elif args.mode == 'train':
        symbol = args.symbols[0] if args.symbols else config.SYMBOLS[0]
        success = train_model(symbol, args.timesteps)
    
    elif args.mode == 'trade':
        symbols = args.symbols or config.SYMBOLS[:1]
        success = run_trading(symbols, args.duration)
    
    elif args.mode == 'all':
        print("\n🚀 Running complete pipeline...\n")
        
        # 1. Setup
        if not setup():
            print("❌ Setup failed")
            return 1
        
        # 2. Train
        symbol = args.symbols[0] if args.symbols else config.SYMBOLS[0]
        if not train_model(symbol, args.timesteps):
            print("❌ Training failed")
            return 1
        
        # 3. Trade
        symbols = args.symbols or [symbol]
        if not run_trading(symbols, args.duration):
            print("❌ Trading failed")
            return 1
        
        success = True
    
    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("❌ FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())