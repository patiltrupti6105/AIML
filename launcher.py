# launcher.py
"""
Easy Launcher - Start everything with one command
"""
import subprocess
import sys
import time
import os


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           📈 REAL-TIME RL TRADING BOT 🤖                      ║
    ║                                                               ║
    ║               Professional Trading System                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def launch_dashboard():
    """Launch the dashboard"""
    print("\n🚀 Launching Dashboard...")
    print("   Dashboard will open in your browser")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Launch streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "live_dashboard.py",
            "--server.headless", "true",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped")


def launch_trading_with_dashboard():
    """Launch both trading and dashboard"""
    import threading
    import config
    from realtime_trader import RealTimeTrader

    print("\n🚀 Starting Trading + Dashboard...")

    # ✅ Don't block if the model is missing — warn and continue
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n⚠️  Model not found: {config.MODEL_PATH}")
        print("   Proceeding without a trained model (heuristic policy will be used).")

    # Start dashboard in background thread
    def run_dashboard():
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "live_dashboard.py",
                "--server.headless", "true",
                "--server.port", "8501"  # ensure fixed port
            ], check=True)
        except Exception as e:
            print(f"Dashboard error: {e}")

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    print("   Dashboard starting...")
    time.sleep(3)
    print("   ✅ Dashboard running at http://localhost:8501")

    # Start trading (works with or without a model)
    print("\n   Starting trading bot...")

    try:
        trader = RealTimeTrader(
            config.MODEL_PATH,
            config.SYMBOLS[:1],  # Start with first symbol
            config.INITIAL_CAPITAL
        )

        print("\n   ✅ Trading started!")
        print("   📊 View live updates at http://localhost:8501")
        print("   Press Ctrl+C to stop\n")

        trader.run(duration_minutes=120)  # Run for 2 hours

    except KeyboardInterrupt:
        print("\n\n✅ Trading stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main_menu():
    """Display main menu"""
    print_banner()
    
    print("\n📋 What would you like to do?\n")
    print("   1. 🎨 Launch Dashboard Only (view past results)")
    print("   2. 🤖 Start Trading (no dashboard)")
    print("   3. 🚀 Start Trading + Live Dashboard")
    print("   4. 📊 Train New Model")
    print("   5. 📥 Fetch Data")
    print("   6. ⚙️  Full Setup (Data + Train + Trade + Dashboard)")
    print("   7. ❌ Exit")
    
    choice = input("\n👉 Enter your choice (1-7): ").strip()
    
    return choice


def main():
    """Main launcher"""
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            # Dashboard only
            launch_dashboard()
        
        elif choice == '2':
            # Trading only
            print("\n🤖 Starting trading...")
            subprocess.run([sys.executable, "run.py", "--mode", "trade"])
        
        elif choice == '3':
            # Trading + Dashboard
            launch_trading_with_dashboard()
        
        elif choice == '4':
            # Train model
            print("\n📊 Training model...")
            print("   This may take 10-30 minutes depending on your hardware")
            
            proceed = input("\n   Continue? (y/n): ").strip().lower()
            if proceed == 'y':
                subprocess.run([sys.executable, "run.py", "--mode", "train"])
        
        elif choice == '5':
            # Fetch data
            print("\n📥 Fetching data...")
            subprocess.run([sys.executable, "run.py", "--mode", "setup"])
        
        elif choice == '6':
            # Full setup
            print("\n⚙️  Running full setup...")
            print("   This will:")
            print("      1. Fetch data")
            print("      2. Train model")
            print("      3. Start trading")
            print("      4. Launch dashboard")
            print("\n   ⏰ This will take 15-45 minutes total")
            
            proceed = input("\n   Continue? (y/n): ").strip().lower()
            if proceed == 'y':
                # Fetch data
                print("\n📥 Step 1/4: Fetching data...")
                subprocess.run([sys.executable, "run.py", "--mode", "setup"])
                
                # Train
                print("\n📊 Step 2/4: Training model...")
                subprocess.run([sys.executable, "run.py", "--mode", "train"])
                
                # Trading + Dashboard
                print("\n🚀 Step 3/4 & 4/4: Starting trading and dashboard...")
                launch_trading_with_dashboard()
        
        elif choice == '7':
            # Exit
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")
            time.sleep(2)
        
        # Ask to continue
        if choice in ['1', '2', '3', '4', '5', '6']:
            print("\n")
            continue_choice = input("Return to main menu? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)