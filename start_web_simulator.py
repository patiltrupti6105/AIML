#!/usr/bin/env python3
"""
Startup script for the web-based trading simulator
"""
import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import websockets
        return True
    except ImportError:
        print("❌ Missing dependencies!")
        print("\nPlease install:")
        print("  pip install fastapi uvicorn websockets")
        return False


def check_data():
    """Check if data exists"""
    import config
    data_dir = Path(config.DATA_DIR)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    csv_files = list(data_dir.glob("data*.csv"))
    if not csv_files:
        print(f"❌ No data files found in {data_dir}")
        print("\nPlease fetch data first:")
        print("  python run.py --mode setup")
        return False
    
    print(f"✅ Found {len(csv_files)} data files")
    return True


def check_model():
    """Check if model exists (optional)"""
    import config
    if os.path.exists(config.MODEL_PATH):
        print(f"✅ Model found: {config.MODEL_PATH}")
        return True
    else:
        print(f"⚠️  Model not found: {config.MODEL_PATH}")
        print("   Simulator will use heuristic strategy")
        print("   To train a model: python run.py --mode train")
        return False


def create_web_directory():
    """Create web directory structure"""
    web_dir = Path(__file__).parent / "web"
    web_dir.mkdir(exist_ok=True)
    
    # Check if files exist
    required_files = ["index.html", "styles.css", "script.js"]
    missing = [f for f in required_files if not (web_dir / f).exists()]
    
    if missing:
        print(f"❌ Missing web files: {missing}")
        print("\nPlease create the following files in the 'web/' directory:")
        for f in missing:
            print(f"  - web/{f}")
        return False
    
    print(f"✅ Web files ready in {web_dir}")
    return True


def start_server():
    """Start the FastAPI server"""
    print("\n" + "="*70)
    print("STARTING WEB SIMULATOR")
    print("="*70)
    
    print("\n🚀 Starting server...")
    print("   URL: http://localhost:8000")
    print("   Press Ctrl+C to stop\n")
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open("http://localhost:8000")
            print("✅ Browser opened")
        except:
            print("⚠️  Could not open browser automatically")
            print("   Please open: http://localhost:8000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start uvicorn
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "web_simulator:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
        return True
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return False


def main():
    """Main startup routine"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           📈 WEB-BASED TRADING SIMULATOR 🌐                   ║
    ║                                                               ║
    ║            Real-time visualization of historical data         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("Performing startup checks...\n")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check data
    if not check_data():
        return 1
    
    # Check model (optional)
    check_model()
    
    # Check web files
    if not create_web_directory():
        return 1
    
    print("\n✅ All checks passed!\n")
    
    # Start server
    if start_server():
        return 0
    else:
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
