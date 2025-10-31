# backend/data_fetch.py
import os
import argparse
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_data(ticker: str = "AAPL", start: str = "2020-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetches historical data for `ticker` using yfinance and saves CSV in backend/data/.
    Returns the pandas DataFrame.
    """
    print(f"Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker and date range.")
    out_path = os.path.join(DATA_DIR, f"data_{ticker}.csv")
    df.to_csv(out_path)
    print(f"Saved CSV to: {out_path} (shape: {df.shape})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    get_data(args.ticker, args.start, args.end)
