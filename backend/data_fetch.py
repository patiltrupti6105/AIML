# backend/data_fetch.py
import os
import argparse
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_data(ticker: str = "AAPL", start: str = "2005-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch historical data for `ticker` using yfinance and save CSV in backend/data/.
    """
    print(f" Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise RuntimeError(" No data downloaded. Check ticker and date range.")

    # Reset index to have Date column
    df = df.reset_index()
    # Ensure column names match our environment (Close, High, Low, Volume)
    df.columns = [c[-1].strip() if isinstance(c, tuple) else c.strip() for c in df.columns]

    rename_map = {c: c.strip() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Fill missing columns if any
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")

    out_path = os.path.join(DATA_DIR, f"data_{ticker}.csv")
    df.to_csv(out_path, index=False)
    print(f" Saved clean CSV to: {out_path} (shape: {df.shape})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    get_data(args.ticker, args.start, args.end)
