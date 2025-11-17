# backend/data_fetch.py
import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for OHLCV only.
    """
    print("  [Preprocessing] Handling missing values...")
    price_cols = ['Open', 'High', 'Low', 'Close']

    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()

    if 'Volume' in df.columns:
        median_vol = df['Volume'].median()
        df['Volume'] = df['Volume'].fillna(median_vol)
        df['Volume'] = df['Volume'].fillna(0)

    df = df.dropna(subset=['Close'])

    return df


def validate_ohlc_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLC: High >= Low, etc.
    Remove invalid rows.
    """
    print("  [Preprocessing] Validating OHLC relationships...")
    initial = len(df)

    invalid = pd.Series([False] * len(df), index=df.index)

    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        invalid |= (df['High'] < df['Open'])
        invalid |= (df['High'] < df['Low'])
        invalid |= (df['High'] < df['Close'])
        invalid |= (df['Low'] > df['Open'])
        invalid |= (df['Low'] > df['Close'])

    df = df[~invalid]

    removed = initial - len(df)
    if removed > 0:
        print(f"    Removed {removed} invalid OHLC rows")

    return df


def remove_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep latest row for duplicate dates.
    """
    print("  [Preprocessing] Removing duplicate dates...")
    if 'Date' in df.columns:
        df = df.drop_duplicates(subset=['Date'], keep='last')
        df = df.sort_values('Date').reset_index(drop=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only basic preprocessing — NO derived features.
    """
    print("\n[Data Preprocessing Pipeline]")
    print("=" * 60)

    df = remove_duplicate_dates(df)
    df = handle_missing_values(df)
    df = validate_ohlc_relationships(df)

    # Convert Date to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    print("[Preprocessing Complete]\n")
    return df


def get_data(ticker: str = "AAPL", start: str = "2021-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch only OHLCV data.
    """
    print(f"\n[Data Fetching] Downloading {ticker} {start} → {end}")

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df.empty:
        raise RuntimeError("No data downloaded!")

    df = df.reset_index()

    # If MultiIndex (happens sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Remove Adj Close completely
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    # Keep only OHLCV
    keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in keep_cols if c in df.columns]]

    # Run preprocessing
    df = preprocess_data(df)

    # Save output in exact 6-column format
    out_path = os.path.join(DATA_DIR, f"data_test_{ticker}.csv")
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()
    get_data(args.ticker, args.start, args.end)
