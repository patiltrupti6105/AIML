# data_fetcher.py
"""
Robust data access for training + realtime.

Behavior:
- Try multiple yfinance paths (download + history) with fallback periods/intervals.
- Validate/clean to exact columns: Date,Open,High,Low,Close,Volume.
- If remote fetch fails or returns epoch/zero junk, generate synthetic OHLCV
  (deterministic per symbol) so the pipeline can proceed.

Public API:
  - fetch_multiple_symbols(symbols, period='1y', interval='1d') -> dict[symbol -> df]
  - DataFetcher(symbol).fetch_historical(...), .fetch_realtime(...)
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────
# Generate synthetic if downloads fail/invalid
ALLOW_SYNTHETIC = True
SYNTHETIC_DEFAULT_BARS = 500  # ~2y of dailies
SYNTHETIC_SEED_OFFSET = 13_337

os.makedirs(config.DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _tz_naive_local(series: pd.Series) -> pd.Series:
    s = series.copy()
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s.dt.tz_localize(None)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    # yfinance may return multiindex columns
    df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    # Bring index out as Date when it is datetime-like
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})

    rename = {
        "Adj Close": "Close",
        "adjclose": "Close",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "date": "Date",
    }
    df = df.rename(columns=rename)

    # Ensure required columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    if "Date" not in df.columns:
        df["Date"] = pd.NaT

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = _tz_naive_local(df["Date"])

    # Cast numerics
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where all OHLC are zero or NaN
    ohlc = df[["Open", "High", "Low", "Close"]].fillna(0.0)
    df = df[~(ohlc == 0.0).all(axis=1)]
    df = df.dropna(subset=["Close"])

    # Sort and keep shape
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def _looks_like_epoch_junk(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    years = df["Date"].dt.year
    # If everything is 1970/1971 and prices are ~zero -> junk
    if years.max() <= 1971:
        ohlc = df[["Open", "High", "Low", "Close"]].fillna(0.0)
        if (ohlc == 0.0).all().all():
            return True
    return False


def _validate_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_cols(df)

    # Remove negative/impossible prices
    for c in ["Open", "High", "Low", "Close"]:
        df = df[df[c] > 0]

    # Drop rows where High/Low are obviously wrong
    if not df.empty:
        bad = (df["High"] < df["Low"]) | (df["Low"] > df["High"])
        df = df[~bad]

    if _looks_like_epoch_junk(df):
        raise ValueError(f"{symbol}: data looks invalid (epoch/zeros)")

    # Need minimum rows for lookbacks
    if len(df) < max(60, getattr(config, "LOOKBACK_WINDOW", 30) + 10):
        raise ValueError(f"{symbol}: not enough rows ({len(df)}) after cleaning")

    return df


def _download_candidates(symbol: str, period: str, interval: str) -> List[Tuple[str, str, str]]:
    """
    Generate a list of (method, period, interval) to try in order.
    method ∈ {"download","history"}
    """
    # Try the requested period/interval first, then expand
    periods = [period, "2y", "5y", "1y"]
    intervals = [interval, "1d", "1wk"]
    tried = []
    for p in periods:
        for iv in intervals:
            tried.append(("download", p, iv))
    # Then history() which often succeeds where download fails
    for p in periods:
        for iv in intervals:
            tried.append(("history", p, iv))
    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for t in tried:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup


def _try_fetch(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    # download
    try:
        raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if raw is not None and not raw.empty:
            return pd.DataFrame(raw)
    except Exception:
        pass
    # history
    try:
        hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        if hist is not None and not hist.empty:
            return pd.DataFrame(hist)
    except Exception:
        pass
    return None


def _save_csv(df: pd.DataFrame, symbol: str) -> str:
    path = os.path.join(config.DATA_DIR, f"data_{symbol}.csv")
    df.to_csv(path, index=False, float_format="%.6f")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic generator (deterministic per symbol)
# ──────────────────────────────────────────────────────────────────────────────
def _seed_from_symbol(symbol: str) -> int:
    # stable small hash to int
    s = symbol.upper().encode("utf-8")
    h = 0
    for b in s:
        h = (h * 131 + b) % 2_147_483_647
    return (h + SYNTHETIC_SEED_OFFSET) % 2_147_483_647


def _make_synth(symbol: str, n: int = SYNTHETIC_DEFAULT_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(_seed_from_symbol(symbol))
    # Geometric Brownian Motion
    dt = 1.0
    mu = 0.0005     # drift per bar
    sigma = 0.02    # volatility per bar
    price0 = 100.0 + (rng.random() * 50.0)

    shocks = rng.normal(loc=(mu - 0.5 * sigma * sigma) * dt, scale=sigma * np.sqrt(dt), size=n)
    log_prices = np.log(price0) + np.cumsum(shocks)
    close = np.exp(log_prices)

    # Build OHLC around close with small intraday ranges
    spread = np.maximum(0.002 * close, 0.05)
    high = close + rng.uniform(0.2, 1.0, size=n) * spread
    low = close - rng.uniform(0.2, 1.0, size=n) * spread
    open_ = close + rng.normal(0.0, 0.3, size=n) * spread
    vol = rng.integers(1e5, 5e6, size=n)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": np.maximum.reduce([open_, high, close]),
            "Low": np.minimum.reduce([open_, low, close]),
            "Close": close,
            "Volume": vol.astype(float),
        }
    )
    df = _normalize_cols(df)
    return df


def _ensure_valid_or_synth(symbol: str, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        if not ALLOW_SYNTHETIC:
            raise ValueError(f"{symbol}: empty download and synthetic disabled")
        return _make_synth(symbol)
    try:
        return _validate_df(df, symbol)
    except Exception:
        if not ALLOW_SYNTHETIC:
            raise
        # Fall back to synthetic clean series
        return _make_synth(symbol)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def fetch_multiple_symbols(symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Try robust downloads for each symbol; if invalid, synthesize clean OHLCV.
    Always writes CSVs: data/data_<SYMBOL>.csv
    Returns dict of DataFrames.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        print(f"↳ Fetching {sym} ({period}, {interval})")
        df: Optional[pd.DataFrame] = None
        for method, p, iv in _download_candidates(sym, period, interval):
            try:
                df_try = _try_fetch(sym, p, iv)
                if df_try is None or df_try.empty:
                    continue
                df = _validate_df(df_try, sym)
                print(f"  ✓ {sym}: {len(df)} rows via {method}({p},{iv})")
                break
            except Exception as e:
                # try next candidate
                continue

        if df is None:
            # last resort: synthetic
            df = _ensure_valid_or_synth(sym, df)
            print(f"  • {sym}: using synthetic series ({len(df)} rows)")

        path = _save_csv(df, sym)
        print(f"    → saved {path}")
        out[sym] = df
    return out


class DataFetcher:
    """Wrapper used by realtime_trader."""

    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch_historical(self, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
        df: Optional[pd.DataFrame] = None
        for method, p, iv in _download_candidates(self.symbol, period, interval):
            try:
                df_try = _try_fetch(self.symbol, p, iv)
                if df_try is None or df_try.empty:
                    continue
                df = _validate_df(df_try, self.symbol)
                break
            except Exception:
                continue

        return _ensure_valid_or_synth(self.symbol, df)

    def fetch_realtime(self, interval: str = "1m") -> Optional[dict]:
        """
        Provide last close as a 'recent bar' if downloads fail.
        """
        try:
            hist = yf.Ticker(self.symbol).history(period="2d", interval=interval, auto_adjust=False)
            df = _ensure_valid_or_synth(self.symbol, pd.DataFrame(hist))
            row = df.iloc[-1]
            return {
                "timestamp": datetime.now(timezone.utc).astimezone().replace(microsecond=0),
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": float(row.get("Volume", 0.0)),
            }
        except Exception:
            # last-resort: synth one tick from synthetic history
            df = _make_synth(self.symbol, 2)
            row = df.iloc[-1]
            return {
                "timestamp": datetime.now(timezone.utc).astimezone().replace(microsecond=0),
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": float(row.get("Volume", 0.0)),
            }
