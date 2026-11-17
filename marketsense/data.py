from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch daily adjusted OHLCV history using yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install -r requirements.txt") from exc
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker cannot be empty.")
    if not symbol.endswith((".NS", ".BO")):
        symbol += ".NS"
    # Cache daily history as CSV in /tmp for six hours. Vercel's writable
    # filesystem is ephemeral, so a cold start may fetch the data again.
    cache_dir = Path("/tmp/marketsense_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (symbol.replace(".", "_") + "_" + period + ".csv")
    frame = None
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 6 * 60 * 60:
        try:
            frame = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except Exception:
            frame = None
    if frame is None or frame.empty:
        frame = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if frame is not None and not frame.empty:
            frame.to_csv(cache_file)
    if frame is None or frame.empty:
        raise ValueError(f"No data returned for {symbol}. Check the NSE ticker.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Close", "High", "Low", "Volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}")
    frame = frame[required].copy()
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Close"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if len(frame) < 260:
        raise ValueError(f"Only {len(frame)} daily observations were found. At least 260 are required for a meaningful regime analysis.")
    frame.attrs["ticker"] = symbol
    return frame
