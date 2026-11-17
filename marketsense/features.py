from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OBSERVATION_COLUMNS = ["daily_return", "realized_volatility", "momentum_20d", "volume_change"]

def make_features(prices: pd.DataFrame):
    """Create interpretable raw features and standardized HMM observations."""
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float)
    returns = close.pct_change()
    raw = pd.DataFrame(index=prices.index)
    raw["daily_return"] = returns
    raw["realized_volatility"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    raw["momentum_20d"] = close.pct_change(20)
    raw["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    raw["ema20"] = close.ewm(span=20, adjust=False).mean()
    raw["ema50"] = close.ewm(span=50, adjust=False).mean()
    raw["trend"] = np.where(raw["ema20"] >= raw["ema50"], "Up", "Down")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    valid = raw.dropna(subset=OBSERVATION_COLUMNS).copy()
    if len(valid) < 100:
        raise ValueError("Not enough valid observations after feature calculation.")
    scaler = StandardScaler()
    X = scaler.fit_transform(valid[OBSERVATION_COLUMNS].to_numpy(dtype=float))
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return valid, X, scaler
