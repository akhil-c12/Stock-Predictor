import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes the RSI for a given series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final, minimal, XGBoost-optimized feature set.
    Uses only stationary, normalized, regime-aware features.
    """
    df = df.copy()

    #  TARGET (Classify: Will next day > 0.1%?)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    # Target: 1 if return > 0.1% (cover fees), else 0
    df["target"] = (df["log_return"].shift(-1) > 0.001).astype(int)

    # Long-term Context
    sma_200 = df["Close"].rolling(200).mean()
    df["dist_sma_200"] = (df["Close"] / sma_200) - 1

    # Short-term Context
    ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
    df["dist_ema_20"] = (df["Close"] / ema_20) - 1

    # MOMENTUM
    df["rsi_14"] = compute_rsi(df["Close"], period=14) / 100.0
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)


    # VOLATILITY
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder's Smoothing for ATR
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["atr_pct"] = atr / df["Close"]  # Normalized to % of price

    # REGIME (ADX - Trend Strength)
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Smooth DM and TR
    tr_smooth = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean() / 100.0

    # VOLUME
    vol_sma = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / vol_sma

    # Cleanup
    features = [
        "dist_sma_200", "dist_ema_20", "rsi_14", "rsi_14_lag1", 
        "atr_pct", "adx", "vol_ratio", "target"
    ]
    return df[features].dropna()