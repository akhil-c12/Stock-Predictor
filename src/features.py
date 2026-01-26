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

def compute_macd(series: pd.Series) -> tuple:
    """Computes MACD and signal line."""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature set with MACD, Bollinger Bands, and momentum indicators.
    """
    df = df.copy()

    #  TARGET (Classify: Will next day > 0.1%?)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    # Target: 1 if return > 0.1% (cover fees), else 0
    df["target"] = (df["log_return"].shift(-1) > 0.001).astype(int)

    # Price momentum
    df["returns_5d"] = df["close"].pct_change(5)
    df["returns_10d"] = df["close"].pct_change(10)

    # Long-term Context
    sma_50 = df["close"].rolling(50).mean()
    sma_200 = df["close"].rolling(200).mean()
    df["dist_sma_50"] = (df["close"] / sma_50) - 1
    df["dist_sma_200"] = (df["close"] / sma_200) - 1
    df["sma_crossover"] = (sma_50 > sma_200).astype(int)

    # Short-term Context
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_20 = df["close"].ewm(span=20, adjust=False).mean()
    df["dist_ema_12"] = (df["close"] / ema_12) - 1
    df["dist_ema_20"] = (df["close"] / ema_20) - 1

    # MOMENTUM - RSI
    df["rsi_14"] = compute_rsi(df["close"], period=14) / 100.0
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)

    # MOMENTUM - MACD
    macd, macd_signal = compute_macd(df["close"])
    df["macd"] = macd / df["close"]  # Normalize
    df["macd_signal"] = macd_signal / df["close"]
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # VOLATILITY - ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Wilder's Smoothing for ATR
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["atr_pct"] = atr / df["close"]  # Normalized to % of price

    # VOLATILITY - Bollinger Bands
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma_20 + (std_20 * 2)
    df["bb_lower"] = sma_20 - (std_20 * 2)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # 0-1

    # REGIME (ADX - Trend Strength)
    up = df["high"].diff()
    down = -df["low"].diff()
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
    vol_sma = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_sma
    df["vol_trend"] = df["volume"].rolling(10).mean() / df["volume"].rolling(20).mean()

    # Cleanup - select best features
    features = [
        "dist_sma_50", "dist_sma_200", "sma_crossover",
        "dist_ema_12", "dist_ema_20",
        "rsi_14", "rsi_14_lag1",
        "macd", "macd_signal", "macd_hist",
        "atr_pct", "bb_position",
        "adx", "vol_ratio", "vol_trend",
        "returns_5d", "returns_10d",
        "target"
    ]
    return df[features].dropna()
