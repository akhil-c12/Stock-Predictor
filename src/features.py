import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes the RSI for a given series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(
    df: pd.DataFrame,
    rsi_period: int,
    sma_periods: list,
    ema_periods: list,
    macd_signal_period: int,
    volatility_periods: list,
    atr_period: int,
    volume_sma_period: int,
) -> pd.DataFrame:
    """
    Adds technical indicator f
    eatures to OHLCV data.
    Assumes df is validated and indexed by DatetimeIndex.
    """
    df = df.copy()

    # Returns
    df["return_pct"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Trend
    for period in sma_periods:
        df[f"sma_{period}"] = df["Close"].rolling(period).mean()
    for period in ema_periods:
        df[f"ema_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

    # Momentum
    df[f"rsi_{rsi_period}"] = compute_rsi(df["Close"], period=rsi_period)
    df["macd"] = df[f"ema_{ema_periods[0]}"] - df[f"ema_{ema_periods[1]}"]
    df["macd_signal"] = (
        df["macd"].ewm(span=macd_signal_period, adjust=False).mean()
    )

    # Volatility
    for period in volatility_periods:
        df[f"volatility_{period}"] = df["Close"].rolling(period).std()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat(
        [high_low, high_close, low_close], axis=1
    ).max(axis=1)

    df[f"atr_{atr_period}"] = true_range.rolling(atr_period).mean()

    # Volume
    df[
        f"volume_sma_{volume_sma_period}"
    ] = (
        df["Volume"].rolling(volume_sma_period).mean()
    )
    df["volume_ratio"] = (
        df["Volume"] / df[f"volume_sma_{volume_sma_period}"]
    )

    return df
