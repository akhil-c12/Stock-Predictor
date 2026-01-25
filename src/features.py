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


def add_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    sma_periods: list = [20, 50, 200],
    ema_periods: list = [12, 26],
    macd_signal_period: int = 9,
    volatility_periods: list = [20],
    atr_period: int = 14,
    volume_sma_period: int = 20,
    adx_period: int = 14  # <--- NEW PARAMETER
) -> pd.DataFrame:
    
    df = df.copy()

    # --- 1. Returns (unchanged) ---
    df["return_pct"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- 2. Trend (UPDATED: DISTANCE RATIOS) ---
    # Instead of raw prices, we calculate % distance from the average.
    # 0.05 means "5% above SMA". -0.02 means "2% below SMA".
    for period in sma_periods:
        sma = df["Close"].rolling(period).mean()
        df[f"dist_sma_{period}"] = (df["Close"] / sma) - 1 

    for period in ema_periods:
        ema = df["Close"].ewm(span=period, adjust=False).mean()
        df[f"dist_ema_{period}"] = (df["Close"] / ema) - 1

    # --- 3. Momentum (unchanged) ---
    # Note: You need your compute_rsi function defined elsewhere
    # df[f"rsi_{rsi_period}"] = compute_rsi(df["Close"], period=rsi_period) 
    
    # Recalculate EMAs for MACD (since we didn't save the raw columns above)
    ema_fast = df["Close"].ewm(span=ema_periods[0], adjust=False).mean()
    ema_slow = df["Close"].ewm(span=ema_periods[1], adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=macd_signal_period, adjust=False).mean()
    
    # NORMALIZE MACD: Divide by price to make it relative
    df["macd"] = df["macd"] / df["Close"]
    df["macd_signal"] = df["macd_signal"] / df["Close"]

    # --- 4. Volatility (ATR & ADX) ---
    # Calculate True Range (TR)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR (Using Wilder's Smoothing)
    df[f"atr_{atr_period}"] = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()
    
    # NORMALIZE ATR: ATR is a dollar amount. Divide by Close to get % Volatility
    df[f"atr_pct_{atr_period}"] = df[f"atr_{atr_period}"] / df["Close"]

    # --- NEW: ADX CALCULATION (Trend Strength) ---
    # 1. Calculate Directional Movement (+DM, -DM)
    up = df["High"].diff()
    down = -df["Low"].diff()
    
    # If up > down and up > 0, it's +DM. Else 0.
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    # If down > up and down > 0, it's -DM. Else 0.
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Convert to pandas Series for smoothing
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # 2. Smooth them using Wilder's method
    tr_smooth = tr.ewm(alpha=1/adx_period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / tr_smooth)

    # 3. Calculate ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df[f"adx_{adx_period}"] = dx.ewm(alpha=1/adx_period, adjust=False).mean()

    # --- 5. Volume (unchanged) ---
    df[f"volume_sma_{volume_sma_period}"] = df["Volume"].rolling(volume_sma_period).mean()
    df["volume_ratio"] = df["Volume"] / df[f"volume_sma_{volume_sma_period}"]

    return df