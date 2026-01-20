import pandas as pd
import numpy as np
#generates RSI which decides how overbought or oversold a stock is
def compute_rsi(series:pd.Series,period:int=14)->pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
#add features to the dataframe (Returns,Trend,Momentum,Volataility,Volume)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicator features to OHLCV data.
    Assumes df is validated and indexed by DatetimeIndex.
    """

    df = df.copy()

    # returns
    df["return_pct"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    #Trend
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # momentum
    df["rsi_14"] = compute_rsi(df["Close"], period=14)

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # volatility 
    df["volatility_10"] = df["Close"].rolling(10).std()
    df["volatility_20"] = df["Close"].rolling(20).std()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # Volume
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

    return df