
import pandas as pd
import numpy as np


BULLISH_WEIGHTS = {
    "rsi_oversold": 0.25,
    "ema_bullish": 0.30,
    "above_sma20": 0.20,
    "macd_bullish": 0.15,
    "volume_confirm": 0.10,
}

BEARISH_WEIGHTS = {
    "rsi_overbought": 0.25,
    "ema_bearish": 0.30,
    "below_sma20": 0.20,
    "macd_bearish": 0.25,
}
"""this module generates the reasons for the stock uprise or downfall 
   with confidence score"""

def compute_signal(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    directions = []
    confidences = []
    reasons_list = []

    for _, row in df.iterrows():
        bullish_score = 0.0
        bearish_score = 0.0
        reasons = []

        #  RSI
        if not np.isnan(row.get("rsi_14", np.nan)):
            if row["rsi_14"] < 30:
                bullish_score += BULLISH_WEIGHTS["rsi_oversold"]
                reasons.append("RSI oversold (<30)")
            elif row["rsi_14"] > 70:
                bearish_score += BEARISH_WEIGHTS["rsi_overbought"]
                reasons.append("RSI overbought (>70)")

        # EMA Trend
        if not np.isnan(row.get("ema_12", np.nan)) and not np.isnan(row.get("ema_26", np.nan)):
            if row["ema_12"] > row["ema_26"]:
                bullish_score += BULLISH_WEIGHTS["ema_bullish"]
                reasons.append("EMA(12) above EMA(26)")
            elif row["ema_12"] < row["ema_26"]:
                bearish_score += BEARISH_WEIGHTS["ema_bearish"]
                reasons.append("EMA(12) below EMA(26)")

        # SMA Trend
        if not np.isnan(row.get("sma_20", np.nan)):
            if row["Close"] > row["sma_20"]:
                bullish_score += BULLISH_WEIGHTS["above_sma20"]
                reasons.append("Price above SMA(20)")
            elif row["Close"] < row["sma_20"]:
                bearish_score += BEARISH_WEIGHTS["below_sma20"]
                reasons.append("Price below SMA(20)")

        # MACD Momentum
        if not np.isnan(row.get("macd", np.nan)) and not np.isnan(row.get("macd_signal", np.nan)):
            if row["macd"] > row["macd_signal"]:
                bullish_score += BULLISH_WEIGHTS["macd_bullish"]
                reasons.append("MACD above signal")
            elif row["macd"] < row["macd_signal"]:
                bearish_score += BEARISH_WEIGHTS["macd_bearish"]
                reasons.append("MACD below signal")

        # Volume Confirmation
        if not np.isnan(row.get("volume_ratio", np.nan)) and row["volume_ratio"] > 1.2:
            bullish_score += BULLISH_WEIGHTS["volume_confirm"]
            reasons.append("Volume expansion confirms move")

        # Direction & Confidence
        if bullish_score > bearish_score:
            direction = "UP"
        elif bearish_score > bullish_score:
            direction = "DOWN"
        else:
            direction = None  # neutral 

        confidence = min(abs(bullish_score - bearish_score), 1.0)

        directions.append(direction)
        confidences.append(confidence)
        reasons_list.append(reasons)

    df["direction"] = directions
    df["confidence"] = confidences
    df["reasons"] = reasons_list

    df = df.dropna(subset=["direction"])

    return df
