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

def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    signals, confidences, reasons_list = [], [], []

    for _, row in df.iterrows():
        bull, bear = 0.0, 0.0
        reasons = []

        # RSI
        if pd.notna(row.get("rsi_14")):
            if row["rsi_14"] < 30:
                bull += BULLISH_WEIGHTS["rsi_oversold"]
                reasons.append("RSI oversold (<30)")
            elif row["rsi_14"] > 70:
                bear += BEARISH_WEIGHTS["rsi_overbought"]
                reasons.append("RSI overbought (>70)")

        # EMA
        if pd.notna(row.get("ema_12")) and pd.notna(row.get("ema_26")):
            if row["ema_12"] > row["ema_26"]:
                bull += BULLISH_WEIGHTS["ema_bullish"]
                reasons.append("EMA(12) above EMA(26)")
            else:
                bear += BEARISH_WEIGHTS["ema_bearish"]
                reasons.append("EMA(12) below EMA(26)")

        # SMA
        if pd.notna(row.get("sma_20")):
            if row["close"] > row["sma_20"]:
                bull += BULLISH_WEIGHTS["above_sma20"]
                reasons.append("Price above SMA(20)")
            else:
                bear += BEARISH_WEIGHTS["below_sma20"]
                reasons.append("Price below SMA(20)")

        # MACD
        if pd.notna(row.get("macd")) and pd.notna(row.get("macd_signal")):
            if row["macd"] > row["macd_signal"]:
                bull += BULLISH_WEIGHTS["macd_bullish"]
                reasons.append("MACD above signal")
            else:
                bear += BEARISH_WEIGHTS["macd_bearish"]
                reasons.append("MACD below signal")

        # Volume
        if row.get("volume_ratio", 0) > 1.2:
            bull += BULLISH_WEIGHTS["volume_confirm"]
            reasons.append("Volume expansion")

        signal = "UP" if bull > bear else "DOWN"
        confidence = abs(bull - bear) / max(bull + bear, 1e-6)

        signals.append(signal)
        confidences.append(confidence)
        reasons_list.append(reasons)

    df["signal"] = signals
    df["confidence"] = confidences
    df["reasons"] = reasons_list

    return df
