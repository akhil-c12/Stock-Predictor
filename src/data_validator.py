import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# checks wether the dataframe is valid or not
def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DateTimeIndex")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns:{missing_cols}")
    df = df[REQUIRED_COLUMNS].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    df["Volume"] = df["Volume"].fillna(0)

    if df.isna().any().any():
        raise ValueError("NaNs still present after validation")

    df.columns = df.columns.str.lower()

    return df