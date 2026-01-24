import pandas as pd


def build_direction(df: pd.DataFrame) -> pd.DataFrame:
    """This module creates the decision target for the model
    by comparing it with tomorrow's closing price."""
    df = df.copy()
    df["direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df
