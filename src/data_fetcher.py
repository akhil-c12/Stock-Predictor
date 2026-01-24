from pathlib import Path
import pandas as pd
import yfinance as yf


REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}

"""This module fetches stock data from Yahoo finance and caches it for optimization"""
def fetch_stock_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    data_dir: Path = Path("data/cached"),
    force_refresh: bool = False,) -> pd.DataFrame:
  

    if not start or not end:
        raise ValueError("Both start and end dates must be explicitly provided.")

    data_dir.mkdir(parents=True, exist_ok=True)

    cache_file = data_dir / f"{ticker}_{interval}_{start}_{end}.csv"

    if cache_file.exists() and not force_refresh:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        if not REQUIRED_COLS.issubset(df.columns):
            raise ValueError("Cached data is invalid or corrupted.")

        df.index = df.index.tz_localize(None)
        return df

        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,  
            progress=False,
        )
  

    if df.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[list(REQUIRED_COLS)]
    df = df.sort_index()

    df.to_csv(cache_file)
    return df
