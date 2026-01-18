from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR=Path("data/cached")
DATA_DIR.mkdir(parents=True,exist_ok=True)

def fetch_stock_data(ticker:str,start:str="2015-01-01",end:str|None=None,interval:str="1d",force_refresh:bool=False)->pd.DataFrame:
    cache_file=DATA_DIR/f"{ticker}_{interval}.csv"
    if cache_file.exists() and not force_refresh:
        df=pd.read_csv(cache_file,index_col=0,parse_dates=True)
        return df
    df=yf.download(ticker,start=start,end=None,interval=interval,auto_adjust=False,progress=False)
    if df.empty:
        raise ValueError(f"No data Fetched for ticker: {ticker}")
    df=df[["Open","High","Low","Close","Volume"]]
    
    df=df.sort_index()
    df.to_csv(cache_file)
    return df
    
    
