import pandas as pd
"""this module creates the decisiontarget for model
   by comparing it with tomorrow's closing price"""
def build_direction(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    
    df["close_tomorrow"]=df["Close"].shift(-1)
    df["direction"]=(df["close_tomorrow"]>df["Close"]).astype(int)
    df=df.dropna(subset=["direction"])#closing price is dropeed to prevent data leakage
    df=df.iloc[:-1]
    df.drop(columns=["close_tomorrow"],inplace=True)
    return df