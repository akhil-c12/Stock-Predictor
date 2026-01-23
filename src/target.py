import pandas as pd
"""this module creates the decisiontarget for model
   by comparing it with tomorrow's closing price"""
def build_direction(df:pd.DataFrame)->pd.DataFrame:
   df=df.copy()
    
   df["direction"]=(df["Close"].shift(-1)>df["Close"]).astype(int)
   df=df.dropna()
   df=df.iloc[:-1]
   
   print(df.columns)
   print(df["direction"].value_counts())

   return df