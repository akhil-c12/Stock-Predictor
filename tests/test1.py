from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import get_xgb_features
from src.signal_generator import compute_signal
def main():
    df=fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2023-01-01")
    print("Data fetched successfully")
    df=validate_ohlcv(df)
    print("Data validated successfully")
    df=get_xgb_features(df)
    print("Features computed successfully")
    df=compute_signal(df)
    print("Signals computed successfully")
    print(df.head())
    print(df.tail())
    print(df.columns)
if __name__ == "__main__":
    main()