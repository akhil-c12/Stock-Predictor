from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import add_features
from src.target import build_direction
def main():
    ticker = "AAPL"
    df = fetch_stock_data(ticker)

    print("\n✅ FETCH DONE")
    print("Shape:", df.shape)
    df=add_features(df)
    df=build_direction(df)
    print(df.head())
    print(df.tail())
    validate_ohlcv(df)

    print("\n✅ VALIDATION PASSED")

if __name__ == "__main__":
    main()
