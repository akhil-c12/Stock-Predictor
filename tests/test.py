from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv

def main():
    ticker = "AAPL"
    df = fetch_stock_data(ticker)

    print("\n✅ FETCH DONE")
    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)

    validate_ohlcv(df)

    print("\n✅ VALIDATION PASSED")

if __name__ == "__main__":
    main()
