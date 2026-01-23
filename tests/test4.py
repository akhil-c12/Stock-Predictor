from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import add_features
from src.target import build_direction
from src.signal_generator import compute_signal
from src.train import train_model
from src.evaluate import evaluate
from src.split import time_split
from src.evaluate import find_best_threshold
from src.backtest import backtest_signals

def main():
    ticker = "AAPL"
    df = fetch_stock_data(ticker)
    print("\n✅ FETCH DONE")
    df=validate_ohlcv(df)
    print("\n✅ VALIDATION PASSED")
    df=add_features(df)
    df=build_direction(df)
    print("Shape:", df.shape)
    
    
    print(df.head())
    print(df.tail())
    X_train,X_test,y_train,y_test=time_split(df,target_col="direction",test_size=0.2)
    model=train_model(X_train,y_train)
    best_t = find_best_threshold(model, X_test, y_test, thresholds=[0.55,0.6,0.65,0.7,0.75])
    proba, preds = evaluate(model, X_test, y_test, threshold=best_t)
    results, bt_df = backtest_signals(
    df=X_test.join(y_test),  # keep same index
    proba=proba,
    threshold=best_t,
    fee=0.0005
)
    print("\n📈 BACKTEST RESULTS")
    for k, v in results.items():
        print(k, ":", v)
    print(df.columns)

if __name__ == "__main__":
    main()
