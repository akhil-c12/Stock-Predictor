from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import get_xgb_features
from src.signal_generator import compute_signal
from src.split import time_split_3way
from src.train import train_model
from src.evaluate import evaluate, find_best_threshold
from src.evaluate import add_model_proba
def main():
    df = fetch_stock_data("AAPL", "2020-01-01", "2026-01-01")
    df = validate_ohlcv(df)
    df = get_xgb_features(df)

    df = df.dropna()

    X_train, X_val, X_test, y_train, y_val, y_test = time_split_3way(df)

    model = train_model(
        X_train, y_train,
        X_val, y_val,
        model_params={
            "n_estimators": 300,      # More trees for better stability
            "max_depth": 12,          # Reduced to prevent overfitting
            "min_samples_split": 8,   # Higher to prevent overfitting
            "min_samples_leaf": 4,    # Higher to prevent overfitting
            "max_features": "sqrt",   # Features per split
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    best_t = find_best_threshold(model, X_val, y_val)
    add_model_proba(df, model, best_t)
    evaluate(model, X_test, y_test, best_t)

    df_with_signals = compute_signal(df)
    print(df_with_signals.tail())
    print(df.columns)
if __name__ == "__main__":
    main()
