from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import add_features
from src.target import build_direction
from src.train import train_model
from src.evaluate import evaluate, find_best_threshold
from src.split import time_split
from src.backtest import backtest_trader_mode, optimize_trader_mode


def main():
    ticker = "AAPL"

    # 1) Fetch + validate
    df = fetch_stock_data(ticker)
    print("\n✅ FETCH DONE")

    df = validate_ohlcv(df)
    print("\n✅ VALIDATION PASSED")

    # 2) Features + target
    df = add_features(df)
    df = build_direction(df)

    print("\n✅ DATA READY")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nDirection counts:\n", df["direction"].value_counts())

    # 3) Time split
    X_train, X_test, y_train, y_test = time_split(df, target_col="direction", test_size=0.2)
    print("\n✅ SPLIT DONE")
    print("Total:", len(df))
    print("Train:", len(X_train))
    print("Test :", len(X_test))

    # 4) Train model
    model = train_model(X_train, y_train)

    # 5) Find best threshold (classification perspective)
    best_t = find_best_threshold(model, X_test, y_test, thresholds=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    proba, preds = evaluate(model, X_test, y_test, threshold=best_t)

    # 6) Trader Mode Backtest (Version A)
    results, bt_df = backtest_trader_mode(
        df=X_test.copy(),      # must contain Close column
        proba=proba,
        buy_threshold=0.45,
        sell_threshold=0.40,
        max_hold_days=10,
        fee=0.0005
    )

    print("\n📈 TRADER MODE BACKTEST RESULTS")
    for k, v in results.items():
        print(f"{k} : {v}")

    # 7) Optimize Trader Mode for maximum returns (fast wrap-up)
    print("\n🔥 OPTIMIZING TRADER MODE (Top Results)...")
    best_df = optimize_trader_mode(
        df=X_test.copy(),
        proba=proba,
        fee=0.0005,
        max_allowed_drawdown=-0.30,  # allow up to -30% drawdown, adjust if needed
        top_n=10
    )

    if best_df.empty:
        print("❌ No strategy found under drawdown limit.")
    else:
        print(best_df.to_string(index=False))

        best = best_df.iloc[0].to_dict()
        print("\n✅ BEST FINAL STRATEGY (AUTO PICKED)")
        for k, v in best.items():
            print(f"{k} : {v}")


if __name__ == "__main__":
    main()
