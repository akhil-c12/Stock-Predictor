from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import get_xgb_features
from src.split import time_split_3way
from src.train import train_model
from src.evaluate import evaluate
from src.backtest import backtest_trader_mode, optimize_trader_mode
from src.plotting import plot_equity_drawdown
import numpy as np


def main():
    # ---------- DATA ----------
    df_ohlcv = fetch_stock_data("AAPL", "2020-01-01", "2026-01-01")
    df_ohlcv = validate_ohlcv(df_ohlcv)
    df = get_xgb_features(df_ohlcv)
    df = df.dropna()

    # ---------- SPLIT ----------
    X_train, X_val, X_test, y_train, y_val, y_test = time_split_3way(df)

    # ---------- TRAIN ----------
    model = train_model(
        X_train, y_train,
        X_val, y_val,
        model_params={
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    # ---------- PROBABILITIES ----------
    X_val_scaled = model.scaler.transform(X_val)
    X_test_scaled = model.scaler.transform(X_test)

    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ---------- OPTIMIZE TRADER RULES (VAL ONLY) ----------
    print("\n🔍 Optimizing trader rules on VALIDATION...\n")

    opt_results = optimize_trader_mode(
        df=df_ohlcv.loc[X_val.index],
        proba=val_proba,
        fee_per_side=0.0005
    )

    print(opt_results)

    if opt_results.empty:
        print("❌ No valid trading strategy found.")
        return

    best = opt_results.iloc[0]

    buy_t = best["buy_threshold"]
    sell_t = best["sell_threshold"]
    hold_days = best["max_hold_days"]

    print("\n✅ Selected Trading Rules:")
    print(f"Buy threshold : {buy_t}")
    print(f"Sell threshold: {sell_t}")
    print(f"Max hold days : {hold_days}")

    # ---------- FINAL BACKTEST (TEST ONLY) ----------
    print("\n📊 Final Backtest on TEST set...\n")

    final_results, final_df = backtest_trader_mode(
        df=df_ohlcv.loc[X_test.index],
        proba=test_proba,
        buy_threshold=buy_t,
        sell_threshold=sell_t,
        max_hold_days=hold_days,
        fee_per_side=0.0005
    )

    print("📈 FINAL RESULTS")
    for k, v in final_results.items():
        print(f"{k:25s}: {v}")

    # ---------- OPTIONAL: MODEL METRICS ----------
    print("\n📌 Classification Metrics (for reference only)")
    evaluate(model, X_test, y_test, threshold=0.5)
    plot_equity_drawdown(final_df)


if __name__ == "__main__":
    main()
