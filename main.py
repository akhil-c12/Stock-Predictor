import config
from src.data_fetcher import fetch_stock_data
from src.data_validator import validate_ohlcv
from src.features import add_features
from src.target import build_direction
from src.split import time_split_3way
from src.train import train_model
from src.evaluate import evaluate, find_best_threshold
from src.backtest import backtest_trader_mode, optimize_trader_mode


def main():
    """Main function to run the pipeline."""
    # 1) Fetch + validate
    df = fetch_stock_data(
        ticker=config.TICKER,
        data_dir=config.DATA_DIR,
    )
    df = validate_ohlcv(df)

    # 2) Features + target
    df = add_features(
        df,
        rsi_period=config.RSI_PERIOD,
        sma_periods=config.SMA_PERIODS,
        ema_periods=config.EMA_PERIODS,
        macd_signal_period=config.MACD_SIGNAL_PERIOD,
        volatility_periods=config.VOLATILITY_PERIODS,
        atr_period=config.ATR_PERIOD,
        volume_sma_period=config.VOLUME_SMA_PERIOD,
    )
    df = build_direction(df)

    # 3) Time split (Train / Val / Test)
    X_train, X_val, X_test, y_train, y_val, y_test = time_split_3way(
        df,
        target_col=config.TARGET_COL,
        train_size=config.TRAIN_SIZE,
        val_size=config.VAL_SIZE,
    )

    # 4) Train model on TRAIN only
    model = train_model(X_train, y_train, config.MODEL_PARAMS)

    # 5) Classification check on TEST
    best_threshold = find_best_threshold(model, X_test, y_test)
    evaluate(model, X_test, y_test, threshold=best_threshold)

    # 6) Optimize Trader Mode on VAL
    proba_val = model.predict_proba(X_val)[:, 1]
    best_df = optimize_trader_mode(
        df=X_val.copy(),
        proba=proba_val,
        fee=config.FEE,
        max_allowed_drawdown=config.MAX_ALLOWED_DRAWDOWN,
        top_n=config.TOP_N_STRATEGIES,
        buy_thresholds=config.BUY_THRESHOLDS,
        sell_thresholds=config.SELL_THRESHOLDS,
        max_hold_days_list=config.MAX_HOLD_DAYS_LIST,
    )

    if best_df.empty:
        print("No strategy found under drawdown limit.")
        return

    print("Best Strategies:")
    print(best_df.to_string(index=False))

    # 7) FINAL Trader Mode Backtest on TEST
    best = best_df.iloc[0].to_dict()
    proba_test = model.predict_proba(X_test)[:, 1]
    results, _ = backtest_trader_mode(
        df=X_test.copy(),
        proba=proba_test,
        buy_threshold=float(best["buy_threshold"]),
        sell_threshold=float(best["sell_threshold"]),
        max_hold_days=int(best["max_hold_days"]),
        fee=config.FEE,
    )

    print("\nFinal Test Results (Unbiased):")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
