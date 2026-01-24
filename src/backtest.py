import numpy as np
import pandas as pd


def backtest_trader_mode(
    df: pd.DataFrame,
    proba: np.ndarray,
    buy_threshold: float = 0.45,
    sell_threshold: float = 0.40,
    max_hold_days: int = 10,
    fee: float = 0.0005,
):
    """
    Trader-mode backtest:
    - Enter long when proba >= buy_threshold
    - Exit when proba <= sell_threshold OR max_hold_days reached
    - Earn next-day returns while holding position
    """

    df = df.copy()
    df = df.copy().reset_index(drop=True)
    proba = np.asarray(proba).reshape(-1)

    if len(proba) != len(df):
        raise ValueError(f"Length mismatch: proba={len(proba)} vs df={len(df)}")

    df["proba"] = proba
    # next day returns
    df["next_return"] = (
        df["Close"].shift(-1) / df["Close"]
    ) - 1

    df = df.dropna(subset=["next_return"]).reset_index(drop=True)

    # backtest loop
    position = 0
    hold_days = 0

    positions = []
    trades = []

    for i in range(len(df)):
        p = df["proba"].iloc[i]

        if position == 0:
            # enter
            if p >= buy_threshold:
                position = 1
                hold_days = 1
                trades.append(1)  # entry trade
            else:
                trades.append(0)

        else:
            # holding
            hold_days += 1
            exit_now = (p <= sell_threshold) or (hold_days >= max_hold_days)

            if exit_now:
                position = 0
                hold_days = 0
                trades.append(1)  # exit trade
            else:
                trades.append(0)

        positions.append(position)

    df["position"] = positions
    df["trade"] = trades

    # returns earned only when holding
    # We shift position by 1 because we decide to enter/exit at day i,
    # and the return is earned on day i+1.
    df["strategy_return"] = df["position"].shift(1) * df["next_return"]

    # apply fee on trade days
    df["strategy_return"] -= df["trade"] * fee

    # equity curves
    df["equity_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["equity_buyhold"] = (1 + df["next_return"]).cumprod()

    # metrics
    total_return_strategy = float(df["equity_strategy"].iloc[-1] - 1)
    total_return_buyhold = float(df["equity_buyhold"].iloc[-1] - 1)

    holding_days = int(df["position"].sum())
    win_rate = (
        float((df.loc[df["position"] == 1, "next_return"] > 0).mean())
        if holding_days > 0
        else 0.0
    )

    peak = df["equity_strategy"].cummax()
    dd = (df["equity_strategy"] / peak) - 1
    max_drawdown = float(dd.min())

    results = {
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "max_hold_days": max_hold_days,
        "fee": fee,
        "holding_days": holding_days,
        "trade_count": int(df["trade"].sum()),
        "win_rate_holding_days": win_rate,
        "strategy_total_return": total_return_strategy,
        "buyhold_total_return": total_return_buyhold,
        "max_drawdown": max_drawdown,
    }

    return results, df


def optimize_trader_mode(
    df,
    proba,
    fee=0.0005,
    buy_thresholds=None,
    sell_thresholds=None,
    max_hold_days_list=None,
    max_allowed_drawdown=-0.25,  # reject strategies worse than -25%
    top_n=10,
):
    """
    Searches best Trader Mode settings by strategy_total_return
    while keeping drawdown under control.
    Returns a DataFrame of top results.
    """

    if buy_thresholds is None:
        buy_thresholds = [0.35, 0.40, 0.45, 0.50]

    if sell_thresholds is None:
        sell_thresholds = [0.25, 0.30, 0.35, 0.40]

    if max_hold_days_list is None:
        max_hold_days_list = [5, 10, 15]  # ✅ FIXED (correct variable name)

    records = []

    for buy_t in buy_thresholds:
        for sell_t in sell_thresholds:
            if sell_t >= buy_t:
                continue

            for max_hold in max_hold_days_list:
                results, _ = backtest_trader_mode(
                    df=df,
                    proba=proba,
                    buy_threshold=buy_t,
                    sell_threshold=sell_t,
                    max_hold_days=max_hold,
                    fee=fee,
                )

                # skip too risky configs
                if results["max_drawdown"] < max_allowed_drawdown:
                    continue

                records.append(results)

    results_df = pd.DataFrame(records)

    if results_df.empty:
        print("❌ No strategies found under the drawdown limit.")
        return results_df

    results_df = results_df.sort_values(
        by=["strategy_total_return", "max_drawdown"], ascending=[False, False]
    ).head(top_n)

    return results_df
