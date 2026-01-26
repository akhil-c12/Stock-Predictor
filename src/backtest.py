
import numpy as np
import pandas as pd

def optimize_trader_mode(
    df,
    proba,
    fee_per_side=0.0005,
    buy_thresholds=(0.40, 0.45, 0.50),
    sell_thresholds=(0.30, 0.35, 0.40),
    max_hold_days_list=(5, 10, 15),
    max_allowed_drawdown=-0.25,
    top_n=10,
):
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
                    fee_per_side=fee_per_side,
                )

                if results["max_drawdown"] < max_allowed_drawdown:
                    continue

                records.append(results)

    results_df = pd.DataFrame(records)

    if results_df.empty:
        return results_df

    return results_df.sort_values(
        by=["strategy_total_return", "max_drawdown"],
        ascending=[False, False],
    ).head(top_n)


def backtest_trader_mode(
    df: pd.DataFrame,
    proba: np.ndarray,
    buy_threshold: float = 0.45,
    sell_threshold: float = 0.40,
    max_hold_days: int = 10,
    fee_per_side: float = 0.0005,):
    

    df = df.copy().reset_index(drop=True)
    proba = np.asarray(proba).reshape(-1)

    if len(df) != len(proba):
        raise ValueError("Length mismatch between df and proba")

    df["proba"] = proba
    df["next_return"] = (df["close"].shift(-1) / df["close"]) - 1
    df = df.dropna().reset_index(drop=True)

    position = 0
    hold_days = 0

    positions = []
    trades = []

    trade_returns = []
    current_trade_return = 0.0

    for i in range(len(df)):
        p = df.loc[i, "proba"]
        r = df.loc[i, "next_return"]

        trade = 0

        if position == 0:
            if p >= buy_threshold:
                position = 1
                hold_days = 1
                trade = 1  # entry
                current_trade_return -= fee_per_side
        else:
            hold_days += 1
            current_trade_return += r

            exit_now = (p <= sell_threshold) or (hold_days >= max_hold_days)
            if exit_now:
                position = 0
                hold_days = 0
                trade = 1  # exit
                current_trade_return -= fee_per_side
                trade_returns.append(current_trade_return)
                current_trade_return = 0.0

        positions.append(position)
        trades.append(trade)

    df["position"] = positions
    df["trade"] = trades

    df["strategy_return"] = df["position"].shift(1) * df["next_return"]
    df["strategy_return"] -= df["trade"] * fee_per_side

    df["equity_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["equity_buyhold"] = (1 + df["next_return"]).cumprod()

    # ----- Metrics -----
    total_return_strategy = float(df["equity_strategy"].iloc[-1] - 1)
    total_return_buyhold = float(df["equity_buyhold"].iloc[-1] - 1)

    trade_returns = np.array(trade_returns)
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else 0.0

    peak = df["equity_strategy"].cummax()
    drawdown = (df["equity_strategy"] / peak) - 1
    max_drawdown = float(drawdown.min())

    results = {
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "max_hold_days": max_hold_days,
        "fee_per_side": fee_per_side,
        "trades": int(len(trade_returns)),
        "win_rate": win_rate,
        "strategy_total_return": total_return_strategy,
        "buyhold_total_return": total_return_buyhold,
        "max_drawdown": max_drawdown,
    }

    return results, df
