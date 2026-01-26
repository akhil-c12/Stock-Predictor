import matplotlib.pyplot as plt


def plot_equity_drawdown(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["equity_strategy"], label="Strategy", linewidth=2)
    axes[0].plot(df["equity_buyhold"], label="Buy & Hold", linestyle="--")
    axes[0].set_title("Equity Curve")
    axes[0].legend()
    axes[0].grid(True)

    peak = df["equity_strategy"].cummax()
    drawdown = (df["equity_strategy"] / peak) - 1

    axes[1].fill_between(drawdown.index, drawdown, 0)
    axes[1].set_title("Strategy Drawdown")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
