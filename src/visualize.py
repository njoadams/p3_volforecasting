import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/model_metrics.csv")
    df = df.sort_values("rmse", ascending=True).reset_index(drop=True)

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_mae = ax.bar(
        [i - width / 2 for i in x], df["mae"], width, label="MAE"
    )
    bars_rmse = ax.bar(
        [i + width / 2 for i in x], df["rmse"], width, label="RMSE"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Error")
    ax.set_title("Model Accuracy Comparison (Test Set)")
    ax.legend()

    for bars in (bars_mae, bars_rmse):
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig("results/model_comparison.png", dpi=150)

    print("Saved results/model_comparison.png")


if __name__ == "__main__":
    main()
