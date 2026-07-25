import pandas as pd
import numpy as np


def main():

    df = pd.read_parquet("data/raw/spy_daily.parquet")
    df.columns = df.columns.get_level_values(0)

    vix = pd.read_parquet("data/raw/vix_daily.parquet")
    vix.columns = vix.columns.get_level_values(0)

    # returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    return_windows = [5, 10, 20, 60]
    for window in return_windows:
        df[f"ret_{window}d"] = np.log(df["Close"] / df["Close"].shift(window))

    # volitility
    vol_windows = [5, 10, 20, 60]
    for window in vol_windows:
        df[f"rv_{window}d"] = df["log_return"].rolling(window=window).std() * np.sqrt(
            252
        )

    # trend
    ma_windows = [5, 10, 20, 50]
    for window in ma_windows:
        df[f"ma_{window}d"] = df["Close"].rolling(window=window).mean()
        df[f"dist_ma_{window}d"] = (df["Close"] - df[f"ma_{window}d"]) / df[
            f"ma_{window}d"
        ]

    # volume
    df["volume_change_1d"] = np.log(df["Volume"] / df["Volume"].shift(1))

    # momentum
    rsi_window = 14

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()

    rs = avg_gain / avg_loss

    df["rsi_14"] = 100 - (100 / (1 + rs))

    lambda_ = 0.94

    df["ewma_vol"] = df["log_return"].ewm(
        alpha=1 - lambda_, adjust=False
    ).std() * np.sqrt(252)

    # range
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]

    df["overnight_gap"] = np.log(df["Open"] / df["Close"].shift(1))

    # target
    forecast_horizon = 5
    df[f"target_rv_{forecast_horizon}d"] = df["log_return"].rolling(
        window=forecast_horizon
    ).std().shift(-forecast_horizon) * np.sqrt(252)

    # vix
    df["vix"] = vix["Close"] / 100
    df["vix_change_1d"] = np.log(df["vix"] / df["vix"].shift(1))
    df["vix_rv_spread"] = df["vix"] - df["rv_20d"]
    df["vix_rv_ratio"] = df["vix"] / df["rv_20d"]

    df = df.dropna()

    df.to_parquet("data/processed/spy_features.parquet")
    df.to_csv("data/processed/spy_features.csv")

    print(f"Final dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()