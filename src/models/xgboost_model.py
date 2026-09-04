import pandas as pd
from xgboost import XGBRegressor

from src.evaluation import calculate_metrics, save_metrics, save_forecasts


FEATURES = [
    "rv_5d", "rv_10d", "rv_20d", "rv_60d",
    "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "dist_ma_5d", "dist_ma_10d", "dist_ma_20d", "dist_ma_50d",
    "volume_change_1d", "rsi_14", "ewma_vol",
    "daily_range", "overnight_gap",
    "vix", "vix_change_1d", "vix_rv_spread", "vix_rv_ratio",
]


def main():
    df = pd.read_csv(
        "data/processed/spy_features.csv", index_col="Date", parse_dates=True
    )

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    X_train = train[FEATURES]
    y_train = train["target_rv_5d"]

    X_val = val[FEATURES]
    y_val = val["target_rv_5d"]

    X_test = test[FEATURES]
    y_test = test["target_rv_5d"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae, val_rmse = calculate_metrics(y_val, val_pred)
    test_mae, test_rmse = calculate_metrics(y_test, test_pred)

    print("\nXGBoost Model")
    print(f"Validation MAE:  {val_mae:.6f}")
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"Test MAE:        {test_mae:.6f}")
    print(f"Test RMSE:       {test_rmse:.6f}")

    save_metrics("XGBoost", test_mae, test_rmse)

    save_forecasts("XGBoost", test.index, y_test, test_pred)


if __name__ == "__main__":
    main()
