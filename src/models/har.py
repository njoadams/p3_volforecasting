import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation import calculate_metrics, save_metrics, save_forecasts


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

    features = ["rv_5d", "rv_20d", "rv_60d"]

    X_train = train[features]
    y_train = train["target_rv_5d"]

    X_val = val[features]
    y_val = val["target_rv_5d"]

    X_test = test[features]
    y_test = test["target_rv_5d"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae, val_rmse = calculate_metrics(y_val, val_pred)
    test_mae, test_rmse = calculate_metrics(y_test, test_pred)

    print("\nHAR Model")
    print(f"Validation MAE:  {val_mae:.6f}")
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"Test MAE:        {test_mae:.6f}")
    print(f"Test RMSE:       {test_rmse:.6f}")

    save_metrics("HAR", test_mae, test_rmse)

    save_forecasts("HAR", test.index, y_test, test_pred)


if __name__ == "__main__":
    main()
