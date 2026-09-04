import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.evaluation import calculate_metrics, save_metrics, save_forecasts


def main():
    df = pd.read_csv('data/processed/spy_features.csv', index_col='Date', parse_dates=True)
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    val_pred = val['rv_5d']
    test_pred = test['rv_5d']

    val_mae, val_rmse = calculate_metrics(
        val['target_rv_5d'],
        val_pred
    )

    test_mae, test_rmse = calculate_metrics(
        test['target_rv_5d'],
        test_pred
    )

    print('\nPersistence Baseline')
    print(f'Validation MAE:  {val_mae:.6f}')
    print(f'Validation RMSE: {val_rmse:.6f}')
    print(f'Test MAE:        {test_mae:.6f}')
    print(f'Test RMSE:       {test_rmse:.6f}')

    save_metrics(
        'Persistence',
        test_mae,
        test_rmse
    )

    save_forecasts(
        'Persistence',
        test.index,
        test['target_rv_5d'],
        test_pred
    )


if __name__ == "__main__":
    main()