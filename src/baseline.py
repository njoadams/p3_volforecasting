import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    df = pd.read_parquet("data/processed/spy_features.parquet")

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    val_pred = val['rv_5d']
    test_pred = test['rv_5d']

    print('\nValidation:')
    print(pd.DataFrame({
        'actual': val['target_rv_5d'],
        'prediction': val_pred
    }).head())

    print('\nTest:')
    print(pd.DataFrame({
        'actual': test['target_rv_5d'],
        'prediction': test_pred
    }).head())

    val_mae = mean_absolute_error(val['target_rv_5d'], val_pred)
    val_rmse = mean_squared_error(
        val['target_rv_5d'],
        val_pred
    ) ** 0.5

    test_mae = mean_absolute_error(test['target_rv_5d'], test_pred)
    test_rmse = mean_squared_error(
        test['target_rv_5d'],
        test_pred
    ) ** 0.5

    print('\nPersistence Baseline')
    print(f'Validation MAE:  {val_mae:.6f}')
    print(f'Validation RMSE: {val_rmse:.6f}')
    print(f'Test MAE:        {test_mae:.6f}')
    print(f'Test RMSE:       {test_rmse:.6f}')


if __name__ == "__main__":
    main()