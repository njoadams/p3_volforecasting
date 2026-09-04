import pandas as pd
import numpy as np
from src.evaluation import calculate_metrics, save_metrics, save_forecasts


LAMBDA = 0.94


def main():
    df = pd.read_csv(
        'data/processed/spy_features.csv',
        index_col='Date',
        parse_dates=True
    )

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    ewma_variance = (
        df['log_return']
        .pow(2)
        .ewm(alpha=1 - LAMBDA, adjust=False)
        .mean()
    )

    df['ewma_forecast'] = np.sqrt(ewma_variance.shift(1)) * np.sqrt(252)

    val_pred = df.loc[val.index, 'ewma_forecast']
    test_pred = df.loc[test.index, 'ewma_forecast']

    val_mae, val_rmse = calculate_metrics(
        val['target_rv_5d'],
        val_pred
    )

    test_mae, test_rmse = calculate_metrics(
        test['target_rv_5d'],
        test_pred
    )

    print('\nEWMA Model')
    print(f'Validation MAE:  {val_mae:.6f}')
    print(f'Validation RMSE: {val_rmse:.6f}')
    print(f'Test MAE:        {test_mae:.6f}')
    print(f'Test RMSE:       {test_rmse:.6f}')

    save_metrics(
        'EWMA',
        test_mae,
        test_rmse
    )

    save_forecasts(
        'EWMA',
        test.index,
        test['target_rv_5d'],
        test_pred
    )


if __name__ == '__main__':
    main()