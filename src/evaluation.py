from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from pathlib import Path


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return mae, rmse


def save_metrics(model_name, mae, rmse):
    results_path = Path("results/model_metrics.csv")

    new_result = pd.DataFrame({"model": [model_name], "mae": [mae], "rmse": [rmse]})

    if results_path.exists():
        results = pd.read_csv(results_path)
        results = results[results["model"] != model_name]
        results = pd.concat([results, new_result], ignore_index=True)
    else:
        results = new_result

    results.to_csv(results_path, index=False)

def save_forecasts(model_name, dates, y_true, y_pred):
    forecasts_path = Path('results/forecasts')
    forecasts_path.mkdir(parents=True, exist_ok=True)

    forecasts = pd.DataFrame({
        'date': dates,
        'actual': y_true,
        'predicted': y_pred
    })

    file_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    forecasts.to_csv(forecasts_path / f'{file_name}_forecasts.csv', index=False)