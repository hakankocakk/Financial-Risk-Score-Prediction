import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_path_input = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
eval_path = os.path.join(os.path.dirname(__file__), "model", "evaluation")
model_path = model_path = os.path.join(os.path.dirname(__file__), "model", "ml_model")

os.makedirs(eval_path)

def train_val_test_split(dataframe):

    X_ = dataframe.loc[:, ~dataframe.columns.str.contains("RiskScore")]
    y_ = dataframe.loc[:, "RiskScore"]

    return X_, y_


def model_load(model_path):

    ensemble_model = joblib.load(model_path)

    return ensemble_model


def regression_report(model, X, y):

    predict = model.predict(X)

    mse = mean_squared_error(y, predict)
    rmse = mean_squared_error(y, predict, squared=False)
    mae = mean_absolute_error(y, predict)
    r2 = r2_score(y, predict)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    metric_dict = {
        "Mean Squared Error" : mse,
        "Root Mean Squared Error" : rmse,
        "Mean Absolute Error": mae,
        "R2 Score" : r2
    }

    with open(os.path.join(eval_path, 'test_metrics.json'), 'w') as f:
            json.dump(metric_dict, f, indent=4)



test = pd.read_csv(os.path.join(data_path_input, "test.csv"))
X_test, y_test = train_val_test_split(test)

model = model_load(os.path.join(model_path, "ensemble_model.pkl"))

regression_report(model, X_test, y_test)
    