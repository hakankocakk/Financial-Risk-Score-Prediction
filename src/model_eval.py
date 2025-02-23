import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




def load_data(path : str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path} : {e}")


def train_val_test_split(dataframe : pd.DataFrame):
    try:
        X_ = dataframe.loc[:, ~dataframe.columns.str.contains("RiskScore")]
        y_ = dataframe.loc[:, "RiskScore"]
        return X_, y_
    except KeyError as e:
        raise Exception(f"Dataframe not found : {e}")


def model_load(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error load model file {e}")


def regression_report(model, X, y, eval_path):
    try: 
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
    except Exception as e:
        raise Exception(f"An error occured: {e}")
    try:
        with open(os.path.join(eval_path, 'test_metrics.json'), 'w') as f:
                json.dump(metric_dict, f, indent=4)
    except Exception as e:
        raise Exception(f"An error write to test_metrics.json: {e}")
    

def main():

    data_path_input = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    eval_path = os.path.join(os.path.dirname(__file__), "model", "evaluation")
    model_path = model_path = os.path.join(os.path.dirname(__file__), "model", "ml_model")
    os.makedirs(eval_path, exist_ok=True)

    try:
        test = load_data(os.path.join(data_path_input, "test.csv"))
        X_test, y_test = train_val_test_split(test)
        model = model_load(os.path.join(model_path, "ensemble_model.pkl"))
        regression_report(model, X_test, y_test, eval_path)
    except Exception as e:
        raise Exception(f"An error occured: {e}")
    

if __name__ == "__main__":
    main()