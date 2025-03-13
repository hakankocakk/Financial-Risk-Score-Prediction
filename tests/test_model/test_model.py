import os
import pandas as pd
import pytest
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "hakankocakk"
repo_name = "Financial-Risk-Score-Prediction"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "ensemble_model"


def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path} : {e}")


def train_val_test_split(dataframe: pd.DataFrame):
    try:
        X_ = dataframe.loc[:, ~dataframe.columns.str.contains("RiskScore")]
        y_ = dataframe.loc[:, "RiskScore"]
        return X_, y_
    except KeyError as e:
        raise Exception(f"Dataframe not found : {e}")


def get_latest_model_version():
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    return versions


def test_model_in_staging():
    model_versions = get_latest_model_version()
    if not model_versions:
        pytest.skip("No model found in 'Staging' stage.")
    assert len(model_versions) > 0, "No model found in the 'Staging' stage."


def test_model_loading():
    model_versions = get_latest_model_version()
    if not model_versions:
        pytest.skip("No model found in 'Staging' stage.")

    run_id = model_versions[0].run_id
    logged_model = f"runs:/{run_id}/{model_name}"

    try:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
    except Exception as e:
        pytest.fail(f"Failed to load the model: {e}")

    assert loaded_model is not None, "The loaded model is None."
    print(f"Model successfully loaded from {logged_model}.")


def test_model_performance():
    model_versions = get_latest_model_version()
    if not model_versions:
        pytest.skip("No model found in 'Staging' stage.")

    run_id = model_versions[0].run_id
    logged_model = f"runs:/{run_id}/{model_name}"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    processed_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "processed"
    )
    test = load_data(os.path.join(processed_data_path, "test.csv"))

    X_test, y_test = train_val_test_split(test)
    predict = loaded_model.predict(X_test)

    mse = mean_squared_error(y_test, predict)
    rmse = mean_squared_error(y_test, predict, squared=False)
    mae = mean_absolute_error(y_test, predict)
    r2 = r2_score(y_test, predict)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    assert mse <= 5, "Mean Squared Error is above threshold."
    assert rmse <= 2.24, "Root Mean Squared Error is above threshold."
    assert mae <= 1.42, "Mean Absolute Error is above threshold."
    assert r2 >= 0.91, "R2 Score is below threshold."
