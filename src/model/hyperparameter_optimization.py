import pandas as pd
import os
import mlflow
import mlflow.sklearn
import optuna

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import dagshub


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


def Optuna_lightgbm(x_train, y_train, x_val, y_val):
    try:
        with mlflow.start_run(run_name="LightGBM Hyperparameter Optimization"):

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 1000, 10000, step=500
                    ),
                    "num_leaves": trial.suggest_int(
                        "num_leaves", 20, 256
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", 3, 16
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.005, 0.3, log=True
                    ),
                    "min_child_samples": trial.suggest_int(
                        "min_child_samples", 5, 100
                    ),
                    "min_child_weight": trial.suggest_float(
                        "min_child_weight", 1e-3, 10.0, log=True
                    ),
                    "subsample": trial.suggest_float(
                        "subsample", 0.5, 1.0
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "reg_alpha": trial.suggest_float(
                        "reg_alpha", 1e-4, 10.0, log=True
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-4, 10.0, log=True
                    ),
                    "scale_pos_weight": trial.suggest_float(
                        "scale_pos_weight", 1.0, 10.0
                    ),
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": -1,
                    'device': 'gpu',
                    'eval_metric': 'RMSE',
                    'early_stopping_rounds': 100
                }

                model = LGBMRegressor(**params)
                model.fit(
                    x_train, y_train,
                    eval_set=[(x_val, y_val)],
                )

                preds = model.predict(x_val)
                rmse = mean_squared_error(y_val, preds, squared=False)

                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            for i in range(len(study.trials)):
                with mlflow.start_run(
                    run_name=f"Experiment {i+1}", nested=True
                ):
                    mlflow.log_params(study.trials[i].params)
                    mlflow.log_metric(
                        "Root Mean Squared Error",
                        study.trials[i].values[0]
                    )

            mlflow.log_params(study.best_params)

            best_lightgbm = LGBMRegressor(**study.best_params).fit(
                x_train, y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)]
            )

            predict = best_lightgbm.predict(x_val)

            mse = mean_squared_error(y_val, predict)
            rmse = mean_squared_error(y_val, predict, squared=False)
            mae = mean_absolute_error(y_val, predict)
            r2 = r2_score(y_val, predict)

            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("Root Mean Squared Error", rmse)
            mlflow.log_metric("Mean Absolute Error", mae)
            mlflow.log_metric("R2 Score", r2)

            mlflow.sklearn.log_model(best_lightgbm, "BestModel")
            mlflow.log_artifact(__file__)

    except Exception as e:
        raise Exception(f"An error occured: {e}")


def Optuna_xgboost(x_train, y_train, x_val, y_val):
    try:
        with mlflow.start_run(run_name="XGBoost Hyperparameter Optimization"):

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 1000, 10000, step=500
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", 3, 15
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-3, 1.0, log=True
                    ),
                    "subsample": trial.suggest_float(
                        "subsample", 0.5, 1.0
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "colsample_bylevel": trial.suggest_float(
                        "colsample_bylevel", 0.5, 1.0
                    ),
                    "colsample_bynode": trial.suggest_float(
                        "colsample_bynode", 0.5, 1.0
                    ),
                    "min_child_weight": trial.suggest_float(
                        "min_child_weight", 1, 10
                    ),
                    "reg_alpha": trial.suggest_float(
                        "reg_alpha", 1e-3, 10.0, log=True
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-3, 10.0, log=True
                    ),
                    "gamma": trial.suggest_float("gamma", 0, 10.0),
                    "scale_pos_weight": trial.suggest_float(
                        "scale_pos_weight", 0.5, 10.0
                    ),
                    "random_state": 42,
                    "n_jobs": -1,
                    "device": "cuda",
                    "eval_metric": "rmse",
                }

                model = XGBRegressor(**params)
                model.fit(
                    x_train, y_train,
                    eval_set=[(x_val, y_val)],
                    verbose=0
                )

                preds = model.predict(x_val)
                rmse = mean_squared_error(y_val, preds, squared=False)

                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            for i in range(len(study.trials)):
                with mlflow.start_run(
                    run_name=f"Experiment {i+1}", nested=True
                ):
                    mlflow.log_params(study.trials[i].params)
                    mlflow.log_metric(
                        "Root Mean Squared Error",
                        study.trials[i].values[0]
                    )

            mlflow.log_params(study.best_params)

            best_xgboost = XGBRegressor(**study.best_params).fit(
                x_train, y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                verbose=0
            )

            predict = best_xgboost.predict(x_val)

            mse = mean_squared_error(y_val, predict)
            rmse = mean_squared_error(y_val, predict, squared=False)
            mae = mean_absolute_error(y_val, predict)
            r2 = r2_score(y_val, predict)

            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("Root Mean Squared Error", rmse)
            mlflow.log_metric("Mean Absolute Error", mae)
            mlflow.log_metric("R2 Score", r2)

            mlflow.sklearn.log_model(best_xgboost, "BestModel")
            mlflow.log_artifact(__file__)

    except Exception as e:
        raise Exception(f"An error occured: {e}")


def Optuna_catboost(x_train, y_train, x_val, y_val):
    try:
        with mlflow.start_run(run_name="CatBoost Hyperparameter Optimization"):

            def objective(trial):
                grow_policy = trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
                )

                boosting_type = (
                    trial.suggest_categorical(
                        "boosting_type", ["Plain", "Ordered"]
                    )
                    if grow_policy == "SymmetricTree"
                    else "Plain"
                )

                params = {
                    "iterations": trial.suggest_int(
                        "iterations", 1000, 10000, step=500
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.005, 0.3, log=True
                    ),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float(
                        "l2_leaf_reg", 1e-3, 10.0, log=True
                    ),
                    "bagging_temperature": trial.suggest_float(
                        "bagging_temperature", 0, 1.0
                    ),
                    "border_count": trial.suggest_int(
                        "border_count", 32, 255
                    ),
                    "random_strength": trial.suggest_float(
                        "random_strength", 0.1, 10
                    ),
                    "grow_policy": grow_policy,
                    "boosting_type": boosting_type,
                    "random_seed": 42,
                    "verbose": 0,
                    'eval_metric': 'RMSE',
                    "task_type": "GPU",
                    'early_stopping_rounds': 100
                }

                model = CatBoostRegressor(**params)
                model.fit(
                    x_train, y_train,
                    eval_set=[(x_val, y_val)],
                    verbose=0
                )

                preds = model.predict(x_val)
                rmse = mean_squared_error(y_val, preds, squared=False)

                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=3, show_progress_bar=True)

            for i in range(len(study.trials)):
                with mlflow.start_run(
                    run_name=f"Experiment {i+1}", nested=True
                ):
                    mlflow.log_params(study.trials[i].params)
                    mlflow.log_metric(
                        "Root Mean Squared Error",
                        study.trials[i].values[0]
                    )

            mlflow.log_params(study.best_params)

            best_catboost = CatBoostRegressor(**study.best_params).fit(
                x_train, y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)]
            )

            predict = best_catboost.predict(x_val)

            mse = mean_squared_error(y_val, predict)
            rmse = mean_squared_error(y_val, predict, squared=False)
            mae = mean_absolute_error(y_val, predict)
            r2 = r2_score(y_val, predict)

            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("Root Mean Squared Error", rmse)
            mlflow.log_metric("Mean Absolute Error", mae)
            mlflow.log_metric("R2 Score", r2)

            mlflow.sklearn.log_model(best_catboost, "BestModel")
            mlflow.log_artifact(__file__)

    except Exception as e:
        raise Exception(f"An error occured: {e}")


def main():

    dagshub.init(repo_owner='hakankocakk',
                 repo_name='Financial-Risk-Score-Prediction', mlflow=True)

    mlflow.set_experiment("Financial_Risk_Score_Prediction_Experiments")
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri(
        "https://dagshub.com/hakankocakk/"
        "Financial-Risk-Score-Prediction.mlflow"
    )

    processed_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "processed"
    )

    try:
        train = load_data(
            os.path.join(processed_data_path, "train.csv")
        )
        validation = load_data(
            os.path.join(processed_data_path, "validation.csv")
        )

        X_train, y_train = train_val_test_split(train)
        X_val, y_val = train_val_test_split(validation)

        Optuna_lightgbm(X_train, y_train, X_val, y_val)
        Optuna_xgboost(X_train, y_train, X_val, y_val)
        Optuna_catboost(X_train, y_train, X_val, y_val)

    except Exception as e:
        raise Exception(f"An error occured: {e}")


if __name__ == "__main__":
    main()
