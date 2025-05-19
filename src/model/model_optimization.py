import pandas as pd
import numpy as np
import os
import joblib
import cupy as cp
import gc

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def model_load(path: str):
    ensemble_model = joblib.load(path)
    return ensemble_model


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


def validation_matrix(
        lightgbm_model, xgboost_model, catboost_model, X_val, y_val
):
    lightgbm_val_pred = lightgbm_model.predict(X_val)
    xgboost_val_pred = xgboost_model.predict(X_val)
    catboost_val_pred = catboost_model.predict(X_val)

    Val_preds_matrix = [lightgbm_val_pred, xgboost_val_pred, catboost_val_pred]
    Val_actuals_matrix = [y_val.values, y_val.values, y_val.values]

    Val_preds_matrix = np.stack(Val_preds_matrix).T
    print("Our combined OOF have shape:", Val_preds_matrix.shape)

    Val_actuals_matrix = np.stack(Val_actuals_matrix).T
    print("Our combined PRED have shape:", Val_actuals_matrix.shape)

    return Val_preds_matrix, Val_actuals_matrix


def multiple_rmse_scores(actual, predicted):
    if len(actual.shape) == 1:
        actual = actual[:, cp.newaxis]
    rmse = cp.sqrt(cp.mean((actual-predicted)**2.0, axis=0))
    return rmse


def compute_metric_rmse(p, y_val):
    rmse = np.sqrt(np.mean((p-y_val.values)**2.0))
    return rmse


def single_model_evaluation(models, Val_preds_matrix, y_val):
    best_score = 40
    best_index = -1

    for k, name in enumerate(models):
        s = compute_metric_rmse(Val_preds_matrix[:, k], y_val)
        if s < best_score:
            best_score = s
            best_index = k
        print(f'RMSE {s:0.5f} {name}')
    print()
    print(
        f'Best single model is {models[best_index]}'
        f'with RMSE = {best_score:0.5f}'
    )

    return best_score, best_index


def hill_climb_iteration(
        x_train2, best_ensemble, truth, ww, best_score, files
):
    nn = len(ww)
    best_index = -1
    best_weight = 0
    potential_ensemble = None

    for k, ff in enumerate(files):
        new_model = x_train2[:, k]
        m1 = cp.repeat(best_ensemble[:, cp.newaxis], nn, axis=1) * (1 - ww)
        m2 = cp.repeat(new_model[:, cp.newaxis], nn, axis=1) * ww
        mm = m1 + m2
        new_aucs = multiple_rmse_scores(truth, mm)
        new_score = cp.min(new_aucs).item()

        if new_score < best_score:
            best_score = new_score
            best_index = k
            ii = np.argmin(new_aucs).item()
            best_weight = ww[ii].item()
            potential_ensemble = mm[:, ii]

    del new_model, m1, m2, mm, new_aucs
    gc.collect()
    return best_score, best_index, best_weight, potential_ensemble


def compute_model_weights_df(
        models, weights, metrics
):
    model_names = ["lightgbm", "xgboost", "catboost"]
    wgt = calculate_weights(weights)
    rows = []
    for m, w, s in zip(models, wgt, metrics):
        rows.append({
            "model": model_names[m],
            "weight": w
        })
    df_weight = pd.DataFrame(rows)
    df_weight = df_weight.groupby(
        'model'
    ).agg('sum').reset_index().sort_values(
        'weight', ascending=False
    )
    return df_weight.reset_index(drop=True)


def calculate_weights(weights):
    wgt = np.array([1])
    for w in weights:
        wgt = wgt * (1 - w)
        wgt = np.concatenate([wgt, np.array([w])])
    return wgt


def ensemble_model(
        lightgbm_model, xgboost_model, catboost_model, x_train: pd.DataFrame,
        y_train: pd.DataFrame, x_val: pd.DataFrame, y_val: pd.DataFrame,
        model_path: pd.DataFrame, weight: pd.DataFrame
) -> None:

    try:
        with mlflow.start_run(run_name="Ensemble Model") as run:
            voting_regressor = VotingRegressor(
                estimators=[
                    ('lightgbm', lightgbm_model),
                    ('xgboost', xgboost_model),
                    ('catboost', catboost_model)
                ],
                weights=[
                    weight["lightgbm"], weight["xgboost"], weight["catboost"]
                ]
            )
            voting_regressor.fit(x_train, y_train)
            predict = voting_regressor.predict(x_val)

            mse = mean_squared_error(y_val, predict)
            rmse = mean_squared_error(y_val, predict, squared=False)
            mae = mean_absolute_error(y_val, predict)
            r2 = r2_score(y_val, predict)

            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("Root Mean Squared Error", rmse)
            mlflow.log_metric("Mean Absolute Error", mae)
            mlflow.log_metric("R2 Score", r2)

            mlflow.sklearn.log_model(voting_regressor, "ensemble_model")
            mlflow.log_artifact(__file__)

            run_id = run.info.run_id

            client = MlflowClient()
            model_uri = f"runs:/{run_id}/artifacts/ensemble_model"
            reg = mlflow.register_model(
                model_uri, "optimization_ensemble_model"
            )
            model_version = reg.version
            new_state = "Staging"

            client.transition_model_version_stage(
                 name="optimization_ensemble_model",
                 version=model_version,
                 stage=new_state,
                 archive_existing_versions=True
            )

    except Exception as e:
        raise Exception(f"Error model training : {e}")
    try:
        joblib.dump(
            voting_regressor, os.path.join(
                model_path, 'optimization_ensemble_model.pkl'
            )
        )
    except Exception as e:
        raise Exception(f"Error save model : {e}")


def main():
    USE_NEGATIVE_WGT = True
    MAX_MODELS = 1000
    TOL = 1e-5

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models"
    )

    processed_data_path = os.path.join(
         os.path.dirname(__file__), "..", "..", "datas", "processed"
    )

    train = load_data(
             os.path.join(processed_data_path, "train.csv")
        )

    validation = load_data(
             os.path.join(processed_data_path, "validation.csv")
        )

    X_train, y_train = train_val_test_split(train)
    X_val, y_val = train_val_test_split(validation)

    lightgbm_model = model_load(os.path.join(model_path, "lightgbm.pkl"))
    xgboost_model = model_load(os.path.join(model_path, "xgboost.pkl"))
    catboost_model = model_load(os.path.join(model_path, "catboost.pkl"))

    Val_preds_matrix, Val_actuals_matrix = validation_matrix(
        lightgbm_model, xgboost_model, catboost_model, X_val, y_val
    )

    files = [lightgbm_model, xgboost_model, catboost_model]

    best_score, best_index = single_model_evaluation(
        files, Val_preds_matrix, y_val
    )

    x_train2 = cp.array(Val_preds_matrix)
    best_ensemble = x_train2[:, best_index]
    truth = cp.array(y_val.values)

    start = -0.50 if USE_NEGATIVE_WGT else 0.01
    ww = cp.arange(start, 0.51, 0.01)

    models = [best_index]
    weights = []
    metrics = [best_score]
    old_best_score = best_score
    indices = [best_index]

    for kk in range(1_000_000):
        best_score, best_index, best_weight, potential_ensemble = (
            hill_climb_iteration(
                x_train2, best_ensemble, truth, ww, best_score, files
            )
        )

        indices.append(best_index)
        indices = list(np.unique(indices))

        if len(indices) > MAX_MODELS:
            print(f"=> We reached {MAX_MODELS} models")
            indices = indices[:-1]
            break

        if -1 * (best_score - old_best_score) < TOL:
            print(f"=> We reached tolerance {TOL}")
            break

        print(
            kk+1, 'New best RMSE', best_score,
            f'adding "{files[best_index]}"',
            'with weight', f'{best_weight:0.3f}'
        )

        models.append(best_index)
        weights.append(best_weight)
        metrics.append(best_score)
        best_ensemble = potential_ensemble
        old_best_score = best_score

    df_weight = compute_model_weights_df(models, weights, metrics)
    df_weight['weight'] = df_weight['weight'].round(4)
    weight = dict(zip(df_weight['model'], df_weight['weight']))

    ensemble_model(
             lightgbm_model, xgboost_model,
             catboost_model, X_train,
             y_train, X_val, y_val, model_path,
             weight
        )


if __name__ == "__main__":
    main()
