import pandas as pd
import os
import mlflow
import mlflow.sklearn


from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor



def load_data(path : str) -> pd.DataFrame:
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
    

def GridSearch_lightgbm(x_train, y_train, x_val, y_val):
    try:
        with mlflow.start_run(run_name="LightGBM Hyperparameter Optimization") as parent_run:

            lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
                        "n_estimators": [200, 300, 350, 400],
                        "colsample_bytree": [0.9, 0.8, 1],
                        "device": ["gpu"]}
            
            lgbm = LGBMRegressor(verbosity=0, device='gpu')
            lgbm_search = GridSearchCV(lgbm,
                                       lgbm_params,
                                       cv=5,
                                       n_jobs=-1,
                                       verbose=0).fit(x_train, y_train)
            
            for i in range(len(lgbm_search.cv_results_["params"])):
                with mlflow.start_run(run_name=f"Experiment {i+1}", nested=True) as child_run:
                    mlflow.log_params(lgbm_search.cv_results_["params"][i])
                    mlflow.log_metric("mean test score", lgbm_search.cv_results_['mean_test_score'][i])

            mlflow.log_params(lgbm_search.best_params_)

            best_lightgbm = lgbm_search.best_estimator_
            best_lightgbm.fit(x_train, y_train)

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
    

def GridSearch_xgboost(x_train, y_train, x_val, y_val):
    try:
        with mlflow.start_run(run_name="XGBoost Hyperparameter Optimization") as parent_run:

            xgboost_params = {
                "learning_rate": [0.001, 0.01, 0.1],
                "max_depth": [5, 8, None],
                "n_estimators": [100, 500, 1000],
                "colsample_bytree": [None, 0.7, 1]
            }
            
            xgboost = XGBRegressor(verbosity=0, device='gpu')
            xgboost_search = GridSearchCV(xgboost,
                                          xgboost_params,
                                          cv=5,
                                          n_jobs=-1,
                                          verbose=0).fit(x_train, y_train)
            
            for i in range(len(xgboost_search.cv_results_["params"])):
                with mlflow.start_run(run_name=f"Experiment {i+1}", nested=True) as child_run:
                    mlflow.log_params(xgboost_search.cv_results_["params"][i])
                    mlflow.log_metric("mean test score", xgboost_search.cv_results_['mean_test_score'][i])

            mlflow.log_params(xgboost_search.best_params_)

            best_xgboost = xgboost_search.best_estimator_
            best_xgboost.fit(x_train, y_train)

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
    

def GridSearch_catboost(x_train, y_train, x_val, y_val):
    try:
        with mlflow.start_run(run_name="CatBoost Hyperparameter Optimization") as parent_run:

            catboost_params = {
                "iterations": [200, 500, 700],
                "learning_rate": [0.01, 0.02, 0.05, 0.1],
                "depth": [3, 6, None],
                "task_type": ["GPU"]
            }
            
            catboost = CatBoostRegressor()
            catboost_search = GridSearchCV(catboost,
                                           catboost_params,
                                           cv=5,
                                           n_jobs=1,
                                           verbose=False).fit(x_train, y_train)
            
            for i in range(len(catboost_search.cv_results_["params"])):
                with mlflow.start_run(run_name=f"Experiment {i+1}", nested=True) as child_run:
                    mlflow.log_params(catboost_search.cv_results_["params"][i])
                    mlflow.log_metric("mean test score", catboost_search.cv_results_['mean_test_score'][i])

            mlflow.log_params(catboost_search.best_params_)

            best_catboost = catboost_search.best_estimator_
            best_catboost.fit(x_train, y_train)

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

    mlflow.set_experiment("Financial_Risk_Score_Prediction_Experiments")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    processed_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "datas", "processed")


    try:
        train = load_data(os.path.join(processed_data_path, "train.csv"))
        validation = load_data(os.path.join(processed_data_path, "validation.csv"))

        X_train, y_train = train_val_test_split(train)
        X_val, y_val = train_val_test_split(validation)

        GridSearch_lightgbm(X_train, y_train, X_val, y_val)
        GridSearch_xgboost(X_train, y_train, X_val, y_val)
        GridSearch_catboost(X_train, y_train, X_val, y_val)

    except Exception as e:
        raise Exception(f"An error occured: {e}")
    
if __name__ == "__main__":
    main()
