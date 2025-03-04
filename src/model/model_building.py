import pandas as pd
import os
import yaml

from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

from lightgbm import LGBMRegressor, callback
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback
from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor




class SaveAllIterationsCallback_Xgboost(TrainingCallback):
    def __init__(self, log_path):
        self.model_log = []
        self.log_path = log_path

    def after_iteration(self, model, epoch, evals_log):
        
        train_rmse = evals_log['validation_0']['rmse'][epoch]
        valid_rmse = evals_log['validation_1']['rmse'][epoch]

        evals_result = {
            'iteration': epoch,
            'train_rmse': train_rmse,
            'valid_rmse': valid_rmse
        }
        
        self.model_log.append(evals_result)

        with open(os.path.join(self.log_path, 'xgboost_model_iterations.json'), 'w') as f:
            json.dump(self.model_log, f, indent=4)


class SaveAllIterationsCallback_Catboost:
    def __init__(self, log_path):
        self.model_log = []
        self.log_path = log_path
         
    def after_iteration(self, info):
        evals_result = {
            'iteration': info.iteration,
            'train_rmse': info.metrics['validation_0']['RMSE'],
            'valid_rmse': info.metrics['validation_1']['RMSE']
        }

        self.model_log.append(evals_result)

        with open(os.path.join(self.log_path,'catboost_model_iterations.json'), 'w') as f:
            json.dump(self.model_log, f, indent=4)
            
        return True



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


def load_yaml(yaml_path : str) -> dict:
    try:
        with open(yaml_path, 'r') as file:
            size_params = yaml.safe_load(file)['model_building']
            return size_params
    except Exception as e:
        raise Exception(f"Error loading paramaters from {yaml_path} : {e}")
    

def log_callback(env):

    print(f"Iteration {env.iteration}: Train RMSE={env.evaluation_result_list[0][2]:.4f}, "
          f"Valid RMSE={env.evaluation_result_list[1][2]:.4f}")
    

model_log = []
def save_all_iterations_callback(env):
        
    evals_result = {
        'iteration': env.iteration,
        'train_rmse': env.evaluation_result_list[0][2],
        'valid_rmse': env.evaluation_result_list[1][2]
    }

    model_log.append(evals_result)

    log_path = os.path.join(os.path.dirname(__file__), "..", "..","models", "logs")
    with open(os.path.join(log_path,'lightgbm_model_iterations.json'), 'w') as f:
        json.dump(model_log, f, indent=4)



def lightgbm_final_model(lightgbm_params : dict, x_train : pd.DataFrame, y_train : pd.DataFrame, x_val : pd.DataFrame, y_val : pd.DataFrame, model_path : str):
    try:
        best_params = lightgbm_params
        lgbm_final = LGBMRegressor(**best_params).fit(
            x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_metric = 'rmse',
            callbacks = [log_callback, save_all_iterations_callback]
        )
    except (TypeError, ValueError) as e:
        raise Exception(f"Parameter error : {e}")
    except Exception as e:
            raise Exception(f"Error model training : {e}")
    
    try:
        joblib.dump(lgbm_final, os.path.join(model_path, 'lightgbm.pkl'))
        return lgbm_final
    except Exception as e:
            raise Exception(f"Error save model : {e}")


def xgboost_final_model(xgboost_params : dict, x_train : pd.DataFrame, y_train : pd.DataFrame, x_val : pd.DataFrame, y_val : pd.DataFrame, save_callback_xgboost, model_path : str):
    try:
        xgboost_best_params = xgboost_params
        xgboost_final = XGBRegressor(**xgboost_best_params).fit(
            x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            callbacks = [save_callback_xgboost]
        )
    except (TypeError, ValueError) as e:
        raise Exception(f"Parameter error : {e}")
    except Exception as e:
            raise Exception(f"Error model training : {e}")
    try:
        joblib.dump(xgboost_final, os.path.join(model_path, 'xgboost.pkl'))
        return xgboost_final
    except Exception as e:
            raise Exception(f"Error save model : {e}")


def catboost_final_model(catboost_params : dict, x_train : pd.DataFrame, y_train : pd.DataFrame, x_val : pd.DataFrame, y_val : pd.DataFrame, save_callback_catboost,  model_path : str):
    try:
        catboost_best_params = catboost_params
        
        catboost_final = CatBoostRegressor(**catboost_best_params).fit(
            x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            callbacks = [save_callback_catboost],
            logging_level='Silent'
        )
    except (TypeError, ValueError) as e:
        raise Exception(f"Parameter error : {e}")
    except Exception as e:
            raise Exception(f"Error model training : {e}")
    try:
        joblib.dump(catboost_final, os.path.join(model_path, 'catboost.pkl'))
        return catboost_final
    except Exception as e:
            raise Exception(f"Error save model : {e}")



def ensemble_model(lightgbm, xgboost, catboost, x_train : pd.DataFrame, y_train : pd.DataFrame, model_path : pd.DataFrame) ->None:
    try: 
        voting_regressor = VotingRegressor(
            estimators=[
                ('lightgbm', lightgbm),
                ('xgboost', xgboost),
                ('catboost', catboost)
            ]
        )
        voting_regressor.fit(x_train, y_train)
    except Exception as e:
        raise Exception(f"Error model training : {e}")
    try:
        joblib.dump(voting_regressor, os.path.join(model_path, 'ensemble_model.pkl'))
    except Exception as e:
            raise Exception(f"Error save model : {e}")



def main():

    processed_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "datas", "processed")
    model_path = os.path.join(os.path.dirname(__file__), "..", "..","models")
    log_path = os.path.join(os.path.dirname(__file__), "..", "..","models", "logs")
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..")
    
    os.makedirs(log_path, exist_ok=True)

    try:
        train = load_data(os.path.join(processed_data_path, "train.csv"))
        validation = load_data(os.path.join(processed_data_path, "validation.csv"))

        X_train, y_train = train_val_test_split(train)
        X_val, y_val = train_val_test_split(validation)

        model_params =load_yaml(os.path.join(yaml_path, 'params.yaml'))
        
        save_callback_xgboost = SaveAllIterationsCallback_Xgboost(log_path)
        save_callback_catboost = SaveAllIterationsCallback_Catboost(log_path)

        lightgbm_model = lightgbm_final_model(model_params["lightgbm_model"], X_train, y_train, X_val, y_val, model_path)
        xgboost_model = xgboost_final_model(model_params["xgboost_model"], X_train, y_train, X_val, y_val, save_callback_xgboost, model_path)
        catboost_model = catboost_final_model(model_params["catboost_model"], X_train, y_train, X_val, y_val, save_callback_catboost, model_path)

        ensemble_model(lightgbm_model, xgboost_model, catboost_model, X_train, y_train, model_path)
    except Exception as e:
        raise Exception(f"An error occured: {e}")
    
if __name__ == "__main__":
    main()

