import pandas as pd
import os

from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

from lightgbm import LGBMRegressor, callback
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback
from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor


data_path_input = os.path.join(os.path.dirname(__file__), "data", "processed")
model_path = os.path.join(os.path.dirname(__file__), "model", "ml_model")
log_path = os.path.join(os.path.dirname(__file__), "model", "log")

os.makedirs(model_path)
os.makedirs(log_path)



class SaveAllIterationsCallback_Xgboost(TrainingCallback):
    def __init__(self):
        self.model_log = []

    def after_iteration(self, model, epoch, evals_log):
        
        train_rmse = evals_log['validation_0']['rmse'][epoch]
        valid_rmse = evals_log['validation_1']['rmse'][epoch]

        evals_result = {
            'iteration': epoch,
            'train_rmse': train_rmse,
            'valid_rmse': valid_rmse
        }
        
        self.model_log.append(evals_result)

        with open(os.path.join(log_path, 'xgboost_model_iterations.json'), 'w') as f:
            json.dump(self.model_log, f, indent=4)

save_callback_xgboost = SaveAllIterationsCallback_Xgboost()



class SaveAllIterationsCallback_Catboost:
    def __init__(self):
        self.model_log = []
         
    def after_iteration(self, info):
        evals_result = {
            'iteration': info.iteration,
            'train_rmse': info.metrics['validation_0']['RMSE'],
            'valid_rmse': info.metrics['validation_1']['RMSE']
        }

        self.model_log.append(evals_result)

        # JSON dosyasına yazma
        with open(os.path.join(log_path,'catboost_model_iterations.json'), 'w') as f:
            json.dump(self.model_log, f, indent=4)
            
        return True

save_callback_catboost = SaveAllIterationsCallback_Catboost()



def train_val_test_split(train, validation):

    X_train = train.loc[:, ~train.columns.str.contains("RiskScore")]
    y_train = train.loc[:, "RiskScore"]

    X_val = validation.loc[:, ~validation.columns.str.contains("RiskScore")]
    y_val = validation.loc[:, "RiskScore"]
    
    return X_train, y_train, X_val, y_val


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

    with open(os.path.join(log_path,'lightgbm_model_iterations.json'), 'w') as f:
        json.dump(model_log, f, indent=4)



def lightgbm_final_model(x_train, y_train, x_val, y_val):

    best_params = {
        'colsample_bytree': 0.8,
        'device': 'gpu',
        'learning_rate': 0.05,
        'n_estimators': 400,
        'verbosity': -1
        }
    
    lgbm_final = LGBMRegressor(**best_params).fit(
        x_train, y_train,
        eval_set = [(x_train, y_train), (x_val, y_val)],
        eval_metric = 'rmse',
        callbacks = [log_callback, save_all_iterations_callback]
    )
    joblib.dump(lgbm_final, os.path.join(model_path, 'lightgbm.pkl'))

    return lgbm_final


def xgboost_final_model(x_train, y_train, x_val, y_val):

    xgboost_best_params = {
        'colsample_bytree': 0.7,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 500,
        'eval_metric': 'rmse'
        }

    xgboost_final = XGBRegressor(**xgboost_best_params).fit(
        x_train, y_train,
        eval_set = [(x_train, y_train), (x_val, y_val)],
        callbacks = [save_callback_xgboost]
    )
    joblib.dump(xgboost_final, os.path.join(model_path, 'xgboost.pkl'))

    return xgboost_final


def catboost_final_model(x_train, y_train, x_val, y_val):

    catboost_best_params = {
        'depth': 6,
        'iterations': 700,
        'learning_rate': 0.1,
        'task_type': 'CPU',
        'eval_metric' : 'RMSE'
        }
    
    catboost_final = CatBoostRegressor(**catboost_best_params).fit(
        x_train, y_train,
        eval_set = [(x_train, y_train), (x_val, y_val)],
        callbacks = [save_callback_catboost],
        logging_level='Silent'
    )
    joblib.dump(catboost_final, os.path.join(model_path, 'catboost.pkl'))

    return catboost_final


def ensemble_model(lightgbm, xgboost, catboost, x_train, y_train):
    
    voting_regressor = VotingRegressor(
        estimators=[
            ('lightgbm', lightgbm),
            ('xgboost', xgboost),
            ('catboost', catboost)
        ]
    )
    
    voting_regressor.fit(x_train, y_train)
    joblib.dump(voting_regressor, os.path.join(model_path, 'ensemble_model.pkl'))





train = pd.read_csv(os.path.join(data_path_input, "train.csv"))
validation = pd.read_csv(os.path.join(data_path_input, "validation.csv"))
test = pd.read_csv(os.path.join(data_path_input, "test.csv"))

X_train, y_train, X_val, y_val = train_val_test_split(train, validation)


lightgbm_model = lightgbm_final_model(X_train, y_train, X_val, y_val)
xgboost_model = xgboost_final_model(X_train, y_train, X_val, y_val)
catboost_model = catboost_final_model(X_train, y_train, X_val, y_val)

ensemble_model(lightgbm_model, xgboost_model, catboost_model, X_train, y_train)

