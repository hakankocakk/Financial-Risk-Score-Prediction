import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split



def data_colletion(dataframe, yaml_path):

    size_params = yaml.safe_load(open(os.path.join(yaml_path, 'params.yaml'), 'r'))['data_collection']
 
    dataframe = dataframe.drop(["LoanApproved", "ApplicationDate"], axis=1)
    train, _ = train_test_split(dataframe, test_size=size_params["val_test_size"], random_state=42)
    validation, test = train_test_split(_, test_size=size_params["test_size"], random_state=42)

    return train, validation, test

yaml_path = os.path.join(os.path.dirname(__file__), "..")

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "financial_risk_scores.csv"))
train_data, validation_data, test_data = data_colletion(data, yaml_path)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
validation_data.to_csv(os.path.join(data_path, "validation.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
