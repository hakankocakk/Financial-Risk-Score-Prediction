import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



def data_colletion(dataframe):

    dataframe = dataframe.drop(["LoanApproved", "ApplicationDate"], axis=1)
    train, _ = train_test_split(dataframe, test_size=0.3, random_state=42)
    validation, test = train_test_split(_, test_size=0.5, random_state=42)

    return train, validation, test

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "financial_risk_scores.csv"))
train_data, validation_data, test_data = data_colletion(data)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
validation_data.to_csv(os.path.join(data_path, "validation.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
