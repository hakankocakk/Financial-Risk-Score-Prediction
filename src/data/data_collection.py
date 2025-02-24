import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split


def load_data(path : str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path} : {e}")


def load_yaml(yaml_path : str) -> dict:
    try:
        with open(os.path.join(yaml_path, 'params.yaml'), 'r') as file:
            size_params = yaml.safe_load(file)['data_collection']
            return size_params
    except Exception as e:
        raise Exception(f"Error loading paramaters from {yaml_path} : {e}")
    

def data_split(dataframe: pd.DataFrame, size_params: dict):
    try:
        dataframe = dataframe.drop(["LoanApproved", "ApplicationDate"], axis=1)
    except KeyError as e:
        raise Exception(f"Feature not found: {e}")
    except Exception as e:
        raise Exception(f"Error while dropping features: {e}")
    
    try:
        train, temp = train_test_split(dataframe, test_size=size_params["val_test_size"], random_state=42)
        validation, test = train_test_split(temp, test_size=size_params["test_size"], random_state=42)
        return train, validation, test
    except KeyError as e:
        raise Exception(f"Parameter key error: {e}")
    except Exception as e:
        raise Exception(f"Error during splitting: {e}")
    

def save_data(data: pd.DataFrame,  data_path : str) -> None:
    try:
        data.to_csv(data_path, index=False)
    except KeyError as e:
        raise Exception(f"Dataframe not found : {e}")
    except Exception as e:
        raise Exception(f"Error save data file")


def main():

    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..")
    raw_data_path = os.path.join(os.path.dirname(__file__), "..", "..","data", "raw", "financial_risk_scores.csv")
    interim_data_path = os.path.join(os.path.dirname(__file__), "..", "..","data", "interim")

    try:
        size_params = load_yaml(yaml_path)
        data = load_data(raw_data_path)

        train_data, validation_data, test_data = data_split(data, size_params)

        save_data(train_data, os.path.join(interim_data_path, "train.csv"))
        save_data(validation_data, os.path.join(interim_data_path, "validation.csv"))
        save_data(test_data, os.path.join(interim_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occured: {e}")


if __name__ == "__main__":
    main()