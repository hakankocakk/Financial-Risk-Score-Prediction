import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import joblib


data_path_input = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
data_path_output = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
model_path = os.path.join(os.path.dirname(__file__), "model", "process_model")

os.makedirs(data_path_output)
os.makedirs(model_path)


def create_cols_types(dataframe, threshold_cat=8):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_but_cat_cols = [col for col in num_cols if dataframe[col].nunique() < threshold_cat]
    cat_cols += num_but_cat_cols
    num_cols = [col for col in num_cols if col not in num_but_cat_cols and col not in ["RiskScore"]]

    return cat_cols, num_cols


def feature_engineering(dataframe):

    dataframe["AnIncomeToAssetsRatio"] = dataframe["AnnualIncome"] / dataframe["TotalAssets"]
    dataframe["AnExperienceToAnIncomeRatio"] = dataframe["Experience"] / dataframe["AnnualIncome"]
    dataframe["LoantoAnIncomeRatio"] = dataframe["LoanAmount"] / dataframe["AnnualIncome"]
    dataframe["DependetToAnIncomeRatio"] = dataframe["AnnualIncome"] / (dataframe["NumberOfDependents"] + 1)
    dataframe["LoansToAssetsRatio"] = dataframe["TotalLiabilities"] / dataframe["TotalAssets"]
    dataframe["LoanPaymentToIncomeRatio"] = dataframe["MonthlyLoanPayment"] / dataframe["MonthlyIncome"]
    dataframe["AnIncomeToDepts"] = dataframe["AnnualIncome"] / (dataframe["MonthlyLoanPayment"]*12 + dataframe["MonthlyDebtPayments"]*12)
    dataframe["AssetsToLoan"] = dataframe["TotalAssets"] / (dataframe["TotalLiabilities"] + dataframe["LoanAmount"])

    return dataframe


def ordinalencoding(dataframe, train=True):

    Employment = ['Employed', 'Self-Employed', 'Unemployed']
    columns_to_encode = ["EmploymentStatus"]

    if train:
        enc = OrdinalEncoder(categories=[Employment])
        dataframe[columns_to_encode] = enc.fit_transform(dataframe[columns_to_encode])
        joblib.dump(enc, os.path.join(model_path, 'ordinal_encoder.pkl'))
    else:
        loaded_encoder = joblib.load(os.path.join(model_path, 'ordinal_encoder.pkl'))
        dataframe[columns_to_encode] = loaded_encoder.transform(dataframe[columns_to_encode])

    return dataframe


def onehotencoder(dataframe, train=True):

    one_hot_cat_cols = ['EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose', 'NumberOfDependents']

    if train:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        encoded_cols = ohe.fit_transform(dataframe[one_hot_cat_cols])
        joblib.dump(ohe, os.path.join(model_path, 'one_hot_encoder.pkl'))
        new_columns = ohe.get_feature_names_out(one_hot_cat_cols)
        encoded_df = pd.DataFrame(encoded_cols, columns=new_columns, index=dataframe.index)
        dataframe = pd.concat([dataframe, encoded_df], axis=1)
        dataframe.drop(columns=one_hot_cat_cols, inplace=True)
    else:
        loaded_ohe = joblib.load(os.path.join(model_path, 'one_hot_encoder.pkl'))
        encoded_test_data = loaded_ohe.transform(dataframe[one_hot_cat_cols])
        new_columns = loaded_ohe.get_feature_names_out(one_hot_cat_cols)
        encoded_test_df = pd.DataFrame(encoded_test_data, columns=new_columns, index=dataframe.index)
        dataframe = pd.concat([dataframe, encoded_test_df], axis=1)
        dataframe.drop(columns=one_hot_cat_cols, inplace=True)
    
    return dataframe


def normalization(dataframe, num_cols, train=True):

    if train:
        scaler = StandardScaler()
        dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
        joblib.dump(scaler, os.path.join(model_path, 'standardscaler.pkl'))
    else:
        loaded_scaler = joblib.load(os.path.join(model_path, 'standardscaler.pkl'))
        dataframe[num_cols] = loaded_scaler.transform(dataframe[num_cols])

    return dataframe




train = pd.read_csv(os.path.join(data_path_input, "train.csv"))
validation = pd.read_csv(os.path.join(data_path_input, "validation.csv"))
test = pd.read_csv(os.path.join(data_path_input, "test.csv"))

cat_cols, num_cols = create_cols_types(train)

train = feature_engineering(train)
validation = feature_engineering(validation)
test = feature_engineering(test)

train = ordinalencoding(train, train=True)
validation = ordinalencoding(validation, train=False)
test = ordinalencoding(test, train=False)

train = onehotencoder(train, train=True)
validation = onehotencoder(validation, train=False)
test = onehotencoder(test, train=False)

_, num_cols = create_cols_types(train)
num_cols = [col for col in num_cols if col not in ["RiskScore"]]
train_data = normalization(train, num_cols, train=True)
validation_data = normalization(validation, num_cols, train=False)
test_data = normalization(test, num_cols, train=False)


train_data.to_csv(os.path.join(data_path_output, "train.csv"), index=False)
validation_data.to_csv(os.path.join(data_path_output, "validation.csv"), index=False)
test_data.to_csv(os.path.join(data_path_output, "test.csv"), index=False)