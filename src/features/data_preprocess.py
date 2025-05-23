import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import joblib


def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path} : {e}")


def create_cols_types(dataframe: pd.DataFrame, threshold_cat=8):
    try:
        cat_cols = [col for col in dataframe.columns
                    if dataframe[col].dtypes == "O"]
        num_cols = [col for col in dataframe.columns
                    if dataframe[col].dtypes != "O"]
        num_but_cat_cols = [col for col in num_cols
                            if dataframe[col].nunique() < threshold_cat]
        cat_cols += num_but_cat_cols
        num_cols = [col for col in num_cols
                    if col not in num_but_cat_cols
                    and col not in ["RiskScore"]]
        return cat_cols, num_cols
    except KeyError as e:
        raise Exception(f"Dataframe not found : {e}")


def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    try:
        dataframe["AnIncomeToAssetsRatio"] = (
            dataframe["AnnualIncome"] / dataframe["TotalAssets"]
        )
        dataframe["AnExperienceToAnIncomeRatio"] = (
            dataframe["Experience"] / dataframe["AnnualIncome"]
        )
        dataframe["LoantoAnIncomeRatio"] = (
            dataframe["LoanAmount"] / dataframe["AnnualIncome"]
        )
        dataframe["DependetToAnIncomeRatio"] = (
            dataframe["AnnualIncome"] / (dataframe["NumberOfDependents"] + 1)
        )
        dataframe["LoansToAssetsRatio"] = (
            dataframe["TotalLiabilities"] / dataframe["TotalAssets"]
        )
        dataframe["LoanPaymentToIncomeRatio"] = (
            dataframe["MonthlyLoanPayment"] / dataframe["MonthlyIncome"]
        )
        dataframe["AnIncomeToDepts"] = (
            dataframe["AnnualIncome"] / (
                dataframe["MonthlyLoanPayment"]*12 + (
                    dataframe["MonthlyDebtPayments"]*12
                )
            )
        )
        dataframe["AssetsToLoan"] = (
            dataframe["TotalAssets"] / (
                dataframe["TotalLiabilities"] + dataframe["LoanAmount"]
            )
        )
        return dataframe
    except KeyError as e:
        raise Exception(f"Feature not found: {e}")


def ordinalencoding(
        dataframe: pd.DataFrame, model_path: str, train=True
) -> pd.DataFrame:

    Employment = ['Employed', 'Self-Employed', 'Unemployed']
    columns_to_encode = ["EmploymentStatus"]

    if train:
        try:
            enc = OrdinalEncoder(
                categories=[Employment],
                handle_unknown='use_encoded_value', unknown_value=-1
            )
            dataframe[columns_to_encode] = enc.fit_transform(
                dataframe[columns_to_encode]
            )
            joblib.dump(enc, os.path.join(model_path, 'ordinal_encoder.pkl'))
            return dataframe
        except KeyError as e:
            raise Exception(f"Feature not found: {e}")
        except Exception as e:
            raise Exception(f"Error save model file {e}")
    else:
        try:
            loaded_encoder = joblib.load(
                os.path.join(model_path, 'ordinal_encoder.pkl')
            )
            dataframe[columns_to_encode] = loaded_encoder.transform(
                dataframe[columns_to_encode]
            )
            return dataframe
        except KeyError as e:
            raise Exception(f"Feature not found: {e}")
        except Exception as e:
            raise Exception(f"Error loading model file {e}")


def onehotencoding(
        dataframe: pd.DataFrame, model_path: str, train=True
) -> pd.DataFrame:

    one_hot_cat_cols = ['EducationLevel', 'MaritalStatus',
                        'HomeOwnershipStatus', 'LoanPurpose',
                        'NumberOfDependents']

    if train:
        try:
            ohe = OneHotEncoder(
                handle_unknown='ignore', sparse_output=False, drop='first'
            )
            encoded_cols = ohe.fit_transform(dataframe[one_hot_cat_cols])
            joblib.dump(ohe, os.path.join(model_path, 'one_hot_encoder.pkl'))
            new_columns = ohe.get_feature_names_out(one_hot_cat_cols)
            encoded_df = pd.DataFrame(
                encoded_cols, columns=new_columns, index=dataframe.index
            )
            dataframe = pd.concat([dataframe, encoded_df], axis=1)
            dataframe.drop(columns=one_hot_cat_cols, inplace=True)
            return dataframe
        except KeyError as e:
            raise Exception(f"Feature not found: {e}")
        except Exception as e:
            raise Exception(f"Error save model file {e}")
    else:
        try:
            loaded_ohe = joblib.load(
                os.path.join(model_path, 'one_hot_encoder.pkl')
            )
            encoded_test_data = loaded_ohe.transform(
                dataframe[one_hot_cat_cols]
            )
            new_columns = loaded_ohe.get_feature_names_out(one_hot_cat_cols)
            encoded_test_df = pd.DataFrame(
                encoded_test_data, columns=new_columns, index=dataframe.index
            )
            dataframe = pd.concat([dataframe, encoded_test_df], axis=1)
            dataframe.drop(columns=one_hot_cat_cols, inplace=True)
            return dataframe
        except KeyError as e:
            raise Exception(f"Feature not found: {e}")
        except Exception as e:
            raise Exception(f"Error load model file {e}")


def normalization(
        dataframe: pd.DataFrame, model_path: str, train=True
) -> pd.DataFrame:

    num_cols = ['Age', 'AnnualIncome', 'CreditScore',
                'Experience', 'LoanAmount', 'LoanDuration',
                'MonthlyDebtPayments', 'CreditCardUtilizationRate',
                'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
                'DebtToIncomeRatio', 'PaymentHistory', 'LengthOfCreditHistory',
                'SavingsAccountBalance', 'CheckingAccountBalance',
                'TotalAssets', 'TotalLiabilities', 'MonthlyIncome',
                'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth',
                'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment',
                'TotalDebtToIncomeRatio', 'AnIncomeToAssetsRatio',
                'AnExperienceToAnIncomeRatio', 'LoantoAnIncomeRatio',
                'DependetToAnIncomeRatio', 'LoansToAssetsRatio',
                'LoanPaymentToIncomeRatio', 'AnIncomeToDepts', 'AssetsToLoan']

    if train:
        try:
            scaler = StandardScaler()
            dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
            joblib.dump(scaler, os.path.join(model_path, 'standardscaler.pkl'))
            return dataframe
        except KeyError as e:
            raise Exception(f"Feature not found: {e}")
        except Exception as e:
            raise Exception(f"Error save model file {e}")
    else:
        try:
            loaded_scaler = joblib.load(
                os.path.join(model_path, 'standardscaler.pkl')
            )
            dataframe[num_cols] = loaded_scaler.transform(dataframe[num_cols])
            return dataframe
        except KeyError as e:
            raise Exception(f"Feature not found: {e}")
        except Exception as e:
            raise Exception(f"Error load model file {e}")


def save_data(data: pd.DataFrame,  data_path: str) -> None:
    try:
        data.to_csv(data_path, index=False)
    except KeyError as e:
        raise Exception(f"Dataframe not found : {e}")
    except Exception as e:
        raise Exception(f"Error save data file {e}")


def main():
    interim_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "interim"
    )
    processed_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "datas", "processed"
    )
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models"
    )
    os.makedirs(processed_data_path, exist_ok=True)

    try:
        train = load_data(
            os.path.join(interim_data_path, "train.csv")
        )
        validation = load_data(
            os.path.join(interim_data_path, "validation.csv")
        )
        test = load_data(
            os.path.join(interim_data_path, "test.csv")
        )

        train = feature_engineering(train)
        validation = feature_engineering(validation)
        test = feature_engineering(test)

        train = ordinalencoding(train, model_path, train=True)
        validation = ordinalencoding(validation, model_path, train=False)
        test = ordinalencoding(test, model_path, train=False)

        train = onehotencoding(train, model_path, train=True)
        validation = onehotencoding(validation, model_path, train=False)
        test = onehotencoding(test, model_path, train=False)

        train_data = normalization(train, model_path, train=True)
        validation_data = normalization(validation, model_path, train=False)
        test_data = normalization(test, model_path, train=False)

        save_data(train_data,
                  os.path.join(processed_data_path, "train.csv"))
        save_data(validation_data,
                  os.path.join(processed_data_path, "validation.csv"))
        save_data(test_data,
                  os.path.join(processed_data_path, "test.csv"))

    except Exception as e:
        raise Exception(f"An error occured: {e}")


if __name__ == "__main__":
    main()
