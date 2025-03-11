import pytest
import pandas as pd
import os
import numpy as np
import src.features.data_preprocess as dp


def test_load_data_valid_file(tmp_path):
    test_file = tmp_path / "test.csv"
    test_file.write_text("col1,col2\n1,2\n3,4")

    df = dp.load_data(str(test_file))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_feature_engineering_valid():

    data = {
        "AnnualIncome": [50000, 60000],
        "TotalAssets": [200000, 300000],
        "Experience": [5, 10],
        "LoanAmount": [20000, 25000],
        "NumberOfDependents": [2, 3],
        "TotalLiabilities": [50000, 60000],
        "MonthlyLoanPayment": [1000, 1200],
        "MonthlyIncome": [4000, 5000],
        "MonthlyDebtPayments": [500, 700]
    }
    df = pd.DataFrame(data)
    transformed_df = dp.feature_engineering(df)
    expected_columns = [
        "AnIncomeToAssetsRatio", "AnExperienceToAnIncomeRatio",
        "LoantoAnIncomeRatio", "DependetToAnIncomeRatio",
        "LoansToAssetsRatio", "LoanPaymentToIncomeRatio",
        "AnIncomeToDepts", "AssetsToLoan"
    ]

    for col in expected_columns:
        assert col in transformed_df.columns, f"{col} is missing!"

    assert isinstance(transformed_df, pd.DataFrame)


@pytest.fixture
def mock_model_path(tmp_path):
    return tmp_path


def test_ordinalencoding_train(mock_model_path):

    data = {
        "EmploymentStatus": ["Employed", "Self-Employed",
                             "Unemployed", "Employed"]
    }
    df1 = pd.DataFrame(data)

    transformed_df = dp.ordinalencoding(df1, str(mock_model_path), train=True)

    assert transformed_df["EmploymentStatus"].iloc[0] == 0
    assert transformed_df["EmploymentStatus"].iloc[1] == 1
    assert transformed_df["EmploymentStatus"].iloc[2] == 2

    assert os.path.exists(os.path.join(mock_model_path, 'ordinal_encoder.pkl'))


def test_ordinalencoding_test(mock_model_path):

    data = {
        "EmploymentStatus": ["Employed", "Self-Employed",
                             "Unemployed", "Employed"]
    }
    df_train = pd.DataFrame(data)
    df_test = pd.DataFrame(data)

    transformed_df_train = dp.ordinalencoding(
        df_train, str(mock_model_path), train=True
    )
    transformed_df_test = dp.ordinalencoding(
        df_test, str(mock_model_path), train=False
    )

    assert transformed_df_train["EmploymentStatus"].equals(
        transformed_df_test["EmploymentStatus"]
    )


def test_onehotencoding_train(mock_model_path):

    data = {
        "EducationLevel": ["Master", "Associate", "Bachelor", "High School"],
        "MaritalStatus": ["Married", "Single", "Married", "Single"],
        "HomeOwnershipStatus": ["Rent", "Own", "Mortgage", "Rent"],
        "LoanPurpose": ["Auto", "Debt Consolidation", "Home", "Other"],
        "NumberOfDependents": [0, 2, 1, 3]
    }
    sample_dataframe = pd.DataFrame(data)

    transformed_df = dp.onehotencoding(
        sample_dataframe, mock_model_path, train=True
    )

    assert "EducationLevel" not in transformed_df.columns
    assert "MaritalStatus" not in transformed_df.columns
    assert "HomeOwnershipStatus" not in transformed_df.columns
    assert "LoanPurpose" not in transformed_df.columns
    assert "NumberOfDependents" not in transformed_df.columns

    assert any(col.startswith("EducationLevel_")
               for col in transformed_df.columns)
    assert any(col.startswith("MaritalStatus_")
               for col in transformed_df.columns)
    assert any(col.startswith("HomeOwnershipStatus_")
               for col in transformed_df.columns)
    assert any(col.startswith("LoanPurpose_")
               for col in transformed_df.columns)
    assert any(col.startswith("NumberOfDependents_")
               for col in transformed_df.columns)

    model_file = os.path.join(mock_model_path, "one_hot_encoder.pkl")
    assert os.path.exists(model_file)


def test_onehotencoding_test(mock_model_path):

    data = {
        "EducationLevel": ["Master", "Associate", "Bachelor", "High School"],
        "MaritalStatus": ["Married", "Single", "Married", "Single"],
        "HomeOwnershipStatus": ["Rent", "Own", "Mortgage", "Rent"],
        "LoanPurpose": ["Auto", "Debt Consolidation", "Home", "Other"],
        "NumberOfDependents": [0, 2, 1, 3]
    }
    sample_dataframe = pd.DataFrame(data)

    _ = dp.onehotencoding(
        sample_dataframe, str(mock_model_path), train=True
    )

    transformed_df_test = dp.onehotencoding(
        sample_dataframe, str(mock_model_path), train=False
    )

    assert "EducationLevel" not in transformed_df_test.columns
    assert "MaritalStatus" not in transformed_df_test.columns
    assert "HomeOwnershipStatus" not in transformed_df_test.columns
    assert "LoanPurpose" not in transformed_df_test.columns
    assert "NumberOfDependents" not in transformed_df_test.columns

    assert any(col.startswith("EducationLevel_")
               for col in transformed_df_test.columns)
    assert any(col.startswith("MaritalStatus_")
               for col in transformed_df_test.columns)
    assert any(col.startswith("HomeOwnershipStatus_")
               for col in transformed_df_test.columns)
    assert any(col.startswith("LoanPurpose_")
               for col in transformed_df_test.columns)
    assert any(col.startswith("NumberOfDependents_")
               for col in transformed_df_test.columns)


def test_normalization_train(mock_model_path):

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

    data = np.random.rand(10, len(num_cols)) * 100
    df_train = pd.DataFrame(data, columns=num_cols)

    df_normalized = dp.normalization(
        df_train, str(mock_model_path), train=True
    )

    model_file = os.path.join(mock_model_path, 'standardscaler.pkl')
    assert os.path.exists(model_file)

    assert np.all(
        np.isclose(df_normalized.mean(), 0, atol=1e-1)
    )
    assert np.all(
        np.isclose(df_normalized.std(), 1, atol=1e-1)
    )


def test_normalization_test(mock_model_path):

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

    data = np.random.rand(10, len(num_cols)) * 100
    df_train = pd.DataFrame(data, columns=num_cols)
    df_test = pd.DataFrame(data, columns=num_cols)

    df_train_normalized = dp.normalization(
        df_train, str(mock_model_path), train=True
    )
    df_test_normalized = dp.normalization(
        df_test, (mock_model_path), train=False
    )

    np.testing.assert_array_almost_equal(
        df_train_normalized.values, df_test_normalized, decimal=5
    )
