import pytest
import pandas as pd
import os
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
