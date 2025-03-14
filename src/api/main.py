from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os
from src.data.data_model import FinancialInformation
import src.features.data_preprocess as dp


app = FastAPI(
    title="Financial Risk Score Prediction",
    description="Predicting Financial Risk Score"
)


def data(FinancialInformation: FinancialInformation, model_path):
    sample = pd.DataFrame({
        "Age": [FinancialInformation.Age],
        "AnnualIncome": [FinancialInformation.AnnualIncome],
        "CreditScore": [FinancialInformation.CreditScore],
        "EmploymentStatus": [FinancialInformation.EmploymentStatus],
        "EducationLevel": [FinancialInformation.EducationLevel],
        "Experience": [FinancialInformation.Experience],
        "LoanAmount": [FinancialInformation.LoanAmount],
        "LoanDuration": [FinancialInformation.LoanDuration],
        "MaritalStatus": [FinancialInformation.MaritalStatus],
        "NumberOfDependents": [FinancialInformation.NumberOfDependents],
        "HomeOwnershipStatus": [FinancialInformation.HomeOwnershipStatus],
        "MonthlyDebtPayments": [
            FinancialInformation.MonthlyDebtPayments
        ],
        "CreditCardUtilizationRate": [
            FinancialInformation.CreditCardUtilizationRate
        ],
        "NumberOfOpenCreditLines": [
            FinancialInformation.NumberOfOpenCreditLines
        ],
        "NumberOfCreditInquiries": [
            FinancialInformation.NumberOfCreditInquiries
        ],
        "DebtToIncomeRatio": [FinancialInformation.DebtToIncomeRatio],
        "BankruptcyHistory": [FinancialInformation.BankruptcyHistory],
        "LoanPurpose": [FinancialInformation.LoanPurpose],
        "PreviousLoanDefaults": [FinancialInformation.PreviousLoanDefaults],
        "PaymentHistory": [FinancialInformation.PaymentHistory],
        "LengthOfCreditHistory": [FinancialInformation.LengthOfCreditHistory],
        "SavingsAccountBalance": [FinancialInformation.SavingsAccountBalance],
        "CheckingAccountBalance": [
            FinancialInformation.CheckingAccountBalance
        ],
        "TotalAssets": [FinancialInformation.TotalAssets],
        "TotalLiabilities": [
            FinancialInformation.TotalLiabilities
        ],
        "MonthlyIncome": [FinancialInformation.MonthlyIncome],
        "UtilityBillsPaymentHistory": [
            FinancialInformation.UtilityBillsPaymentHistory
        ],
        "JobTenure": [FinancialInformation.JobTenure],
        "NetWorth": [FinancialInformation.NetWorth],
        "BaseInterestRate": [FinancialInformation.BaseInterestRate],
        "InterestRate": [FinancialInformation.InterestRate],
        "MonthlyLoanPayment": [FinancialInformation.MonthlyLoanPayment],
        "TotalDebtToIncomeRatio": [FinancialInformation.TotalDebtToIncomeRatio]
    })

    sample = dp.feature_engineering(sample)
    sample = dp.ordinalencoding(sample, model_path, train=False)
    sample = dp.onehotencoding(sample, model_path, train=False)
    sample = dp.normalization(sample, model_path, train=False)

    return sample


def model_load(path: str):
    ensemble_model = joblib.load(path)
    return ensemble_model


@app.get("/")
def index():
    return "Welcome to Financial Risk Score Prediction FastAPI"


@app.post("/prediction")
def model_predict(FinancialInformation: FinancialInformation):
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    sample = data(FinancialInformation, model_path)
    model = model_load(os.path.join(model_path, "ensemble_model.pkl"))
    predicted_score = model.predict(sample)
    return np.round(predicted_score[0], 2)
