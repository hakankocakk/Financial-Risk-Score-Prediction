from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os
import sqlite3
from src.data.data_model import FinancialInformation
import src.features.data_preprocess as dp


app = FastAPI(
    title="Financial Risk Score Prediction",
    description="Predicting Financial Risk Score"
)


def save_to_db(
        FinancialInformation: FinancialInformation, predicted_score: float
):
    with sqlite3.connect("database/customers.db") as connect:
        cursor = connect.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                annual_income INTEGER,
                credit_score INTEGER,
                employment_status TEXT,
                education_level TEXT,
                experience INTEGER,
                loan_amount INTEGER,
                loan_duration INTEGER,
                marital_status TEXT,
                number_of_dependents INTEGER,
                home_ownership_status TEXT,
                monthly_debt_payments INTEGER,
                credit_card_utilization_rate REAL,
                number_of_open_credit_lines INTEGER,
                number_of_credit_inquiries INTEGER,
                debt_to_income_ratio REAL,
                bankruptcy_history INTEGER,
                loan_purpose TEXT,
                previous_loan_defaults INTEGER,
                payment_history INTEGER,
                length_of_credit_history INTEGER,
                savings_account_balance INTEGER,
                checking_account_balance INTEGER,
                total_assets INTEGER,
                total_liabilities INTEGER,
                monthly_income INTEGER,
                utility_bills_payment_history REAL,
                job_tenure INTEGER,
                net_worth INTEGER,
                base_interest_rate REAL,
                interest_rate REAL,
                monthly_loan_payment INTEGER,
                total_debt_to_income_ratio REAL,
                predicted_risk_score REAL
            )""")

        cursor.execute("""
            INSERT INTO predictions (
                age, annual_income, credit_score, employment_status,
                education_level, experience, loan_amount, loan_duration,
                marital_status, number_of_dependents, home_ownership_status,
                monthly_debt_payments, credit_card_utilization_rate,
                number_of_open_credit_lines, number_of_credit_inquiries,
                debt_to_income_ratio, bankruptcy_history, loan_purpose,
                previous_loan_defaults, payment_history,
                length_of_credit_history, savings_account_balance,
                checking_account_balance, total_assets,
                total_liabilities, monthly_income,
                utility_bills_payment_history, job_tenure, net_worth,
                base_interest_rate, interest_rate,
                monthly_loan_payment, total_debt_to_income_ratio,
                predicted_risk_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",

        (
            FinancialInformation.Age,
            FinancialInformation.AnnualIncome,
            FinancialInformation.CreditScore,
            FinancialInformation.EmploymentStatus,
            FinancialInformation.EducationLevel,
            FinancialInformation.Experience,
            FinancialInformation.LoanAmount,
            FinancialInformation.LoanDuration,
            FinancialInformation.MaritalStatus,
            FinancialInformation.NumberOfDependents,
            FinancialInformation.HomeOwnershipStatus,
            FinancialInformation.MonthlyDebtPayments,
            FinancialInformation.CreditCardUtilizationRate,
            FinancialInformation.NumberOfOpenCreditLines,
            FinancialInformation.NumberOfCreditInquiries,
            FinancialInformation.DebtToIncomeRatio,
            FinancialInformation.BankruptcyHistory,
            FinancialInformation.LoanPurpose,
            FinancialInformation.PreviousLoanDefaults,
            FinancialInformation.PaymentHistory,
            FinancialInformation.LengthOfCreditHistory,
            FinancialInformation.SavingsAccountBalance,
            FinancialInformation.CheckingAccountBalance,
            FinancialInformation.TotalAssets,
            FinancialInformation.TotalLiabilities,
            FinancialInformation.MonthlyIncome,
            FinancialInformation.UtilityBillsPaymentHistory,
            FinancialInformation.JobTenure,
            FinancialInformation.NetWorth,
            FinancialInformation.BaseInterestRate,
            FinancialInformation.InterestRate,
            FinancialInformation.MonthlyLoanPayment,
            FinancialInformation.TotalDebtToIncomeRatio,
            predicted_score
        ))
        connect.commit()


def data(
        FinancialInformation: FinancialInformation, model_path
):
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

    save_to_db(FinancialInformation, predicted_score[0])

    return np.round(predicted_score[0], 2)
