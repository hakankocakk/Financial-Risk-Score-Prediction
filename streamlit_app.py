import streamlit as st
import requests
import json


def main():
    st.title("Financial Risk Score Prediction")
    st.write(
        "Please fill in the information below and press the guess button."
    )

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=None
    )
    annual_income = st.number_input(
        "Annual Income", min_value=0, value=None
    )
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=None
    )
    employment_status = st.selectbox(
        "Employment Status", [
            "Employed", "Unemployed", "Self-Employed",
        ]
    )
    education_level = st.selectbox(
        "Education Level", [
            "High School", "Associate", "Bachelor", "Master", "Doctorate"
        ]
    )
    experience = st.number_input("Experience (Year)", min_value=0, value=None)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=None)
    loan_duration = st.number_input(
        "LoanDuration (Month)", min_value=1, value=None
    )
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Widowed"]
    )
    number_of_dependents = st.number_input(
        "Number Of Dependents", min_value=0, value=None
    )
    home_ownership_status = st.selectbox(
        "Home Ownership Status", ["Own", "Rent", "Mortgage", "Other"]
    )
    monthly_debt_payments = st.number_input(
        "Monthly Debt Payments", min_value=0, value=None
    )
    credit_card_utilization_rate = st.slider(
        "Credit Card Utilization Rate", 0.0, 1.0, None
    )
    number_of_open_credit_lines = st.number_input(
        "Number Of Open Credit Lines", min_value=0, value=None
    )
    number_of_credit_inquiries = st.number_input(
        "Number Of Credit Inquiries", min_value=0, value=None
    )
    debt_to_income_ratio = st.slider("Debt To Income Ratio", 0.0, 1.0, None)
    bankruptcy_history = st.selectbox("Bankruptcy History", [0, 1])
    loan_purpose = st.selectbox(
        "Loan Purpose", [
            "Debt Consolidation", "Home",
            "Education", "Auto", "Other"
        ]
    )
    previous_loan_defaults = st.selectbox(
        "Previous Loan Defaults", [0, 1]
    )
    payment_history = st.number_input(
        "Payment History", min_value=0, value=None
    )
    length_of_credit_history = st.number_input(
        "Length Of Credit History (Year)", min_value=0, value=None
    )
    savings_account_balance = st.number_input(
        "Savings Account Balance", min_value=0, value=None
    )
    checking_account_balance = st.number_input(
        "Checking Account Balance", min_value=0, value=None
    )
    total_assets = st.number_input(
        "Total Assets", min_value=0, value=None
    )
    total_liabilities = st.number_input(
        "Total Liabilities", min_value=0, value=None
    )
    monthly_income = st.number_input("Monthly Income", min_value=0, value=None)
    utility_bills_payment_history = st.slider(
        "Utility Bills Payment History", 0.0, 1.0, None
    )
    job_tenure = st.number_input("Job Tenure (Year)", min_value=0, value=None)
    net_worth = st.number_input("Net Worthr", value=None)
    base_interest_rate = st.slider("Base Interest Rate", 0.0, 1.0, None)
    interest_rate = st.slider("InterestRate", 0.0, 1.0, None)
    monthly_loan_payment = st.number_input(
        "Monthly Loan Payment", min_value=0, value=None
    )
    total_debt_to_income_ratio = st.slider(
        "Total Debt To Income Ratio", 0.0, 1.0, None
    )

    if st.button("Tahmin Yap"):
        input_data = {
            "Age": age,
            "AnnualIncome": annual_income,
            "CreditScore": credit_score,
            "EmploymentStatus": employment_status,
            "EducationLevel": education_level,
            "Experience": experience,
            "LoanAmount": loan_amount,
            "LoanDuration": loan_duration,
            "MaritalStatus": marital_status,
            "NumberOfDependents": number_of_dependents,
            "HomeOwnershipStatus": home_ownership_status,
            "MonthlyDebtPayments": monthly_debt_payments,
            "CreditCardUtilizationRate": credit_card_utilization_rate,
            "NumberOfOpenCreditLines": number_of_open_credit_lines,
            "NumberOfCreditInquiries": number_of_credit_inquiries,
            "DebtToIncomeRatio": debt_to_income_ratio,
            "BankruptcyHistory": bankruptcy_history,
            "LoanPurpose": loan_purpose,
            "PreviousLoanDefaults": previous_loan_defaults,
            "PaymentHistory": payment_history,
            "LengthOfCreditHistory": length_of_credit_history,
            "SavingsAccountBalance": savings_account_balance,
            "CheckingAccountBalance": checking_account_balance,
            "TotalAssets": total_assets,
            "TotalLiabilities": total_liabilities,
            "MonthlyIncome": monthly_income,
            "UtilityBillsPaymentHistory": utility_bills_payment_history,
            "JobTenure": job_tenure,
            "NetWorth": net_worth,
            "BaseInterestRate": base_interest_rate,
            "InterestRate": interest_rate,
            "MonthlyLoanPayment": monthly_loan_payment,
            "TotalDebtToIncomeRatio": total_debt_to_income_ratio
        }

        response = requests.post(
            "http://127.0.0.1:8000/prediction", data=json.dumps(input_data)
        )
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Estimated Financial Risk Score: {prediction}")
        else:
            st.error("The prediction was not possible. Please try again.")


if __name__ == "__main__":
    main()
