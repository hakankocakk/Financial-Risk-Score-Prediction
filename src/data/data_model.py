from pydantic import BaseModel

class FinancialInformation(BaseModel):
    Age : int
    AnnualIncome : int
    CreditScore : int
    EmploymentStatus : object
    EducationLevel : object
    Experience : int
    LoanAmount : int
    LoanDuration : int
    MaritalStatus : object
    NumberOfDependents : int
    HomeOwnershipStatus : object
    MonthlyDebtPayments : int
    CreditCardUtilizationRate : float
    NumberOfOpenCreditLines : int
    NumberOfCreditInquiries : int
    DebtToIncomeRatio : float
    BankruptcyHistory : int
    LoanPurpose : object
    PreviousLoanDefaults : int
    PaymentHistory : int
    LengthOfCreditHistory : int
    SavingsAccountBalance : int
    CheckingAccountBalance : int
    TotalAssets : int
    TotalLiabilities : int
    MonthlyIncome : float
    UtilityBillsPaymentHistory : float
    JobTenure : int
    NetWorth : int
    BaseInterestRate : float
    InterestRate : float
    MonthlyLoanPayment : float
    TotalDebtToIncomeRatio : float