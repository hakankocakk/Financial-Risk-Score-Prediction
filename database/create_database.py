import sqlite3


def main():
    """SQLite veritabanını oluşturur ve tabloyu hazırlar."""
    with sqlite3.connect("database/customers.db") as baglanti:
        imlec = baglanti.cursor()
        imlec.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
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
            )"""
        )
        baglanti.commit()


if __name__ == "__main__":
    main()
