# Financial-Risk-Score-Prediction
A Financial Risk Score is a numerical representation of a customer's potential risk to a lender. It helps banks, financial institutions, and credit agencies assess the likelihood that a customer will default on a loan or default on their financial obligations. 

### The following processes were followed in this project:

#### 1) Data Collection:
Data was taken from Kaggle

#### 2) Data Preprocessing:
- Determination of categorical and numerical features
- Feature engineering
- Digitization of categorical data (Ordinal Encoding and One-Hot Encoding)
- Normalization (Standard Scaler)

#### 3) Model Selection and Training:

- XGBoost hyperparameter optimization (Just added parameters)
- LightGBM hyperparameter optimization (Just added parameters)
- CatBoost hyperparameter optimization (Just added parameters)
- Establishing an ensemble model (VotingRegressor)

#### 4) Model Evaluation:
- Performance analysis with metrics such as "Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error", "R2 Score".

#### 5) API:
- Creating an API with fastapi framework

## Installation

Clone the repository:
```bash
git clone https://github.com/hakankocakk/Financial-Risk-Score-Prediction.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Remove git tracking from src/model folder:
```bash
git rm -r --cached 'src/model'
```

Run the pipeline:
```bash
dvc repro
```

Show model performance
```bash
dvc metrics show
```
