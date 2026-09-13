# Loan Default Prediction

An end-to-end machine learning project that predicts whether a borrower is likely to **default on a loan**. The project combines data cleaning, exploratory data analysis, feature engineering, model comparison, XGBoost hyperparameter tuning, validation-based threshold selection, and SHAP explainability.

## Business Problem

Loan default creates financial risk for lenders. A useful predictive system should identify high-risk applications while balancing the cost of false positives and false negatives.

**Objective:** predict `DEFAULT` vs. `NO DEFAULT` from borrower and loan characteristics.

> This is a portfolio project for demonstrating machine-learning and analytical skills. It is **not** a production lending model and should not be used as the sole basis for real credit decisions.

## Dataset

The raw dataset contains **32,586 observations and 13 columns** before cleaning. After duplicate removal, required-field filtering, data-quality treatment, and the loan-amount boundary applied in the notebook, the modeling dataset contains **32,569 observations**.

Target distribution in the cleaned dataset:

- `NO DEFAULT`: 25,739 (79.03%)
- `DEFAULT`: 6,830 (20.97%)

The dataset file used by the notebook is:

`data/LoanDataset - LoansDatasest.csv`

## Project Workflow

```text
Raw Data
   ↓
Data Quality Checks
   ↓
Cleaning & Missing-Value Treatment
   ↓
EDA
   ↓
Feature Engineering
   ↓
Train / Validation / Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Comparison
   ↓
XGBoost + RandomizedSearchCV
   ↓
Validation Threshold Selection
   ↓
Final Test Evaluation
   ↓
Feature Importance + SHAP
```

## Data Preparation

The notebook performs the following key steps:

- Standardizes column names.
- Removes duplicate rows.
- Converts formatted `loan_amnt` values such as `£35,000.00` to numeric values.
- Converts `customer_income` to numeric.
- Removes `customer_id` because it is an identifier.
- Drops records missing essential income, loan amount, or target values.
- Median-imputes missing `customer_age`, `employment_duration`, and `loan_int_rate`.
- Treats ages below 18 or above 100 as implausible observations.
- Treats employment durations above 50 years as implausible observations.
- Restricts `loan_amnt` to the observed portfolio boundary of £35,000 after inspecting the extreme values.
- Excludes `historical_default` from predictive modeling because its missing/unknown pattern is strongly associated with the target and presents a potential leakage/data-quality issue.

## Feature Engineering

The model uses both original variables and derived features:

### Numerical

- `customer_age`
- `customer_income`
- `employment_duration`
- `loan_amnt`
- `loan_int_rate`
- `term_years`
- `cred_hist_length`
- `loan_to_income`

### Categorical

- `home_ownership`
- `loan_intent`
- `loan_grade`
- `income_band`
- `age_group`
- `interest_rate_category`

The key engineered feature is:

**Loan-to-income ratio**

```text
loan_to_income = loan_amnt / customer_income
```

## Exploratory Findings

The EDA identified several useful associations:

- Lower loan grades showed substantially higher observed default rates.
- RENT borrowers had a higher observed default rate than OWN or MORTGAGE borrowers.
- Debt-consolidation loans showed one of the highest observed default rates among loan intents.
- Defaulted borrowers had lower median income than non-defaulted borrowers.
- Defaulted borrowers had higher median interest rates.
- Defaulted borrowers had a higher median loan-to-income ratio.
- Age and credit-history length showed a strong positive correlation in the numerical analysis.

These findings describe **associations in the dataset**, not causal relationships.

## Modeling

Four classifiers are evaluated:

1. Logistic Regression — linear baseline
2. Decision Tree — nonlinear interpretable baseline
3. Random Forest — bagging-based ensemble
4. XGBoost — gradient-boosted tree model

Preprocessing is implemented with a scikit-learn `ColumnTransformer` and `Pipeline`:

- Median imputation + standardization for numerical variables
- Most-frequent imputation + one-hot encoding for categorical variables

### Hyperparameter tuning

XGBoost is tuned with `RandomizedSearchCV` using:

- 20 random parameter combinations
- 3-fold cross-validation
- ROC-AUC as the optimization metric
- training subset only

Best configuration from the corrected workflow:

```text
n_estimators       = 500
max_depth          = 5
learning_rate      = 0.05
subsample          = 0.90
colsample_bytree   = 0.80
```

Best cross-validation ROC-AUC: **0.9414**

## Final Model Performance

The classification threshold was selected on the validation set by maximizing F1-score. The selected threshold was **0.45**.

The final model was then refit on the complete training set and evaluated once on the untouched test set.

| Metric | Test Score |
|---|---:|
| Accuracy | **93.51%** |
| Precision | **92.98%** |
| Recall | **74.67%** |
| F1-score | **82.83%** |
| ROC-AUC | **95.23%** |

Confusion matrix at threshold 0.45:

```text
                 Predicted
               No Default  Default
Actual No Default   5071       77
       Default       346     1020
```

The threshold was deliberately selected before final test evaluation. This is important because choosing a threshold after inspecting test performance would produce an optimistic estimate.

## Explainability

The project uses both model feature importance and SHAP analysis.

SHAP is used to answer questions such as:

- Which variables most influence default predictions?
- Does a high loan-to-income ratio increase predicted risk?
- How do income, interest rate, loan grade, home ownership, and loan intent affect predictions?

The analysis indicates that **loan grade, home ownership, loan intent, loan-to-income ratio, customer income, and interest rate** are among the strongest contributors to model predictions.

## Business Recommendations

Based on the analysis, a lending workflow could:

1. Give additional risk-review attention to applications with lower loan grades.
2. Incorporate loan-to-income ratio into affordability/risk assessment.
3. Use income and requested loan amount together rather than evaluating either in isolation.
4. Use loan intent and home ownership as supporting risk-segmentation variables.
5. Treat the model as decision support rather than an autonomous lending decision.
6. Tie the final probability threshold to the actual financial cost of false approvals and false rejections.

## Limitations & Future Work

### Limitations

- The target is imbalanced, so accuracy alone is insufficient.
- The source data contains anomalous observations and a suspicious missing-value pattern in `historical_default`.
- The selected threshold optimizes F1, but a real lender should optimize a business-specific cost function.
- The analysis is observational and does not establish causality.
- No external or time-based validation dataset is available.

### Future improvements

- Probability calibration and calibration curves
- Cost-sensitive threshold optimization
- Fairness/subgroup evaluation
- Time-based and external validation
- Model monitoring and drift detection
- Production API/Streamlit interface

## Repository Structure

```text
loan-default-prediction/
│
├── data/
│   └── LoanDataset - LoansDatasest.csv
│
├── models/
│   ├── loan_default_xgb_pipeline.joblib
│   └── metadata.json
│
├── notebooks/
│   ├── loan_default_prediction.ipynb
│   └── original_backup.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd loan-default-prediction
```

### 2. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook notebooks/loan_default_prediction.ipynb
```

Run the notebook from top to bottom.

## Skills Demonstrated

**Python · Pandas · NumPy · Matplotlib · Scikit-learn · XGBoost · SHAP · EDA · Feature Engineering · Classification · Hyperparameter Tuning · Model Evaluation · Explainable AI · Data Quality · Business Analytics**

## Author

**Raghav Ratan Yadav**

Integrated M.Sc. Mathematics | Data Analytics & Machine Learning
