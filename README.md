# Explainable Credit Risk Modelling for Thin-File Borrowers
## Complete Reproducible Codebase — Garima Diyawar, MBA (Finance) Dissertation

---

## Project Structure

```
credit_risk_pipeline/
│
├── README.md                    ← This file
├── requirements.txt             ← All Python dependencies with versions
├── config.py                    ← Central configuration (paths, hyperparameters, flags)
│
├── data/                        ← Place Kaggle dataset files here (see below)
│
├── 01_data_loading.py           ← Load & join all 7 dataset files
├── 02_preprocessing.py          ← Cleaning, encoding, imputation, class imbalance
├── 03_feature_engineering.py    ← All engineered features from supplementary tables
├── 04_train_test_split.py       ← Stratified split + thin-file cohort partitioning
├── 05_model_training.py         ← LR, RF, XGBoost, LightGBM training & CV
├── 06_hyperparameter_tuning.py  ← Optuna TPE tuning for XGBoost & LightGBM
├── 07_explainability.py         ← SHAP (global + local + interaction) + LIME
├── 08_fairness_evaluation.py    ← Demographic parity, equalised odds, bootstrap CI
├── 09_results_reporting.py      ← All tables, figures, and summary outputs
│
├── run_full_pipeline.py         ← Master script: runs all modules in sequence
│
├── outputs/                     ← Model outputs, serialised models
├── models/                      ← Saved model objects (.joblib)
├── plots/                       ← All SHAP and evaluation plots (.png)
└── reports/                     ← CSV/Excel summary tables
```

---

## Dataset Download

1. Go to: https://www.kaggle.com/c/home-credit-default-risk/data
2. Download all files and place them in the `data/` folder:
   - application_train.csv
   - bureau.csv
   - bureau_balance.csv
   - previous_application.csv
   - POS_CASH_balance.csv
   - installments_payments.csv
   - credit_card_balance.csv

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset CSVs in data/ folder

# 3. Run full pipeline
python run_full_pipeline.py

# OR run modules individually in order:
python 01_data_loading.py
python 02_preprocessing.py
python 03_feature_engineering.py
python 04_train_test_split.py
python 05_model_training.py
python 06_hyperparameter_tuning.py   # Takes ~90 min on GPU
python 07_explainability.py
python 08_fairness_evaluation.py
python 09_results_reporting.py
```

---

## Hardware Requirements

- RAM: minimum 16 GB (32 GB recommended for bureau_balance.csv aggregation)
- Disk: ~5 GB free for dataset + intermediate files
- GPU: Optional but strongly recommended for Step 06 (Optuna tuning)
  - With GPU (CUDA): ~90 minutes total
  - CPU only: ~5-6 hours total

---

## Expected Key Outputs

| Metric                          | Expected Value |
|---------------------------------|----------------|
| XGBoost Full Dataset AUC        | ~0.78–0.80     |
| XGBoost Thin-File AUC           | ~0.74–0.78     |
| LR Bureau-Only Thin-File AUC    | ~0.58–0.62     |
| SHAP Stability (mean Spearman)  | ~0.92–0.95     |
| Worst-Case Demographic Parity   | ~0.15–0.20     |

Note: Exact values depend on random seed, Optuna trial outcomes, and dataset version.
All random seeds are fixed at 42 for reproducibility.
