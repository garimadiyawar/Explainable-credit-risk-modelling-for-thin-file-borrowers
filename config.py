"""
config.py
Central configuration for the credit risk modelling pipeline.
All paths, constants, feature lists, and flags are defined here.
Edit this file to change behaviour across the entire pipeline.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset files ─────────────────────────────────────────────────────────────
DATA_FILES = {
    "application":    os.path.join(DATA_DIR, "application_train.csv"),
    "bureau":         os.path.join(DATA_DIR, "bureau.csv"),
    "bureau_balance": os.path.join(DATA_DIR, "bureau_balance.csv"),
    "previous":       os.path.join(DATA_DIR, "previous_application.csv"),
    "pos_cash":       os.path.join(DATA_DIR, "POS_CASH_balance.csv"),
    "installments":   os.path.join(DATA_DIR, "installments_payments.csv"),
    "credit_card":    os.path.join(DATA_DIR, "credit_card_balance.csv"),
}

# ── Intermediate artifact paths ───────────────────────────────────────────────
ARTIFACTS = {
    "raw_joined":          os.path.join(OUTPUT_DIR, "01_raw_joined.parquet"),
    "preprocessed":        os.path.join(OUTPUT_DIR, "02_preprocessed.parquet"),
    "featured":            os.path.join(OUTPUT_DIR, "03_featured.parquet"),
    "train_test":          os.path.join(OUTPUT_DIR, "04_train_test.npz"),
    "feature_names":       os.path.join(OUTPUT_DIR, "04_feature_names.json"),
    "thin_file_mask_test": os.path.join(OUTPUT_DIR, "04_thin_file_mask_test.npy"),
    "cv_results":          os.path.join(OUTPUT_DIR, "05_cv_results.json"),
    "best_params_xgb":     os.path.join(OUTPUT_DIR, "06_best_params_xgb.json"),
    "best_params_lgb":     os.path.join(OUTPUT_DIR, "06_best_params_lgb.json"),
    "shap_values":         os.path.join(OUTPUT_DIR, "07_shap_values.npy"),
    "shap_expected":       os.path.join(OUTPUT_DIR, "07_shap_expected_value.npy"),
    "shap_importance":     os.path.join(OUTPUT_DIR, "07_shap_importance.csv"),
    "fairness_results":    os.path.join(OUTPUT_DIR, "08_fairness_results.json"),
    "final_summary":       os.path.join(REPORT_DIR,  "final_summary.xlsx"),
}

# ── Model artifact paths ──────────────────────────────────────────────────────
MODELS = {
    "lr_bureau":   os.path.join(MODEL_DIR, "lr_bureau_only.joblib"),
    "lr_combined": os.path.join(MODEL_DIR, "lr_combined.joblib"),
    "rf_bureau":   os.path.join(MODEL_DIR, "rf_bureau_only.joblib"),
    "rf_combined": os.path.join(MODEL_DIR, "rf_combined.joblib"),
    "xgb_bureau":  os.path.join(MODEL_DIR, "xgb_bureau_only.joblib"),
    "xgb_combined":os.path.join(MODEL_DIR, "xgb_combined.joblib"),
    "lgb_bureau":  os.path.join(MODEL_DIR, "lgb_bureau_only.joblib"),
    "lgb_combined":os.path.join(MODEL_DIR, "lgb_combined.joblib"),
    "scaler":      os.path.join(MODEL_DIR, "standard_scaler.joblib"),
}

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Target variable ───────────────────────────────────────────────────────────
TARGET_COL  = "TARGET"
ID_COL      = "SK_ID_CURR"

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE   = 0.20          # 20% held-out test set
CV_FOLDS    = 5             # Stratified K-fold

# ── Class imbalance ───────────────────────────────────────────────────────────
# Computed at runtime: n_negative / n_positive ≈ 11.3
# Set to None to compute automatically
SCALE_POS_WEIGHT = None

# ── Thin-file definition ──────────────────────────────────────────────────────
# Applicants with zero bureau records are defined as thin-file
THIN_FILE_COL = "THIN_FILE_FLAG"

# ── Missingness thresholds ────────────────────────────────────────────────────
MISSING_LOW_THRESHOLD      = 0.05   # < 5%  → mean/mode imputation only
MISSING_MODERATE_THRESHOLD = 0.40   # 5–40% → mean/mode + indicator flag
# > 40% → sentinel (-999) + indicator flag
SENTINEL_VALUE             = -999.0

# ── Feature set definitions ───────────────────────────────────────────────────
# Bureau-only features (traditional): columns from application + bureau tables
# These are defined dynamically in 04_train_test_split.py
# but the prefix list helps distinguish categories:
BUREAU_FEATURE_PREFIXES = [
    "BUREAU_", "BB_", "EXT_SOURCE",
    "AMT_REQ_CREDIT_BUREAU",
]

ALTERNATIVE_FEATURE_PREFIXES = [
    "INST_", "CC_", "POS_", "PREV_",
]

# ── Feature engineering: ratio features from application table ────────────────
APP_RATIO_FEATURES = {
    "CREDIT_INCOME_RATIO":   ("AMT_CREDIT",      "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO":  ("AMT_ANNUITY",      "AMT_INCOME_TOTAL"),
    "CREDIT_TERM":           ("AMT_CREDIT",       "AMT_ANNUITY"),
    "GOODS_PRICE_RATIO":     ("AMT_GOODS_PRICE",  "AMT_CREDIT"),
    "INCOME_PER_PERSON":     ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    "CHILDREN_RATIO":        ("CNT_CHILDREN",     "CNT_FAM_MEMBERS"),
}

# ── Categorical columns to one-hot encode ─────────────────────────────────────
OHE_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

# ── Binary columns to label encode ───────────────────────────────────────────
BINARY_COLS = [
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

# ── Log-transform columns (right-skewed financial amounts) ────────────────────
LOG_TRANSFORM_COLS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

# ── Fairness / demographic proxy columns ──────────────────────────────────────
FAIRNESS_COLS = [
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
]

# ── Hyperparameter Optuna settings ────────────────────────────────────────────
OPTUNA_N_TRIALS      = 200
OPTUNA_TIMEOUT       = None          # seconds; None = run all trials
OPTUNA_DIRECTION     = "maximize"    # maximize AUC

# XGBoost search space bounds
XGB_SEARCH_SPACE = {
    "n_estimators":     (100,  3000),
    "learning_rate":    (0.005, 0.3),
    "max_depth":        (3, 10),
    "min_child_weight": (1, 300),
    "subsample":        (0.4, 1.0),
    "colsample_bytree": (0.4, 1.0),
    "gamma":            (0.0, 5.0),
    "reg_lambda":       (0.1, 100.0),
    "reg_alpha":        (1e-8, 10.0),
}

# LightGBM search space bounds
LGB_SEARCH_SPACE = {
    "n_estimators":     (100, 3000),
    "learning_rate":    (0.005, 0.3),
    "num_leaves":       (20, 300),
    "max_depth":        (3, 12),
    "min_child_samples":(5, 200),
    "subsample":        (0.4, 1.0),
    "colsample_bytree": (0.4, 1.0),
    "reg_lambda":       (0.1, 100.0),
    "reg_alpha":        (1e-8, 10.0),
}

# ── SHAP settings ─────────────────────────────────────────────────────────────
SHAP_MAX_DISPLAY      = 20        # features in beeswarm plot
SHAP_N_LIME_SAMPLES   = 5000      # perturbation samples for LIME
SHAP_TOP_INTERACTION  = 10        # features for interaction value analysis
ADVERSE_ACTION_TOP_N  = 3         # top N reasons for adverse action notice

# ── Evaluation settings ───────────────────────────────────────────────────────
BOOTSTRAP_N           = 1000      # bootstrap iterations for fairness CI
FAIRNESS_ALPHA        = 0.05      # significance level

# ── Verbosity ─────────────────────────────────────────────────────────────────
VERBOSE = True
