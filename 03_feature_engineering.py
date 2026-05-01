"""
03_feature_engineering.py
--------------------------
Constructs all engineered features from:
  - Application table (ratio features, derived scores)
  - Bureau + Bureau Balance (credit history aggregations)
  - Instalment Payments (payment discipline features)
  - Credit Card Balance (utilisation and delinquency features)
  - POS Cash Balance (arrears and loan status features)
  - Previous Applications (outcome and approval rate features)

Input:  outputs/02_preprocessed.parquet  +  raw CSV files
Output: outputs/03_featured.parquet
"""

import os
import sys
import warnings
import numpy  as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_FILES, ARTIFACTS, TARGET_COL, ID_COL,
    THIN_FILE_COL, RANDOM_SEED, VERBOSE, APP_RATIO_FEATURES, SENTINEL_VALUE
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[03_feature_engineering] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Application table ratio features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_app_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standard ratios from config
    for new_col, (num_col, den_col) in APP_RATIO_FEATURES.items():
        # Use log-transformed versions where available
        n = f"LOG_{num_col}" if f"LOG_{num_col}" in df.columns else num_col
        d = f"LOG_{den_col}" if f"LOG_{den_col}" in df.columns else den_col
        if n in df.columns and d in df.columns:
            df[new_col] = df[n] / (df[d] + 1e-8)

    # Age in years (DAYS_BIRTH is negative)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"]      = df["DAYS_BIRTH"] / -365.0

    # Employment tenure in years
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"].clip(upper=0) / -365.0
        df["DAYS_EMPLOYED_RATIO"] = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1e-8)

    # EXT_SOURCE composite
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"]  = df[ext_cols].std(axis=1).fillna(0)
        df["EXT_SOURCE_MIN"]  = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"]  = df[ext_cols].max(axis=1)

    # Address mismatch count
    addr_cols = [c for c in df.columns if "REG_CITY_NOT" in c or "REG_REGION_NOT" in c]
    if addr_cols:
        df["REG_MISMATCH_COUNT"] = df[addr_cols].sum(axis=1)

    # Document completeness score
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOC_SCORE"] = df[doc_cols].sum(axis=1)

    # Credit term (months)
    if "LOG_AMT_CREDIT" in df.columns and "LOG_AMT_ANNUITY" in df.columns:
        df["CREDIT_TERM"] = np.expm1(df["LOG_AMT_CREDIT"]) / (np.expm1(df["LOG_AMT_ANNUITY"]) + 1e-8)

    log(f"  Application feature engineering complete. New features added.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bureau features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_bureau_features() -> pd.DataFrame:
    log("  Engineering bureau features ...")
    bureau = pd.read_csv(DATA_FILES["bureau"])

    agg = bureau.groupby(ID_COL).agg(
        BUREAU_LOAN_COUNT          = (ID_COL,                  "count"),
        BUREAU_ACTIVE_COUNT        = ("CREDIT_ACTIVE",          lambda x: (x == "Active").sum()),
        BUREAU_CLOSED_COUNT        = ("CREDIT_ACTIVE",          lambda x: (x == "Closed").sum()),
        BUREAU_AMT_CREDIT_SUM      = ("AMT_CREDIT_SUM",         "sum"),
        BUREAU_AMT_CREDIT_MEAN     = ("AMT_CREDIT_SUM",         "mean"),
        BUREAU_AMT_CREDIT_MAX      = ("AMT_CREDIT_SUM",         "max"),
        BUREAU_AMT_DEBT_SUM        = ("AMT_CREDIT_SUM_DEBT",    "sum"),
        BUREAU_DPD_MAX             = ("CREDIT_DAY_OVERDUE",     "max"),
        BUREAU_DPD_MEAN            = ("CREDIT_DAY_OVERDUE",     "mean"),
        BUREAU_OVERDUE_COUNT       = ("CREDIT_DAY_OVERDUE",     lambda x: (x > 0).sum()),
        BUREAU_ENDDATE_MEAN        = ("DAYS_CREDIT_ENDDATE",    "mean"),
        BUREAU_CREDIT_LIMIT_MEAN   = ("AMT_CREDIT_SUM_LIMIT",   "mean"),
        BUREAU_PROLONG_COUNT       = ("CNT_CREDIT_PROLONG",     "sum"),
        BUREAU_DAYS_CREDIT_MEAN    = ("DAYS_CREDIT",            "mean"),
        BUREAU_DAYS_CREDIT_MIN     = ("DAYS_CREDIT",            "min"),
    ).reset_index()

    # Debt ratio
    agg["BUREAU_DEBT_RATIO"] = (
        agg["BUREAU_AMT_DEBT_SUM"] /
        (agg["BUREAU_AMT_CREDIT_SUM"] + 1e-8)
    )

    # Active ratio
    agg["BUREAU_ACTIVE_RATIO"] = (
        agg["BUREAU_ACTIVE_COUNT"] /
        (agg["BUREAU_LOAN_COUNT"] + 1e-8)
    )

    log(f"  Bureau agg shape: {agg.shape}")
    return agg


def engineer_bureau_balance_features() -> pd.DataFrame:
    log("  Engineering bureau_balance features ...")
    bb = pd.read_csv(DATA_FILES["bureau_balance"])

    # Status: 0=no DPD, 1-5=increasing DPD bands, C=closed, X=unknown
    bb["DPD_FLAG"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.int8)
    bb["STATUS_NUM"] = pd.to_numeric(
        bb["STATUS"].replace({"C": 0, "X": np.nan}), errors="coerce"
    )

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_DPD_FRACTION  = ("DPD_FLAG",    "mean"),
        BB_DPD_COUNT     = ("DPD_FLAG",    "sum"),
        BB_MONTHS_COUNT  = ("MONTHS_BALANCE", "count"),
        BB_STATUS_MAX    = ("STATUS_NUM",  "max"),
        BB_STATUS_MEAN   = ("STATUS_NUM",  "mean"),
    ).reset_index()

    # Join to bureau to get SK_ID_CURR
    bureau = pd.read_csv(DATA_FILES["bureau"])[[ID_COL, "SK_ID_BUREAU"]]
    bb_agg = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bb_curr = bb_agg.groupby(ID_COL).agg(
        BB_DPD_FRAC_MEAN  = ("BB_DPD_FRACTION", "mean"),
        BB_DPD_FRAC_MAX   = ("BB_DPD_FRACTION", "max"),
        BB_DPD_COUNT_SUM  = ("BB_DPD_COUNT",    "sum"),
        BB_MONTHS_TOTAL   = ("BB_MONTHS_COUNT", "sum"),
        BB_STATUS_MAX     = ("BB_STATUS_MAX",   "max"),
        BB_STATUS_MEAN    = ("BB_STATUS_MEAN",  "mean"),
    ).reset_index()

    log(f"  Bureau balance agg shape: {bb_curr.shape}")
    return bb_curr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Instalment payments features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_installments_features() -> pd.DataFrame:
    log("  Engineering installments_payments features ...")
    inst = pd.read_csv(DATA_FILES["installments"])

    inst["PAYMENT_RATIO"]   = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1e-8)
    inst["DAYS_PAST_DUE"]   = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["DAYS_BEFORE_DUE"] = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    inst["PAYMENT_DIFF"]    = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    inst["SHORTPAY_FLAG"]   = (inst["PAYMENT_RATIO"] < 0.95).astype(np.int8)
    inst["OVERPAY_FLAG"]    = (inst["PAYMENT_RATIO"] > 1.05).astype(np.int8)

    agg = inst.groupby(ID_COL).agg(
        INST_PAYMENT_RATIO_MEAN  = ("PAYMENT_RATIO",   "mean"),
        INST_PAYMENT_RATIO_MIN   = ("PAYMENT_RATIO",   "min"),
        INST_PAYMENT_RATIO_MAX   = ("PAYMENT_RATIO",   "max"),
        INST_PAYMENT_RATIO_STD   = ("PAYMENT_RATIO",   "std"),
        INST_DPD_MEAN            = ("DAYS_PAST_DUE",   "mean"),
        INST_DPD_MAX             = ("DAYS_PAST_DUE",   "max"),
        INST_DPD_SUM             = ("DAYS_PAST_DUE",   "sum"),
        INST_DPD_COUNT           = ("DAYS_PAST_DUE",   lambda x: (x > 0).sum()),
        INST_DPD_30_COUNT        = ("DAYS_PAST_DUE",   lambda x: (x > 30).sum()),
        INST_EARLY_DAYS_MEAN     = ("DAYS_BEFORE_DUE", "mean"),
        INST_SHORTPAY_COUNT      = ("SHORTPAY_FLAG",   "sum"),
        INST_OVERPAY_COUNT       = ("OVERPAY_FLAG",    "sum"),
        INST_PAYMENT_DIFF_MEAN   = ("PAYMENT_DIFF",    "mean"),
        INST_AMT_PAYMENT_SUM     = ("AMT_PAYMENT",     "sum"),
        INST_COUNT               = ("NUM_INSTALMENT_NUMBER", "count"),
    ).reset_index()

    # Shortpay rate
    agg["INST_SHORTPAY_RATE"] = agg["INST_SHORTPAY_COUNT"] / (agg["INST_COUNT"] + 1e-8)
    agg["INST_DPD_RATE"]      = agg["INST_DPD_COUNT"]      / (agg["INST_COUNT"] + 1e-8)

    log(f"  Installments agg shape: {agg.shape}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 4. Credit card balance features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_credit_card_features() -> pd.DataFrame:
    log("  Engineering credit_card_balance features ...")
    cc = pd.read_csv(DATA_FILES["credit_card"])

    cc["UTILISATION"] = (
        cc["AMT_BALANCE"] /
        (cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan) + 1e-8)
    ).clip(0, 1)

    cc["PAYMENT_RATIO"] = (
        cc["AMT_PAYMENT_CURRENT"] /
        (cc["AMT_INST_MIN_REGULARITY"].replace(0, np.nan) + 1e-8)
    ).clip(0, 5)

    cc["DPD_FLAG"]     = (cc["SK_DPD"]     > 0).astype(np.int8)
    cc["DPD_DEF_FLAG"] = (cc["SK_DPD_DEF"] > 0).astype(np.int8)

    agg = cc.groupby(ID_COL).agg(
        CC_BALANCE_MEAN       = ("AMT_BALANCE",             "mean"),
        CC_BALANCE_MAX        = ("AMT_BALANCE",             "max"),
        CC_BALANCE_STD        = ("AMT_BALANCE",             "std"),
        CC_UTILISATION_MEAN   = ("UTILISATION",             "mean"),
        CC_UTILISATION_MAX    = ("UTILISATION",             "max"),
        CC_PAYMENT_RATIO_MEAN = ("PAYMENT_RATIO",           "mean"),
        CC_PAYMENT_RATIO_MIN  = ("PAYMENT_RATIO",           "min"),
        CC_DRAWING_MEAN       = ("AMT_DRAWINGS_CURRENT",    "mean"),
        CC_DRAWING_SUM        = ("AMT_DRAWINGS_CURRENT",    "sum"),
        CC_DPD_COUNT          = ("DPD_FLAG",                "sum"),
        CC_DPD_DEF_COUNT      = ("DPD_DEF_FLAG",            "sum"),
        CC_DPD_MAX            = ("SK_DPD",                  "max"),
        CC_LIMIT_MEAN         = ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_LIMIT_MAX          = ("AMT_CREDIT_LIMIT_ACTUAL", "max"),
        CC_MONTHS_COUNT       = ("MONTHS_BALANCE",          "count"),
        CC_RECEIVABLE_MEAN    = ("AMT_RECEIVABLE_PRINCIPAL","mean"),
        CC_ATM_DRAW_MEAN      = ("AMT_DRAWINGS_ATM_CURRENT","mean"),
    ).reset_index()

    agg["CC_DPD_RATE"]     = agg["CC_DPD_COUNT"]     / (agg["CC_MONTHS_COUNT"] + 1e-8)
    agg["CC_DPD_DEF_RATE"] = agg["CC_DPD_DEF_COUNT"] / (agg["CC_MONTHS_COUNT"] + 1e-8)

    log(f"  Credit card agg shape: {agg.shape}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5. POS Cash balance features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_pos_cash_features() -> pd.DataFrame:
    log("  Engineering POS_CASH_balance features ...")
    pos = pd.read_csv(DATA_FILES["pos_cash"])

    pos["DPD_FLAG"] = (pos["SK_DPD"] > 0).astype(np.int8)

    agg = pos.groupby(ID_COL).agg(
        POS_DPD_MEAN         = ("SK_DPD",             "mean"),
        POS_DPD_MAX          = ("SK_DPD",             "max"),
        POS_DPD_COUNT        = ("DPD_FLAG",           "sum"),
        POS_MONTHS_COUNT     = ("MONTHS_BALANCE",     "count"),
        POS_COMPLETED_COUNT  = ("NAME_CONTRACT_STATUS",
                                lambda x: (x == "Completed").sum()),
        POS_ACTIVE_COUNT     = ("NAME_CONTRACT_STATUS",
                                lambda x: (x == "Active").sum()),
        POS_FUTURE_INST_MEAN = ("CNT_INSTALMENT_FUTURE", "mean"),
        POS_FUTURE_INST_MAX  = ("CNT_INSTALMENT_FUTURE", "max"),
    ).reset_index()

    agg["POS_DPD_RATE"]       = agg["POS_DPD_COUNT"]       / (agg["POS_MONTHS_COUNT"] + 1e-8)
    agg["POS_COMPLETED_RATE"] = agg["POS_COMPLETED_COUNT"] / (agg["POS_MONTHS_COUNT"] + 1e-8)

    log(f"  POS Cash agg shape: {agg.shape}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Previous applications features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_previous_app_features() -> pd.DataFrame:
    log("  Engineering previous_application features ...")
    prev = pd.read_csv(DATA_FILES["previous"])

    prev["APPROVED_FLAG"]  = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    prev["REFUSED_FLAG"]   = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    prev["CANCELLED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.int8)

    prev["CREDIT_RATIO"] = (
        prev["AMT_CREDIT"] /
        (prev["AMT_APPLICATION"].replace(0, np.nan) + 1e-8)
    ).clip(0, 5)

    prev["DOWN_PAYMENT_RATE"] = (
        prev["AMT_DOWN_PAYMENT"] /
        (prev["AMT_CREDIT"].replace(0, np.nan) + 1e-8)
    ).clip(0, 1)

    agg = prev.groupby(ID_COL).agg(
        PREV_APPLICATION_COUNT  = (ID_COL,                "count"),
        PREV_APPROVED_COUNT     = ("APPROVED_FLAG",       "sum"),
        PREV_REFUSED_COUNT      = ("REFUSED_FLAG",        "sum"),
        PREV_CANCELLED_COUNT    = ("CANCELLED_FLAG",      "sum"),
        PREV_CREDIT_RATIO_MEAN  = ("CREDIT_RATIO",        "mean"),
        PREV_CREDIT_RATIO_MIN   = ("CREDIT_RATIO",        "min"),
        PREV_DOWN_PMT_MEAN      = ("DOWN_PAYMENT_RATE",   "mean"),
        PREV_AMT_CREDIT_MEAN    = ("AMT_CREDIT",          "mean"),
        PREV_AMT_ANNUITY_MEAN   = ("AMT_ANNUITY",         "mean"),
        PREV_DAYS_DECISION_MEAN = ("DAYS_DECISION",       "mean"),
        PREV_DAYS_DECISION_MIN  = ("DAYS_DECISION",       "min"),
        PREV_INTEREST_RATE_MEAN = ("RATE_INTEREST_PRIMARY","mean"),
    ).reset_index()

    agg["PREV_APPROVAL_RATE"] = (
        agg["PREV_APPROVED_COUNT"] /
        (agg["PREV_APPLICATION_COUNT"] + 1e-8)
    )
    agg["PREV_REFUSAL_RATE"] = (
        agg["PREV_REFUSED_COUNT"] /
        (agg["PREV_APPLICATION_COUNT"] + 1e-8)
    )

    log(f"  Previous applications agg shape: {agg.shape}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. Join all engineered features to application table
# ─────────────────────────────────────────────────────────────────────────────

def join_all_features(app: pd.DataFrame) -> pd.DataFrame:
    log("\n  Joining all engineered feature tables ...")
    original_cols = app.shape[1]

    # Bureau
    bureau_agg = engineer_bureau_features()
    app = app.merge(bureau_agg, on=ID_COL, how="left")

    # Bureau balance
    bb_agg = engineer_bureau_balance_features()
    app = app.merge(bb_agg, on=ID_COL, how="left")

    # Instalment payments
    inst_agg = engineer_installments_features()
    app = app.merge(inst_agg, on=ID_COL, how="left")

    # Credit card
    cc_agg = engineer_credit_card_features()
    app = app.merge(cc_agg, on=ID_COL, how="left")

    # POS Cash
    pos_agg = engineer_pos_cash_features()
    app = app.merge(pos_agg, on=ID_COL, how="left")

    # Previous applications
    prev_agg = engineer_previous_app_features()
    app = app.merge(prev_agg, on=ID_COL, how="left")

    new_cols = app.shape[1] - original_cols
    log(f"  Total features added by joining: {new_cols}")
    log(f"  Final dataframe shape: {app.shape}")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# 8. Post-join missing value fill for alternative features
# ─────────────────────────────────────────────────────────────────────────────

def fill_alternative_feature_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative features are structurally missing for applicants who
    have no records in the corresponding supplementary table.
    Apply sentinel + indicator strategy to all new alternative feature columns.
    """
    alt_prefixes = ("INST_", "CC_", "POS_", "PREV_", "BUREAU_", "BB_")
    alt_cols = [c for c in df.columns
                if any(c.startswith(p) for p in alt_prefixes)
                and c not in (ID_COL, TARGET_COL, THIN_FILE_COL)]

    for col in alt_cols:
        missing_rate = df[col].isnull().mean()
        if missing_rate > 0:
            # Add indicator
            df[f"{col}_MISSING"] = df[col].isnull().astype(np.int8)
            # Fill with sentinel
            df[col] = df[col].fillna(SENTINEL_VALUE)

    total_indicators = sum(1 for c in df.columns if c.endswith("_MISSING"))
    log(f"  Alternative feature missingness indicators created: {total_indicators}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 03: Feature Engineering")
    log("=" * 60)

    # Load preprocessed application table
    log(f"Loading {ARTIFACTS['preprocessed']} ...")
    df = pd.read_parquet(ARTIFACTS["preprocessed"])
    log(f"  Loaded shape: {df.shape}")

    # 1. Application-level ratio features
    log("\n[1/3] Engineering application-level features ...")
    df = engineer_app_features(df)

    # 2. Join supplementary table aggregations
    log("\n[2/3] Engineering and joining supplementary table features ...")
    df = join_all_features(df)

    # 3. Post-join missingness treatment for alternative features
    log("\n[3/3] Post-join missingness treatment ...")
    df = fill_alternative_feature_missingness(df)

    # Summary
    log("\n===== FEATURE ENGINEERING SUMMARY =====")
    log(f"  Final shape          : {df.shape}")
    log(f"  Total features       : {df.shape[1] - 3}")  # excl. TARGET, ID, THIN_FILE
    log(f"  Remaining NaN count  : {df.isnull().sum().sum()}")
    log("========================================\n")

    # Save
    log(f"Saving to {ARTIFACTS['featured']} ...")
    df.to_parquet(ARTIFACTS["featured"], index=False)
    log("Step 03 complete.\n")
    return df


if __name__ == "__main__":
    main()
