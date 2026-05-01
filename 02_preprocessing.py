"""
02_preprocessing.py
--------------------
Full preprocessing pipeline:
  - Sentinel value replacement (DAYS_EMPLOYED anomaly)
  - Log transforms for skewed financial columns
  - Missing value analysis & indicator-augmented imputation
  - Binary column label encoding
  - One-hot encoding for nominal categoricals
  - StandardScaler fit on training split (saved for test use)
  - Class imbalance strategy (class_weight computation)

Input:  outputs/01_raw_joined.parquet
Output: outputs/02_preprocessed.parquet
        models/standard_scaler.joblib
"""

import os
import sys
import json
import warnings
import numpy  as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACTS, MODELS, TARGET_COL, ID_COL, THIN_FILE_COL,
    MISSING_LOW_THRESHOLD, MISSING_MODERATE_THRESHOLD, SENTINEL_VALUE,
    OHE_COLS, BINARY_COLS, LOG_TRANSFORM_COLS,
    RANDOM_SEED, VERBOSE
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[02_preprocessing] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sentinel value treatment
# ─────────────────────────────────────────────────────────────────────────────

def fix_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    DAYS_EMPLOYED = 365243 is a sentinel for pensioners/non-employed.
    Replace with NaN and create a binary indicator.
    """
    df = df.copy()
    SENTINEL_DAYS_EMPLOYED = 365243

    mask = df["DAYS_EMPLOYED"] == SENTINEL_DAYS_EMPLOYED
    df["DAYS_EMPLOYED_ANOM"] = mask.astype(np.int8)
    df.loc[mask, "DAYS_EMPLOYED"] = np.nan

    log(f"  DAYS_EMPLOYED anomalies replaced: {mask.sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Log transforms
# ─────────────────────────────────────────────────────────────────────────────

def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))
    log(f"  Log-transformed: {LOG_TRANSFORM_COLS}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing value analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_missingness(df: pd.DataFrame) -> dict:
    """Returns dict: feature -> (missing_rate, strategy)."""
    result = {}
    for col in df.columns:
        if col in [TARGET_COL, ID_COL, THIN_FILE_COL, "BUREAU_RECORD_COUNT"]:
            continue
        rate = df[col].isnull().mean()
        if rate == 0:
            strategy = "none"
        elif rate < MISSING_LOW_THRESHOLD:
            strategy = "mean_mode"
        elif rate < MISSING_MODERATE_THRESHOLD:
            strategy = "mean_mode_indicator"
        else:
            strategy = "sentinel_indicator"
        result[col] = {"missing_rate": float(rate), "strategy": strategy}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Imputation
# ─────────────────────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame,
           missingness_map: dict,
           imputation_stats: dict = None,
           fit: bool = True) -> tuple:
    """
    Applies imputation according to missingness_map.
    If fit=True, computes imputation statistics from df.
    If fit=False, applies pre-computed statistics (for test set).

    Returns: (imputed_df, imputation_stats)
    """
    df = df.copy()
    stats = imputation_stats if imputation_stats else {}

    for col, info in missingness_map.items():
        if col not in df.columns:
            continue

        strategy = info["strategy"]
        if strategy == "none":
            continue

        # Add indicator flag if needed
        if strategy in ("mean_mode_indicator", "sentinel_indicator"):
            indicator_col = f"{col}_MISSING"
            df[indicator_col] = df[col].isnull().astype(np.int8)

        if strategy == "sentinel_indicator":
            df[col] = df[col].fillna(SENTINEL_VALUE)

        elif strategy in ("mean_mode", "mean_mode_indicator"):
            if df[col].dtype == "object":
                # Mode imputation
                if fit:
                    fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    stats[col] = fill_val
                else:
                    fill_val = stats.get(col, "Unknown")
                df[col] = df[col].fillna(fill_val)
            else:
                # Mean imputation
                if fit:
                    fill_val = float(df[col].mean())
                    stats[col] = fill_val
                else:
                    fill_val = stats.get(col, 0.0)
                df[col] = df[col].fillna(fill_val)

    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# 5. Binary encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = (df[col] == "Y").astype(np.int8)
    # CODE_GENDER: M=0, F=1, XNA=NaN
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].map({"M": 0, "F": 1, "XNA": np.nan})
        df["CODE_GENDER"] = df["CODE_GENDER"].fillna(-1).astype(np.int8)
    log(f"  Binary-encoded: {BINARY_COLS + ['CODE_GENDER']}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. One-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_encode(df: pd.DataFrame,
                   ohe_categories: dict = None,
                   fit: bool = True) -> tuple:
    """
    One-hot encode nominal categoricals.
    fit=True  → derive categories from df (training)
    fit=False → apply stored categories (test set alignment)
    Returns: (encoded_df, ohe_categories_dict)
    """
    df = df.copy()
    cats = ohe_categories if ohe_categories else {}

    for col in OHE_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Missing").astype(str)

        if fit:
            # Compute categories from training data
            unique_cats = sorted(df[col].unique().tolist())
            cats[col] = unique_cats
        else:
            unique_cats = cats.get(col, [])

        # Create dummy columns, dropping first (reference category = most frequent)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)

        # Align to training categories
        expected_cols = [f"{col}_{c}" for c in unique_cats]
        for ec in expected_cols:
            if ec not in dummies.columns:
                dummies[ec] = 0
        # Drop extra cols not in training set
        dummies = dummies[[c for c in expected_cols if c in dummies.columns]]
        # Drop first category (reference)
        if len(dummies.columns) > 1:
            dummies = dummies.iloc[:, 1:]

        df = df.drop(columns=[col])
        df = pd.concat([df, dummies.astype(np.int8)], axis=1)

    log(f"  One-hot encoded {len(OHE_COLS)} categorical columns.")
    return df, cats


# ─────────────────────────────────────────────────────────────────────────────
# 7. Remove non-feature columns (keep only numeric)
# ─────────────────────────────────────────────────────────────────────────────

def drop_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any remaining object columns that were not OHE-encoded.
    Keep TARGET, ID, THIN_FILE_FLAG regardless.
    """
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    keep_always = [TARGET_COL, ID_COL, THIN_FILE_COL, "BUREAU_RECORD_COUNT"]
    to_drop = [c for c in obj_cols if c not in keep_always]
    if to_drop:
        log(f"  Dropping {len(to_drop)} remaining object columns: {to_drop[:10]} ...")
        df = df.drop(columns=to_drop)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Print preprocessing summary
# ─────────────────────────────────────────────────────────────────────────────

def preprocessing_summary(df: pd.DataFrame, missingness_map: dict):
    strat_counts = {}
    for info in missingness_map.values():
        s = info["strategy"]
        strat_counts[s] = strat_counts.get(s, 0) + 1

    log("\n===== PREPROCESSING SUMMARY =====")
    log(f"  Final dataframe shape : {df.shape}")
    log(f"  Remaining NaNs        : {df.isnull().sum().sum()}")
    log(f"  Imputation strategies :")
    for s, cnt in strat_counts.items():
        log(f"    {s:30s}: {cnt} features")
    log("==================================\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 02: Preprocessing")
    log("=" * 60)

    # Load
    log(f"Loading {ARTIFACTS['raw_joined']} ...")
    df = pd.read_parquet(ARTIFACTS["raw_joined"])
    log(f"  Loaded shape: {df.shape}")

    # 1. Sentinel fixes
    log("\n[1/7] Fixing sentinel values ...")
    df = fix_sentinel_values(df)

    # 2. Log transforms
    log("\n[2/7] Applying log transforms ...")
    df = apply_log_transforms(df)

    # 3. Binary encoding
    log("\n[3/7] Binary encoding ...")
    df = encode_binary(df)

    # 4. OHE (fit on full dataset here; will be re-applied in train/test split)
    log("\n[4/7] One-hot encoding ...")
    df, ohe_cats = one_hot_encode(df, fit=True)

    # 5. Analyse missingness
    log("\n[5/7] Analysing missingness ...")
    missingness_map = analyse_missingness(df)
    sentinel_count   = sum(1 for v in missingness_map.values() if v["strategy"] == "sentinel_indicator")
    indicator_count  = sum(1 for v in missingness_map.values() if v["strategy"] == "mean_mode_indicator")
    log(f"  Sentinel+indicator strategy: {sentinel_count} features")
    log(f"  Mean/mode+indicator strategy: {indicator_count} features")

    # 6. Imputation
    log("\n[6/7] Imputing missing values ...")
    df, imputation_stats = impute(df, missingness_map, fit=True)

    # 7. Drop non-numeric
    log("\n[7/7] Dropping remaining object columns ...")
    df = drop_non_numeric(df)

    # Summary
    preprocessing_summary(df, missingness_map)

    # Save preprocessed data
    log(f"Saving preprocessed data to {ARTIFACTS['preprocessed']} ...")
    df.to_parquet(ARTIFACTS["preprocessed"], index=False)

    # Save preprocessing artifacts for test-set application
    preproc_artifacts = {
        "missingness_map":   missingness_map,
        "imputation_stats":  imputation_stats,
        "ohe_categories":    ohe_cats,
    }
    import pickle
    with open(os.path.join(os.path.dirname(ARTIFACTS["preprocessed"]),
                           "02_preproc_artifacts.pkl"), "wb") as f:
        pickle.dump(preproc_artifacts, f)
    log("Preprocessing artifacts saved.")

    log("\nStep 02 complete.\n")
    return df


if __name__ == "__main__":
    main()
