"""
04_train_test_split.py
-----------------------
Creates stratified train / test splits and identifies:
  - Feature sets: bureau-only vs combined
  - Thin-file sub-cohort mask for test set
  - Saves arrays as .npz and feature names as JSON

Input:  outputs/03_featured.parquet
Output: outputs/04_train_test.npz
        outputs/04_feature_names.json
        outputs/04_thin_file_mask_test.npy
"""

import os
import sys
import json
import warnings
import numpy  as np
import pandas as pd
import joblib
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACTS, MODELS, TARGET_COL, ID_COL, THIN_FILE_COL,
    TEST_SIZE, RANDOM_SEED, VERBOSE,
    BUREAU_FEATURE_PREFIXES, ALTERNATIVE_FEATURE_PREFIXES,
    SCALE_POS_WEIGHT
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[04_train_test_split] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature set construction
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_sets(df: pd.DataFrame) -> tuple:
    """
    Returns:
      all_features      : all numeric predictors
      bureau_features   : subset containing only traditional (bureau + application) features
      alternative_features : subset of alternative (supplementary table) features
    """
    # Columns to always exclude
    exclude_cols = {
        TARGET_COL, ID_COL, THIN_FILE_COL,
        "BUREAU_RECORD_COUNT",
    }

    all_features = [c for c in df.columns
                    if c not in exclude_cols
                    and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.int8, np.uint8)]

    # Bureau-only features: application table + bureau table columns
    # (excludes engineered features from supplementary tables)
    alt_prefixes  = tuple(ALTERNATIVE_FEATURE_PREFIXES + ["INST_", "CC_", "POS_", "PREV_", "BB_"])
    bureau_features = [f for f in all_features
                       if not any(f.startswith(p) for p in alt_prefixes)]

    alternative_features = [f for f in all_features if f not in bureau_features]

    log(f"  Total features     : {len(all_features)}")
    log(f"  Bureau-only set    : {len(bureau_features)}")
    log(f"  Alternative set    : {len(alternative_features)}")

    return all_features, bureau_features, alternative_features


# ─────────────────────────────────────────────────────────────────────────────
# Train-test split
# ─────────────────────────────────────────────────────────────────────────────

def create_split(df: pd.DataFrame, feature_list: list) -> tuple:
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    y = df[TARGET_COL].values.astype(np.int8)
    X = df[feature_list].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Scale features for LR
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_save_scaler(X_train: np.ndarray, feature_names: list) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS["scaler"])
    log(f"  StandardScaler fitted and saved to {MODELS['scaler']}")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Class imbalance ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos
    log(f"  scale_pos_weight = {spw:.4f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")
    return float(spw)


# ─────────────────────────────────────────────────────────────────────────────
# Thin-file mask
# ─────────────────────────────────────────────────────────────────────────────

def get_thin_file_mask(df_full: pd.DataFrame, test_indices: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array of shape (n_test,) where True = thin-file applicant.
    """
    thin_flags = df_full[THIN_FILE_COL].values
    return thin_flags[test_indices].astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 04: Train-Test Split")
    log("=" * 60)

    log(f"Loading {ARTIFACTS['featured']} ...")
    df = pd.read_parquet(ARTIFACTS["featured"])
    log(f"  Loaded shape: {df.shape}")

    # Replace inf/-inf with sentinel
    df = df.replace([np.inf, -np.inf], -999.0)

    # Get feature sets
    log("\nBuilding feature sets ...")
    all_features, bureau_features, alt_features = get_feature_sets(df)

    # ── COMBINED SPLIT ────────────────────────────────────────────────────────
    log("\nCreating combined-feature train/test split ...")
    y   = df[TARGET_COL].values.astype(np.int8)
    idx = np.arange(len(df))

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    X_combined = df[all_features].values.astype(np.float32)
    X_train_comb = X_combined[idx_train]
    X_test_comb  = X_combined[idx_test]

    log(f"  Train shape: {X_train_comb.shape} | Test shape: {X_test_comb.shape}")
    log(f"  Train default rate: {y_train.mean():.4f}")
    log(f"  Test  default rate: {y_test.mean():.4f}")

    # ── BUREAU-ONLY SPLIT (same indices) ─────────────────────────────────────
    X_bureau = df[bureau_features].values.astype(np.float32)
    X_train_bur = X_bureau[idx_train]
    X_test_bur  = X_bureau[idx_test]

    # ── SCALER ───────────────────────────────────────────────────────────────
    log("\nFitting StandardScaler on training data ...")
    scaler = fit_and_save_scaler(X_train_comb, all_features)
    X_train_comb_scaled = scaler.transform(X_train_comb)
    X_test_comb_scaled  = scaler.transform(X_test_comb)
    X_train_bur_scaled  = scaler.transform(
        np.column_stack([X_train_bur,
                         np.zeros((len(X_train_bur),
                                   X_train_comb.shape[1] - X_train_bur.shape[1]))])
    )[:, :X_train_bur.shape[1]]
    X_test_bur_scaled   = scaler.transform(
        np.column_stack([X_test_bur,
                         np.zeros((len(X_test_bur),
                                   X_test_comb.shape[1] - X_test_bur.shape[1]))])
    )[:, :X_test_bur.shape[1]]

    # ── CLASS WEIGHT ─────────────────────────────────────────────────────────
    log("\nComputing class imbalance ratio ...")
    spw = compute_scale_pos_weight(y_train)

    # ── THIN-FILE MASK ────────────────────────────────────────────────────────
    log("\nBuilding thin-file mask for test set ...")
    thin_file_mask_test = df[THIN_FILE_COL].values[idx_test].astype(bool)
    n_thin_test  = thin_file_mask_test.sum()
    n_total_test = len(thin_file_mask_test)
    log(f"  Thin-file in test set: {n_thin_test:,} / {n_total_test:,} ({n_thin_test/n_total_test*100:.1f}%)")

    # ── SAVE ARRAYS ──────────────────────────────────────────────────────────
    log(f"\nSaving arrays to {ARTIFACTS['train_test']} ...")
    np.savez_compressed(
        ARTIFACTS["train_test"],
        # Combined feature set (unscaled — for tree models)
        X_train_comb        = X_train_comb,
        X_test_comb         = X_test_comb,
        # Combined feature set (scaled — for LR)
        X_train_comb_scaled = X_train_comb_scaled,
        X_test_comb_scaled  = X_test_comb_scaled,
        # Bureau-only feature set (unscaled — for tree models)
        X_train_bur         = X_train_bur,
        X_test_bur          = X_test_bur,
        # Bureau-only feature set (scaled — for LR)
        X_train_bur_scaled  = X_train_bur_scaled,
        X_test_bur_scaled   = X_test_bur_scaled,
        # Labels
        y_train             = y_train,
        y_test              = y_test,
        # Index tracking
        idx_train           = idx_train,
        idx_test            = idx_test,
        # Class weight
        scale_pos_weight    = np.array([spw]),
    )

    # Save thin-file mask separately
    np.save(ARTIFACTS["thin_file_mask_test"], thin_file_mask_test)

    # Save feature name lists
    feature_names_out = {
        "all_features":         all_features,
        "bureau_features":      bureau_features,
        "alternative_features": alt_features,
    }
    with open(ARTIFACTS["feature_names"], "w") as f:
        json.dump(feature_names_out, f, indent=2)

    log(f"Feature names saved to {ARTIFACTS['feature_names']}")

    # Summary
    log("\n===== SPLIT SUMMARY =====")
    log(f"  Training samples     : {len(y_train):,}")
    log(f"  Test samples         : {len(y_test):,}")
    log(f"  Combined features    : {len(all_features)}")
    log(f"  Bureau-only features : {len(bureau_features)}")
    log(f"  Alternative features : {len(alt_features)}")
    log(f"  Thin-file test size  : {n_thin_test:,}")
    log(f"  scale_pos_weight     : {spw:.4f}")
    log("=========================\n")

    log("Step 04 complete.\n")


if __name__ == "__main__":
    main()
