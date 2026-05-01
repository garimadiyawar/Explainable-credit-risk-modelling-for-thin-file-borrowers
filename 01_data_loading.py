"""
01_data_loading.py
------------------
Loads all seven Home Credit dataset files, validates their presence,
identifies thin-file borrowers (zero bureau records), and saves the
joined application table with the THIN_FILE_FLAG column to disk.

Output: outputs/01_raw_joined.parquet
"""

import os
import sys
import json
import numpy  as np
import pandas as pd
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_FILES, ARTIFACTS, TARGET_COL, ID_COL,
    THIN_FILE_COL, VERBOSE, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    if VERBOSE:
        print(f"[01_data_loading] {msg}")


def check_files():
    """Verify all required data files exist before loading."""
    missing = [k for k, v in DATA_FILES.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files: {missing}\n"
            f"Download from https://www.kaggle.com/c/home-credit-default-risk/data "
            f"and place CSVs in the data/ folder."
        )
    log("All dataset files found.")


# ─────────────────────────────────────────────────────────────────────────────
# Loading functions
# ─────────────────────────────────────────────────────────────────────────────

def load_application() -> pd.DataFrame:
    log("Loading application_train.csv ...")
    df = pd.read_csv(DATA_FILES["application"])
    log(f"  Shape: {df.shape} | Default rate: {df[TARGET_COL].mean():.4f}")
    return df


def load_bureau() -> pd.DataFrame:
    log("Loading bureau.csv ...")
    df = pd.read_csv(DATA_FILES["bureau"])
    log(f"  Shape: {df.shape}")
    return df


def load_bureau_balance() -> pd.DataFrame:
    log("Loading bureau_balance.csv ...")
    df = pd.read_csv(DATA_FILES["bureau_balance"])
    log(f"  Shape: {df.shape}")
    return df


def load_previous() -> pd.DataFrame:
    log("Loading previous_application.csv ...")
    df = pd.read_csv(DATA_FILES["previous"])
    log(f"  Shape: {df.shape}")
    return df


def load_pos_cash() -> pd.DataFrame:
    log("Loading POS_CASH_balance.csv ...")
    df = pd.read_csv(DATA_FILES["pos_cash"])
    log(f"  Shape: {df.shape}")
    return df


def load_installments() -> pd.DataFrame:
    log("Loading installments_payments.csv ...")
    df = pd.read_csv(DATA_FILES["installments"])
    log(f"  Shape: {df.shape}")
    return df


def load_credit_card() -> pd.DataFrame:
    log("Loading credit_card_balance.csv ...")
    df = pd.read_csv(DATA_FILES["credit_card"])
    log(f"  Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Thin-file identification
# ─────────────────────────────────────────────────────────────────────────────

def identify_thin_file(app: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Mark applicants with zero records in bureau.csv as thin-file.
    THIN_FILE_FLAG = 1  → thin-file borrower
    THIN_FILE_FLAG = 0  → credit-active borrower
    """
    applicants_with_bureau = set(bureau[ID_COL].unique())
    app[THIN_FILE_COL] = (~app[ID_COL].isin(applicants_with_bureau)).astype(np.int8)

    n_thin   = app[THIN_FILE_COL].sum()
    n_total  = len(app)
    pct      = n_thin / n_total * 100

    log(f"  Thin-file applicants: {n_thin:,} / {n_total:,} ({pct:.1f}%)")
    log(f"  Credit-active applicants: {n_total - n_thin:,} ({100 - pct:.1f}%)")

    # Default rate by thin-file status
    for flag, label in [(0, "Credit-active"), (1, "Thin-file")]:
        subset      = app[app[THIN_FILE_COL] == flag]
        default_rt  = subset[TARGET_COL].mean()
        log(f"  {label} default rate: {default_rt:.4f}")

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Bureau table basic aggregation (for joining thin-file flag lookup)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_bureau_counts(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple count of bureau records per applicant.
    This is saved alongside the application table for later use.
    """
    counts = (
        bureau.groupby(ID_COL)
        .size()
        .reset_index(name="BUREAU_RECORD_COUNT")
    )
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Dataset summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(app: pd.DataFrame):
    log("\n========== DATASET SUMMARY ==========")
    log(f"Total applications         : {len(app):>10,}")
    log(f"Features (raw)             : {app.shape[1]:>10,}")
    log(f"Target = 1 (default)       : {app[TARGET_COL].sum():>10,}  ({app[TARGET_COL].mean()*100:.2f}%)")
    log(f"Target = 0 (no default)    : {(app[TARGET_COL]==0).sum():>10,}  ({(1-app[TARGET_COL].mean())*100:.2f}%)")
    log(f"Thin-file borrowers        : {app[THIN_FILE_COL].sum():>10,}  ({app[THIN_FILE_COL].mean()*100:.2f}%)")

    missing_pct = app.isnull().mean().sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 0.40]
    log(f"Features with >40% missing : {len(high_missing):>10,}")
    log("======================================\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 01: Data Loading and Thin-File Identification")
    log("=" * 60)

    # 1. Validate files
    check_files()

    # 2. Load all tables
    app    = load_application()
    bureau = load_bureau()
    _      = load_bureau_balance()   # loaded but not joined here — used in FE
    _      = load_previous()
    _      = load_pos_cash()
    _      = load_installments()
    _      = load_credit_card()

    # 3. Identify thin-file borrowers
    log("\nIdentifying thin-file borrowers ...")
    app = identify_thin_file(app, bureau)

    # 4. Add bureau record count (useful for sub-cohort analysis)
    bureau_counts = aggregate_bureau_counts(bureau)
    app = app.merge(bureau_counts, on=ID_COL, how="left")
    app["BUREAU_RECORD_COUNT"] = app["BUREAU_RECORD_COUNT"].fillna(0).astype(int)

    # 5. Print summary
    print_dataset_summary(app)

    # 6. Save joined application table
    log(f"Saving to {ARTIFACTS['raw_joined']} ...")
    app.to_parquet(ARTIFACTS["raw_joined"], index=False)
    log(f"Saved. Shape: {app.shape}")

    log("\nStep 01 complete.\n")
    return app


if __name__ == "__main__":
    main()
