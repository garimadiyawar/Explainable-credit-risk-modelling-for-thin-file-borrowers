"""
08_fairness_evaluation.py
--------------------------
Fairness audit across demographic subgroups for:
  - LR Bureau-Only (baseline)
  - XGBoost Combined (best model)

Metrics:
  - Demographic Parity Difference (DPD)
  - Equalised Odds Difference (EOD)
  - Average Odds Difference (AOD)
  - Bootstrap 95% confidence intervals for each

Sensitive attributes (proxy demographics available in dataset):
  - NAME_EDUCATION_TYPE
  - NAME_FAMILY_STATUS

Input:  models/*.joblib
        outputs/04_train_test.npz
        outputs/03_featured.parquet   (for demographic columns)
Output: outputs/08_fairness_results.json
"""

import os
import sys
import json
import warnings
import numpy  as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACTS, MODELS, RANDOM_SEED, VERBOSE,
    TARGET_COL, ID_COL, THIN_FILE_COL,
    FAIRNESS_COLS, BOOTSTRAP_N, FAIRNESS_ALPHA
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[08_fairness_evaluation] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Fairness metric helpers (no fairlearn dependency required)
# ─────────────────────────────────────────────────────────────────────────────

def demographic_parity_difference(y_true, y_pred, groups):
    """Max approval rate minus min approval rate across subgroups."""
    unique_g = np.unique(groups)
    rates    = {g: y_pred[groups == g].mean() for g in unique_g}
    vals     = list(rates.values())
    return float(max(vals) - min(vals)), rates


def equalised_odds_difference(y_true, y_pred, groups):
    """
    EOD = max over groups of |TPR_g - TPR_ref| + |FPR_g - FPR_ref|
    where ref = group with lowest default rate.
    """
    unique_g = np.unique(groups)

    def tpr(g):
        mask = (groups == g) & (y_true == 1)
        return y_pred[mask].mean() if mask.sum() > 0 else 0.0

    def fpr(g):
        mask = (groups == g) & (y_true == 0)
        return y_pred[mask].mean() if mask.sum() > 0 else 0.0

    # Reference = highest TPR (best-performing) group
    tprs = {g: tpr(g) for g in unique_g}
    fprs = {g: fpr(g) for g in unique_g}
    ref  = max(tprs, key=tprs.get)

    eod  = max(
        abs(tprs[g] - tprs[ref]) + abs(fprs[g] - fprs[ref])
        for g in unique_g if g != ref
    ) if len(unique_g) > 1 else 0.0

    return float(eod), tprs, fprs


def average_odds_difference(y_true, y_pred, groups):
    """AOD = average of |ΔTPR| and |ΔFPR| across groups vs reference."""
    unique_g = np.unique(groups)

    def tpr(g):
        mask = (groups == g) & (y_true == 1)
        return y_pred[mask].mean() if mask.sum() > 0 else 0.0

    def fpr(g):
        mask = (groups == g) & (y_true == 0)
        return y_pred[mask].mean() if mask.sum() > 0 else 0.0

    ref  = max(unique_g, key=lambda g: tpr(g))
    aod  = np.mean([
        0.5 * (abs(tpr(g) - tpr(ref)) + abs(fpr(g) - fpr(ref)))
        for g in unique_g if g != ref
    ]) if len(unique_g) > 1 else 0.0
    return float(aod)


def bootstrap_metric(y_true, y_pred, groups, metric_fn,
                     n_bootstrap=BOOTSTRAP_N, alpha=FAIRNESS_ALPHA):
    """Bootstrap confidence interval for a fairness metric."""
    rng      = np.random.RandomState(RANDOM_SEED)
    n        = len(y_true)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx   = rng.choice(n, n, replace=True)
        val   = metric_fn(y_true[idx], y_pred[idx], groups[idx])
        if isinstance(val, tuple):
            val = val[0]
        boot_vals.append(val)
    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Load data and models
# ─────────────────────────────────────────────────────────────────────────────

def load_everything():
    log("Loading data and models ...")

    data = np.load(ARTIFACTS["train_test"], allow_pickle=True)
    X_test_comb   = data["X_test_comb"]
    X_test_bur    = data["X_test_bur"]
    X_test_comb_s = data["X_test_comb_scaled"]
    X_test_bur_s  = data["X_test_bur_scaled"]
    y_test        = data["y_test"]
    idx_test      = data["idx_test"]

    # Load featured dataframe for demographic columns
    df_full = pd.read_parquet(ARTIFACTS["featured"])

    # Models
    lr_bur  = joblib.load(MODELS["lr_bureau"])
    xgb_comb = joblib.load(MODELS["xgb_combined"])

    log(f"  Test size: {len(y_test):,}")
    return (X_test_comb, X_test_bur, X_test_comb_s, X_test_bur_s,
            y_test, idx_test, df_full, lr_bur, xgb_comb)


# ─────────────────────────────────────────────────────────────────────────────
# Run fairness audit for one model × one sensitive attribute
# ─────────────────────────────────────────────────────────────────────────────

def audit_one(y_true, y_pred_proba, y_pred_class, groups, model_label, attr_label):
    log(f"\n  [{model_label}] × [{attr_label}]")
    unique_g = np.unique(groups)
    log(f"  Subgroups: {list(unique_g)}")

    results = {
        "model":     model_label,
        "attribute": attr_label,
        "subgroups": {},
        "dpd":       None, "dpd_ci": None,
        "eod":       None, "eod_ci": None,
        "aod":       None, "aod_ci": None,
    }

    # Per-subgroup stats
    for g in unique_g:
        mask  = groups == g
        n_g   = mask.sum()
        if n_g < 10:
            continue
        auc_g = roc_auc_score(y_true[mask], y_pred_proba[mask]) if y_true[mask].sum() > 0 else np.nan
        apr_g = y_pred_class[mask].mean()
        def_g = y_true[mask].mean()
        results["subgroups"][str(g)] = {
            "n":            int(n_g),
            "default_rate": float(def_g),
            "approval_rate":float(apr_g),
            "auc":          float(auc_g) if not np.isnan(auc_g) else None,
        }
        log(f"    {str(g):40s}  n={n_g:5d}  default={def_g:.3f}  approval={apr_g:.3f}  AUC={auc_g:.3f}")

    # DPD
    dpd, _  = demographic_parity_difference(y_true, y_pred_class, groups)
    dpd_lo, dpd_hi = bootstrap_metric(y_true, y_pred_class, groups,
                                       lambda yt, yp, g: demographic_parity_difference(yt, yp, g)[0])
    results["dpd"]    = round(dpd, 4)
    results["dpd_ci"] = [round(dpd_lo, 4), round(dpd_hi, 4)]
    log(f"  Demographic Parity Diff  : {dpd:.4f}  95% CI [{dpd_lo:.4f}, {dpd_hi:.4f}]")

    # EOD
    eod, _, _ = equalised_odds_difference(y_true, y_pred_class, groups)
    eod_lo, eod_hi = bootstrap_metric(y_true, y_pred_class, groups,
                                       lambda yt, yp, g: equalised_odds_difference(yt, yp, g)[0])
    results["eod"]    = round(eod, 4)
    results["eod_ci"] = [round(eod_lo, 4), round(eod_hi, 4)]
    log(f"  Equalised Odds Diff      : {eod:.4f}  95% CI [{eod_lo:.4f}, {eod_hi:.4f}]")

    # AOD
    aod = average_odds_difference(y_true, y_pred_class, groups)
    aod_lo, aod_hi = bootstrap_metric(y_true, y_pred_class, groups,
                                       average_odds_difference)
    results["aod"]    = round(aod, 4)
    results["aod_ci"] = [round(aod_lo, 4), round(aod_hi, 4)]
    log(f"  Average Odds Diff        : {aod:.4f}  95% CI [{aod_lo:.4f}, {aod_hi:.4f}]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 08: Fairness Evaluation")
    log("=" * 60)

    (X_test_comb, X_test_bur, X_test_comb_s, X_test_bur_s,
     y_test, idx_test, df_full, lr_bur, xgb_comb) = load_everything()

    # Predicted probabilities
    lr_proba   = lr_bur.predict_proba(X_test_bur_s)[:, 1]
    xgb_proba  = xgb_comb.predict_proba(X_test_comb)[:, 1]

    # Optimal thresholds (use 0.3 as default — adjust based on Step 05 output)
    LR_THRESH  = 0.3
    XGB_THRESH = 0.35
    lr_pred    = (lr_proba  >= LR_THRESH).astype(int)
    xgb_pred   = (xgb_proba >= XGB_THRESH).astype(int)

    # Extract demographic columns from the full featured dataframe
    # (test set rows identified by idx_test)
    df_test = df_full.iloc[idx_test].reset_index(drop=True)

    all_results = []

    for attr in FAIRNESS_COLS:
        if attr not in df_test.columns:
            # Try OHE-encoded version — find original categorical from featured df
            # Fall back to checking raw parquet
            log(f"  Column '{attr}' not in featured df — checking raw ...")
            # Reload the raw preprocessed df which still has original categoricals
            try:
                df_raw = pd.read_parquet(ARTIFACTS["preprocessed"])
                if attr in df_raw.columns:
                    groups_raw = df_raw[attr].iloc[idx_test].fillna("Unknown").values
                else:
                    log(f"  Skipping '{attr}' — not found.")
                    continue
                groups = groups_raw
            except Exception:
                log(f"  Skipping '{attr}' — error loading.")
                continue
        else:
            groups = df_test[attr].fillna("Unknown").astype(str).values

        log(f"\n{'='*50}")
        log(f"Sensitive attribute: {attr}")
        log(f"{'='*50}")

        # LR Bureau-Only baseline
        r_lr  = audit_one(y_test, lr_proba,  lr_pred,  groups, "LR_Bureau-Only",  attr)
        # XGB Combined best model
        r_xgb = audit_one(y_test, xgb_proba, xgb_pred, groups, "XGB_Combined",    attr)

        # Compute delta (XGB - LR) for each metric
        delta = {
            "dpd_delta": round(r_xgb["dpd"] - r_lr["dpd"], 4),
            "eod_delta": round(r_xgb["eod"] - r_lr["eod"], 4),
            "aod_delta": round(r_xgb["aod"] - r_lr["aod"], 4),
        }
        log(f"\n  DELTA (XGB - LR):  DPD={delta['dpd_delta']:+.4f}  "
            f"EOD={delta['eod_delta']:+.4f}  AOD={delta['aod_delta']:+.4f}")
        p1_indicator = "SUPPORTED" if delta["dpd_delta"] <= 0.01 else "NOT SUPPORTED"
        log(f"  P1 (Fairness Non-Deterioration): {p1_indicator}")

        all_results.append({
            "attribute":   attr,
            "lr_results":  r_lr,
            "xgb_results": r_xgb,
            "delta":       delta,
            "p1_supported": p1_indicator == "SUPPORTED",
        })

    # Save
    with open(ARTIFACTS["fairness_results"], "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nFairness results saved to {ARTIFACTS['fairness_results']}")

    # Summary
    log("\n===== FAIRNESS AUDIT SUMMARY =====")
    for r in all_results:
        attr  = r["attribute"]
        d     = r["delta"]
        log(f"  {attr:30s}  ΔDPD={d['dpd_delta']:+.4f}  ΔEOD={d['eod_delta']:+.4f}  "
            f"P1={'✓' if r['p1_supported'] else '✗'}")
    log("===================================")
    log("\nStep 08 complete.\n")


if __name__ == "__main__":
    main()
