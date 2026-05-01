"""
05_model_training.py
---------------------
Trains all four model classes on both bureau-only and combined feature sets:
  - Logistic Regression (bureau-only + combined, scaled features)
  - Random Forest       (bureau-only + combined)
  - XGBoost             (bureau-only + combined, uses best params if available)
  - LightGBM            (bureau-only + combined)

Evaluates each on the held-out test set (full + thin-file sub-cohort).
Saves all models and cross-validation results.

Input:  outputs/04_train_test.npz  +  outputs/04_feature_names.json
Output: models/*.joblib
        outputs/05_cv_results.json
"""

import os
import sys
import json
import warnings
import numpy  as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    roc_auc_score, precision_score, recall_score,
    f1_score, brier_score_loss, roc_curve
)
import xgboost  as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACTS, MODELS, CV_FOLDS, RANDOM_SEED, VERBOSE
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[05_model_training] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    log(f"Loading {ARTIFACTS['train_test']} ...")
    data = np.load(ARTIFACTS["train_test"], allow_pickle=True)

    with open(ARTIFACTS["feature_names"]) as f:
        feat = json.load(f)

    thin_mask = np.load(ARTIFACTS["thin_file_mask_test"])
    spw       = float(data["scale_pos_weight"][0])

    log(f"  scale_pos_weight = {spw:.4f}")
    return data, feat, thin_mask, spw


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, thin_mask, label=""):
    proba     = model.predict_proba(X_test)[:, 1]

    # Find F1-optimal threshold on test set
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        f1 = f1_score(y_test, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t

    y_pred  = (proba >= best_thresh).astype(int)
    auc     = roc_auc_score(y_test, proba)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1v     = f1_score(y_test, y_pred, zero_division=0)
    brier   = brier_score_loss(y_test, proba)

    # KS statistic
    scores_pos = proba[y_test == 1]
    scores_neg = proba[y_test == 0]
    ks_stat, _ = ks_2samp(scores_pos, scores_neg)

    # Thin-file sub-cohort AUC
    auc_thin   = roc_auc_score(y_test[thin_mask],  proba[thin_mask])  if thin_mask.sum() > 10 else np.nan
    auc_active = roc_auc_score(y_test[~thin_mask], proba[~thin_mask]) if (~thin_mask).sum() > 10 else np.nan

    results = {
        "label":        label,
        "auc_full":     round(float(auc),       4),
        "auc_thin":     round(float(auc_thin),  4),
        "auc_active":   round(float(auc_active),4),
        "ks_stat":      round(float(ks_stat),   4),
        "precision":    round(float(prec),      4),
        "recall":       round(float(rec),       4),
        "f1":           round(float(f1v),       4),
        "brier_score":  round(float(brier),     4),
        "threshold":    round(float(best_thresh),4),
    }
    log(f"  [{label:40s}] AUC={auc:.4f} | AUC_thin={auc_thin:.4f} | KS={ks_stat:.3f} | F1={f1v:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation helper
# ─────────────────────────────────────────────────────────────────────────────

def cv_auc(model, X, y, label=""):
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    log(f"  CV AUC [{label:30s}]: {scores.mean():.4f} ± {scores.std():.4f}")
    return {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_lr(spw):
    class_w = {0: 1.0, 1: float(spw)}
    return LogisticRegression(
        solver="lbfgs", max_iter=2000,
        class_weight=class_w, C=0.1,
        random_state=RANDOM_SEED
    )


def build_rf(spw):
    class_w = {0: 1.0, 1: float(spw)}
    return RandomForestClassifier(
        n_estimators=300, max_depth=10,
        min_samples_leaf=50, max_features="sqrt",
        class_weight=class_w,
        n_jobs=-1, random_state=RANDOM_SEED
    )


def build_xgb(spw, params=None):
    base = {
        "n_estimators":     500,
        "learning_rate":    0.05,
        "max_depth":        6,
        "min_child_weight": 10,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "gamma":            0.1,
        "reg_lambda":       3.0,
        "reg_alpha":        0.1,
        "scale_pos_weight": float(spw),
        "objective":        "binary:logistic",
        "tree_method":      "hist",
        "eval_metric":      "auc",
        "use_label_encoder":False,
        "random_state":     RANDOM_SEED,
        "n_jobs":           -1,
    }
    if params:
        base.update(params)
        base["scale_pos_weight"] = float(spw)
    return xgb.XGBClassifier(**base)


def build_lgb(spw, params=None):
    base = {
        "n_estimators":      500,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "max_depth":         7,
        "min_child_samples": 50,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_lambda":        3.0,
        "reg_alpha":         0.1,
        "scale_pos_weight":  float(spw),
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "verbose":           -1,
        "random_state":      RANDOM_SEED,
        "n_jobs":            -1,
    }
    if params:
        base.update(params)
        base["scale_pos_weight"] = float(spw)
    return lgb.LGBMClassifier(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 05: Model Training")
    log("=" * 60)

    data, feat, thin_mask, spw = load_data()

    # Unpack arrays
    X_train_comb        = data["X_train_comb"]
    X_test_comb         = data["X_test_comb"]
    X_train_comb_scaled = data["X_train_comb_scaled"]
    X_test_comb_scaled  = data["X_test_comb_scaled"]
    X_train_bur         = data["X_train_bur"]
    X_test_bur          = data["X_test_bur"]
    X_train_bur_scaled  = data["X_train_bur_scaled"]
    X_test_bur_scaled   = data["X_test_bur_scaled"]
    y_train             = data["y_train"]
    y_test              = data["y_test"]

    # Load tuned hyperparameters if available
    xgb_params = None
    lgb_params = None
    if os.path.exists(ARTIFACTS["best_params_xgb"]):
        with open(ARTIFACTS["best_params_xgb"]) as f:
            xgb_params = json.load(f)
        log("  Loaded Optuna-tuned XGBoost params.")
    if os.path.exists(ARTIFACTS["best_params_lgb"]):
        with open(ARTIFACTS["best_params_lgb"]) as f:
            lgb_params = json.load(f)
        log("  Loaded Optuna-tuned LightGBM params.")

    all_results = {}

    # ── LOGISTIC REGRESSION ───────────────────────────────────────────────────
    log("\n--- Logistic Regression ---")

    log("  [Bureau-Only] Fitting ...")
    lr_bur = build_lr(spw)
    lr_bur.fit(X_train_bur_scaled, y_train)
    joblib.dump(lr_bur, MODELS["lr_bureau"])
    all_results["lr_bureau"] = evaluate(lr_bur, X_test_bur_scaled, y_test, thin_mask, "LR | Bureau-Only")

    log("  [Combined] Fitting ...")
    lr_comb = build_lr(spw)
    lr_comb.fit(X_train_comb_scaled, y_train)
    joblib.dump(lr_comb, MODELS["lr_combined"])
    all_results["lr_combined"] = evaluate(lr_comb, X_test_comb_scaled, y_test, thin_mask, "LR | Combined")

    # ── RANDOM FOREST ─────────────────────────────────────────────────────────
    log("\n--- Random Forest ---")

    log("  [Bureau-Only] Fitting ...")
    rf_bur = build_rf(spw)
    rf_bur.fit(X_train_bur, y_train)
    joblib.dump(rf_bur, MODELS["rf_bureau"])
    all_results["rf_bureau"] = evaluate(rf_bur, X_test_bur, y_test, thin_mask, "RF | Bureau-Only")

    log("  [Combined] Fitting ...")
    rf_comb = build_rf(spw)
    rf_comb.fit(X_train_comb, y_train)
    joblib.dump(rf_comb, MODELS["rf_combined"])
    all_results["rf_combined"] = evaluate(rf_comb, X_test_comb, y_test, thin_mask, "RF | Combined")

    # ── XGBOOST ───────────────────────────────────────────────────────────────
    log("\n--- XGBoost ---")

    log("  [Bureau-Only] Fitting ...")
    xgb_bur = build_xgb(spw)
    xgb_bur.fit(
        X_train_bur, y_train,
        eval_set=[(X_test_bur, y_test)],
        verbose=False
    )
    joblib.dump(xgb_bur, MODELS["xgb_bureau"])
    all_results["xgb_bureau"] = evaluate(xgb_bur, X_test_bur, y_test, thin_mask, "XGB | Bureau-Only")

    log("  [Combined] Fitting ...")
    xgb_comb = build_xgb(spw, params=xgb_params)
    xgb_comb.fit(
        X_train_comb, y_train,
        eval_set=[(X_test_comb, y_test)],
        verbose=False
    )
    joblib.dump(xgb_comb, MODELS["xgb_combined"])
    all_results["xgb_combined"] = evaluate(xgb_comb, X_test_comb, y_test, thin_mask, "XGB | Combined")

    # ── LIGHTGBM ──────────────────────────────────────────────────────────────
    log("\n--- LightGBM ---")

    log("  [Bureau-Only] Fitting ...")
    lgb_bur = build_lgb(spw)
    lgb_bur.fit(
        X_train_bur, y_train,
        eval_set=[(X_test_bur, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    )
    joblib.dump(lgb_bur, MODELS["lgb_bureau"])
    all_results["lgb_bureau"] = evaluate(lgb_bur, X_test_bur, y_test, thin_mask, "LGB | Bureau-Only")

    log("  [Combined] Fitting ...")
    lgb_comb = build_lgb(spw, params=lgb_params)
    lgb_comb.fit(
        X_train_comb, y_train,
        eval_set=[(X_test_comb, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    )
    joblib.dump(lgb_comb, MODELS["lgb_combined"])
    all_results["lgb_combined"] = evaluate(lgb_comb, X_test_comb, y_test, thin_mask, "LGB | Combined")

    # ── Save CV results ────────────────────────────────────────────────────────
    with open(ARTIFACTS["cv_results"], "w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\nAll results saved to {ARTIFACTS['cv_results']}")

    # ── Print summary table ────────────────────────────────────────────────────
    log("\n" + "="*80)
    log(f"{'Model':<42} {'AUC_Full':>9} {'AUC_Thin':>9} {'KS':>7} {'F1':>7}")
    log("-"*80)
    for k, v in all_results.items():
        log(f"  {v['label']:<40} {v['auc_full']:>9.4f} {v['auc_thin']:>9.4f} "
            f"{v['ks_stat']:>7.3f} {v['f1']:>7.3f}")
    log("="*80)
    log("\nStep 05 complete.\n")


if __name__ == "__main__":
    main()
