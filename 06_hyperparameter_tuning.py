"""
06_hyperparameter_tuning.py
----------------------------
Optuna TPE hyperparameter optimisation for XGBoost and LightGBM.
Optimises 5-fold CV AUC on the training set.
Saves best parameters as JSON for use in 05_model_training.py.

Input:  outputs/04_train_test.npz
Output: outputs/06_best_params_xgb.json
        outputs/06_best_params_lgb.json
"""

import os
import sys
import json
import warnings
import numpy  as np
import joblib
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import roc_auc_score
import xgboost  as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACTS, CV_FOLDS, RANDOM_SEED, VERBOSE,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_DIRECTION,
    XGB_SEARCH_SPACE, LGB_SEARCH_SPACE
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[06_hyperparameter_tuning] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Load training data
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data():
    data = np.load(ARTIFACTS["train_test"], allow_pickle=True)
    X_train  = data["X_train_comb"]
    y_train  = data["y_train"]
    spw      = float(data["scale_pos_weight"][0])
    return X_train, y_train, spw


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost objective
# ─────────────────────────────────────────────────────────────────────────────

def xgb_objective(trial, X_train, y_train, spw):
    sp = XGB_SEARCH_SPACE
    params = {
        "n_estimators":      trial.suggest_int  ("n_estimators",     *sp["n_estimators"]),
        "learning_rate":     trial.suggest_float("learning_rate",    *sp["learning_rate"],    log=True),
        "max_depth":         trial.suggest_int  ("max_depth",        *sp["max_depth"]),
        "min_child_weight":  trial.suggest_float("min_child_weight", *sp["min_child_weight"], log=True),
        "subsample":         trial.suggest_float("subsample",        *sp["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
        "gamma":             trial.suggest_float("gamma",            *sp["gamma"]),
        "reg_lambda":        trial.suggest_float("reg_lambda",       *sp["reg_lambda"],       log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha",        *sp["reg_alpha"],        log=True),
        "scale_pos_weight":  spw,
        "objective":         "binary:logistic",
        "tree_method":       "hist",
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      RANDOM_SEED,
        "n_jobs":            -1,
    }

    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aucs  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xv = X_train[tr_idx], X_train[val_idx]
        ytr, yv = y_train[tr_idx], y_train[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xv, yv)],
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, preds))

        # Report intermediate value for pruning
        trial.report(np.mean(aucs), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(aucs))


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM objective
# ─────────────────────────────────────────────────────────────────────────────

def lgb_objective(trial, X_train, y_train, spw):
    sp = LGB_SEARCH_SPACE
    params = {
        "n_estimators":      trial.suggest_int  ("n_estimators",      *sp["n_estimators"]),
        "learning_rate":     trial.suggest_float("learning_rate",     *sp["learning_rate"],    log=True),
        "num_leaves":        trial.suggest_int  ("num_leaves",        *sp["num_leaves"]),
        "max_depth":         trial.suggest_int  ("max_depth",         *sp["max_depth"]),
        "min_child_samples": trial.suggest_int  ("min_child_samples", *sp["min_child_samples"]),
        "subsample":         trial.suggest_float("subsample",         *sp["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  *sp["colsample_bytree"]),
        "reg_lambda":        trial.suggest_float("reg_lambda",        *sp["reg_lambda"],       log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha",         *sp["reg_alpha"],        log=True),
        "scale_pos_weight":  spw,
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "verbose":           -1,
        "random_state":      RANDOM_SEED,
        "n_jobs":            -1,
    }

    skf  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xv = X_train[tr_idx], X_train[val_idx]
        ytr, yv = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xv, yv)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )
        preds = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, preds))

        trial.report(np.mean(aucs), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(aucs))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 06: Hyperparameter Optimisation (Optuna)")
    log("=" * 60)

    X_train, y_train, spw = load_train_data()
    log(f"  Training shape: {X_train.shape}  |  spw={spw:.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    log(f"\nOptimising XGBoost ({OPTUNA_N_TRIALS} trials) ...")

    xgb_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    xgb_pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1)
    xgb_study   = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        sampler=xgb_sampler,
        pruner=xgb_pruner,
    )
    xgb_study.optimize(
        lambda t: xgb_objective(t, X_train, y_train, spw),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=VERBOSE,
    )

    best_xgb = xgb_study.best_params
    best_xgb["scale_pos_weight"] = spw
    log(f"  Best XGBoost CV AUC: {xgb_study.best_value:.4f}")
    log(f"  Best params: {json.dumps(best_xgb, indent=4)}")

    with open(ARTIFACTS["best_params_xgb"], "w") as f:
        json.dump(best_xgb, f, indent=2)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    log(f"\nOptimising LightGBM ({OPTUNA_N_TRIALS} trials) ...")

    lgb_sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    lgb_pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1)
    lgb_study   = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        sampler=lgb_sampler,
        pruner=lgb_pruner,
    )
    lgb_study.optimize(
        lambda t: lgb_objective(t, X_train, y_train, spw),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=VERBOSE,
    )

    best_lgb = lgb_study.best_params
    best_lgb["scale_pos_weight"] = spw
    log(f"  Best LightGBM CV AUC: {lgb_study.best_value:.4f}")
    log(f"  Best params: {json.dumps(best_lgb, indent=4)}")

    with open(ARTIFACTS["best_params_lgb"], "w") as f:
        json.dump(best_lgb, f, indent=2)

    log("\n===== OPTIMISATION SUMMARY =====")
    log(f"  XGBoost  best CV AUC : {xgb_study.best_value:.4f}")
    log(f"  LightGBM best CV AUC : {lgb_study.best_value:.4f}")
    log(f"  Best params saved to outputs/")
    log("================================")
    log("\nStep 06 complete. Run 05_model_training.py again to use tuned params.\n")


if __name__ == "__main__":
    main()
