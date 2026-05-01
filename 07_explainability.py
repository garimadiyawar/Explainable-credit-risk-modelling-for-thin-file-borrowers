"""
07_explainability.py
---------------------
Full explainability analysis on the best model (XGBoost Combined):
  - SHAP global importance (mean |SHAP| per feature)
  - SHAP beeswarm summary plot
  - SHAP stability across CV folds (Spearman rank correlation)
  - SHAP waterfall plots for thin-file high-risk applicants
  - SHAP interaction values (top-10 feature pairs)
  - LIME instance-level explanations + SHAP-LIME agreement (Kendall tau)
  - Partial Dependence Plots for top features (full + thin-file sub-cohort)

Input:  models/xgb_combined.joblib
        outputs/04_train_test.npz
        outputs/04_feature_names.json
        outputs/04_thin_file_mask_test.npy
Output: outputs/07_shap_values.npy
        outputs/07_shap_expected_value.npy
        outputs/07_shap_importance.csv
        plots/shap_beeswarm.png
        plots/shap_waterfall_TF_*.png
        plots/pdp_*.png
        plots/lime_TF_*.png
"""

import os
import sys
import json
import warnings
import numpy  as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular   import LimeTabularExplainer
from scipy.stats          import spearmanr, kendalltau
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection   import partial_dependence, PartialDependenceDisplay

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACTS, MODELS, PLOT_DIR, CV_FOLDS, RANDOM_SEED, VERBOSE,
    SHAP_MAX_DISPLAY, SHAP_N_LIME_SAMPLES, SHAP_TOP_INTERACTION,
    ADVERSE_ACTION_TOP_N
)

np.random.seed(RANDOM_SEED)


def log(msg):
    if VERBOSE:
        print(f"[07_explainability] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    log("Loading model and data ...")
    model     = joblib.load(MODELS["xgb_combined"])
    data      = np.load(ARTIFACTS["train_test"], allow_pickle=True)
    thin_mask = np.load(ARTIFACTS["thin_file_mask_test"])
    with open(ARTIFACTS["feature_names"]) as f:
        feat = json.load(f)

    X_train = data["X_train_comb"]
    X_test  = data["X_test_comb"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    feature_names = feat["all_features"]
    log(f"  Model loaded. Features: {len(feature_names)}")
    return model, X_train, X_test, y_train, y_test, thin_mask, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# SHAP global explanation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_global(model, X_test, feature_names):
    log("\nComputing SHAP values (TreeExplainer) ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_v  = explainer.expected_value

    log(f"  SHAP values shape: {shap_values.shape}")
    log(f"  Expected value: {expected_v:.4f}")

    # Global importance
    importance = pd.DataFrame({
        "feature":    feature_names,
        "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        "shap_mean":     shap_values.mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    log(f"\n  Top 10 features by mean |SHAP|:")
    for _, row in importance.head(10).iterrows():
        log(f"    {int(row['rank']):>3}. {row['feature']:<45} {row['shap_mean_abs']:.5f}")

    # Save
    np.save(ARTIFACTS["shap_values"],   shap_values)
    np.save(ARTIFACTS["shap_expected"], np.array([expected_v]))
    importance.to_csv(ARTIFACTS["shap_importance"], index=False)

    return shap_values, expected_v, importance


# ─────────────────────────────────────────────────────────────────────────────
# SHAP beeswarm plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_beeswarm(shap_values, X_test, feature_names):
    log("  Generating SHAP beeswarm plot ...")
    # Use a random subsample for speed
    rng   = np.random.RandomState(RANDOM_SEED)
    idx   = rng.choice(len(X_test), min(5000, len(X_test)), replace=False)
    sv_sub  = shap_values[idx]
    X_sub   = X_test[idx]
    Xdf_sub = pd.DataFrame(X_sub, columns=feature_names)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        sv_sub, Xdf_sub,
        max_display=SHAP_MAX_DISPLAY,
        show=False, plot_type="dot"
    )
    plt.title("SHAP Summary Plot (Beeswarm) — XGBoost Combined Model", fontsize=13)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "shap_beeswarm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP stability across CV folds
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_stability(model, X_train, y_train, feature_names):
    log("\nComputing SHAP stability across CV folds ...")
    skf       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rankings  = []
    explainer = shap.TreeExplainer(model)

    for fold, (_, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xv  = X_train[val_idx]
        sv  = explainer.shap_values(Xv)
        imp = np.abs(sv).mean(axis=0)
        # Rank: highest importance = rank 1
        ranking = imp.argsort()[::-1]
        rankings.append(ranking)
        log(f"    Fold {fold+1} top feature: {feature_names[ranking[0]]}")

    # Pairwise Spearman
    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            rho, _ = spearmanr(rankings[i], rankings[j])
            correlations.append(rho)

    mean_rho = np.mean(correlations)
    std_rho  = np.std(correlations)
    min_rho  = np.min(correlations)
    max_rho  = np.max(correlations)

    log(f"\n  SHAP Stability (Spearman rank correlation):")
    log(f"    Mean  : {mean_rho:.4f}")
    log(f"    Std   : {std_rho:.4f}")
    log(f"    Min   : {min_rho:.4f}")
    log(f"    Max   : {max_rho:.4f}")
    log(f"    H4 {'SUPPORTED' if mean_rho > 0.85 else 'NOT SUPPORTED'} (threshold = 0.85)")

    return {
        "mean_spearman": float(mean_rho),
        "std_spearman":  float(std_rho),
        "min_spearman":  float(min_rho),
        "max_spearman":  float(max_rho),
        "pairwise":      [float(r) for r in correlations],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP waterfall plots for thin-file high-risk applicants
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_waterfalls(model, X_test, y_test, thin_mask,
                          shap_values, expected_value, feature_names, n=5):
    log(f"\nGenerating SHAP waterfall plots for {n} thin-file high-risk applicants ...")

    proba = model.predict_proba(X_test)[:, 1]

    # Find thin-file, high-risk (predicted prob > 0.40), non-default (to avoid leakage issues)
    candidates = np.where(thin_mask & (proba > 0.40))[0]

    if len(candidates) == 0:
        log("  No thin-file high-risk applicants found.")
        return []

    selected = candidates[:n]
    adverse_notices = []

    for rank, idx in enumerate(selected, 1):
        sv_i  = shap_values[idx]
        x_i   = X_test[idx]

        # Build explanation object
        exp = shap.Explanation(
            values=sv_i,
            base_values=expected_value,
            data=x_i,
            feature_names=feature_names
        )

        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — Thin-File Applicant TF_{rank:03d} "
                  f"(Pred. Default Prob = {proba[idx]:.3f})", fontsize=11)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"shap_waterfall_TF_{rank:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out}")

        # Extract adverse action reasons (top positive SHAP contributors)
        top_pos_idx = sv_i.argsort()[::-1][:ADVERSE_ACTION_TOP_N]
        adverse_reasons = [
            {"rank": k+1, "feature": feature_names[j], "shap_value": float(sv_i[j])}
            for k, j in enumerate(top_pos_idx) if sv_i[j] > 0
        ]
        adverse_notices.append({
            "applicant_id": f"TF_{rank:03d}",
            "pred_prob":    float(proba[idx]),
            "adverse_reasons": adverse_reasons
        })

    log(f"  Adverse action notices generated for {len(adverse_notices)} applicants.")
    return adverse_notices


# ─────────────────────────────────────────────────────────────────────────────
# SHAP interaction values
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_interactions(model, X_test, feature_names, n_samples=500):
    log(f"\nComputing SHAP interaction values (n_samples={n_samples}) ...")
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sub = X_test[idx]

    explainer    = shap.TreeExplainer(model)
    shap_inter   = explainer.shap_interaction_values(X_sub)  # (n, F, F)
    mean_inter   = np.abs(shap_inter).mean(axis=0)           # (F, F)

    top_n = SHAP_TOP_INTERACTION
    top_idx = np.abs(mean_inter).sum(axis=0).argsort()[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]

    top_inter_matrix = mean_inter[np.ix_(top_idx, top_idx)]

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(top_inter_matrix, cmap="Blues")
    ax.set_xticks(range(top_n)); ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean |SHAP Interaction|")
    ax.set_title("SHAP Interaction Values — Top 10 Features", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "shap_interaction_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out}")

    # Top interaction pairs
    log("  Top 5 feature interaction pairs:")
    pairs = []
    for i in range(top_n):
        for j in range(i+1, top_n):
            pairs.append((top_names[i], top_names[j], float(top_inter_matrix[i, j])))
    pairs.sort(key=lambda x: -x[2])
    for fi, fj, val in pairs[:5]:
        log(f"    {fi:40s} × {fj:40s} : {val:.5f}")

    return top_names, top_inter_matrix


# ─────────────────────────────────────────────────────────────────────────────
# LIME instance-level explanations
# ─────────────────────────────────────────────────────────────────────────────

def compute_lime_explanations(model, X_train, X_test, y_test,
                               thin_mask, feature_names, n=10):
    log(f"\nGenerating LIME explanations for {n} thin-file high-risk applicants ...")

    proba      = model.predict_proba(X_test)[:, 1]
    candidates = np.where(thin_mask & (proba > 0.40))[0][:n]

    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["No Default", "Default"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_SEED
    )

    shap_values_all = np.load(ARTIFACTS["shap_values"])
    agreements      = []
    lime_results    = []

    for rank, idx in enumerate(candidates, 1):
        lime_exp = lime_explainer.explain_instance(
            data_row   = X_test[idx],
            predict_fn = model.predict_proba,
            num_features = 15,
            num_samples  = SHAP_N_LIME_SAMPLES
        )

        fidelity = lime_exp.score  # local R²

        # Top-5 LIME positive contributors (for default class = 1)
        lime_feats = lime_exp.as_list(label=1)
        lime_top5  = sorted(lime_feats, key=lambda x: -x[1])[:5]
        lime_names = [lf[0].split(" ")[0].replace(">", "").replace("<", "").strip()
                      for lf in lime_top5]

        # Top-5 SHAP positive contributors for same instance
        sv_i       = shap_values_all[idx]
        shap_top5_idx   = sv_i.argsort()[::-1][:5]
        shap_top5_names = [feature_names[j] for j in shap_top5_idx]

        # Kendall tau: rank agreement on feature names
        common = set(lime_names) & set(shap_top5_names)
        tau, _  = kendalltau(
            [lime_names.index(f) if f in lime_names else 5 for f in shap_top5_names],
            list(range(5))
        )

        log(f"  Applicant TF_{rank:03d}: fidelity R²={fidelity:.3f} | Kendall tau={tau:.3f} | "
            f"Common features: {len(common)}/5")

        # Save LIME bar chart
        fig = lime_exp.as_pyplot_figure(label=1)
        plt.title(f"LIME Explanation — Applicant TF_{rank:03d} "
                  f"(Pred={proba[idx]:.3f}, R²={fidelity:.3f})", fontsize=10)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"lime_TF_{rank:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()

        agreements.append(float(tau))
        lime_results.append({
            "applicant_id":  f"TF_{rank:03d}",
            "pred_prob":     float(proba[idx]),
            "fidelity_r2":   float(fidelity),
            "kendall_tau":   float(tau),
            "lime_top5":     [lf[0] for lf in lime_top5],
            "shap_top5":     shap_top5_names,
        })

    mean_tau = np.mean(agreements)
    log(f"\n  Mean Kendall tau (SHAP-LIME agreement): {mean_tau:.3f}")
    log(f"  H5 LIME agreement {'SUPPORTED' if mean_tau > 0.60 else 'NOT SUPPORTED'} (threshold=0.60)")
    return lime_results, mean_tau


# ─────────────────────────────────────────────────────────────────────────────
# Partial Dependence Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pdp(model, X_test, y_test, thin_mask, feature_names, importance_df, n_features=5):
    log(f"\nGenerating Partial Dependence Plots for top {n_features} features ...")

    top_feats = importance_df.head(n_features)["feature"].tolist()
    feat_idx  = [feature_names.index(f) for f in top_feats if f in feature_names]

    for fi, fname in zip(feat_idx, top_feats):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, (mask, label) in zip(axes, [
            (np.ones(len(X_test), dtype=bool), "Full Dataset"),
            (thin_mask, "Thin-File Sub-Cohort")
        ]):
            X_sub = X_test[mask]
            if len(X_sub) < 50:
                continue

            grid_vals = np.percentile(X_sub[:, fi], np.linspace(1, 99, 50))
            pdp_vals  = []

            for val in grid_vals:
                X_tmp     = X_sub.copy()
                X_tmp[:, fi] = val
                pdp_vals.append(model.predict_proba(X_tmp)[:, 1].mean())

            ax.plot(grid_vals, pdp_vals, lw=2, color="steelblue")
            ax.fill_between(grid_vals, pdp_vals, alpha=0.15, color="steelblue")
            ax.set_xlabel(fname, fontsize=10)
            ax.set_ylabel("Mean Predicted Default Prob.", fontsize=9)
            ax.set_title(f"PDP: {fname}\n({label})", fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Partial Dependence Plot: {fname}", fontsize=12, y=1.01)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"pdp_{fname[:30].replace('/', '_')}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("STEP 07: Explainability Analysis")
    log("=" * 60)

    model, X_train, X_test, y_train, y_test, thin_mask, feature_names = load_all()

    # 1. SHAP global
    log("\n[1/6] SHAP global importance ...")
    shap_values, expected_value, importance_df = compute_shap_global(model, X_test, feature_names)

    # 2. SHAP beeswarm
    log("\n[2/6] SHAP beeswarm plot ...")
    plot_shap_beeswarm(shap_values, X_test, feature_names)

    # 3. SHAP stability
    log("\n[3/6] SHAP stability across CV folds ...")
    stability = compute_shap_stability(model, X_train, y_train, feature_names)

    # 4. SHAP waterfall + adverse action simulation
    log("\n[4/6] SHAP waterfall plots + adverse action notices ...")
    adverse_notices = plot_shap_waterfalls(
        model, X_test, y_test, thin_mask,
        shap_values, expected_value, feature_names, n=5
    )

    # 5. SHAP interaction values
    log("\n[5/6] SHAP interaction values ...")
    try:
        compute_shap_interactions(model, X_test, feature_names)
    except Exception as e:
        log(f"  Interaction values skipped: {e}")

    # 6. LIME + agreement
    log("\n[6/6] LIME explanations ...")
    lime_results, mean_tau = compute_lime_explanations(
        model, X_train, X_test, y_test, thin_mask, feature_names, n=10
    )

    # 7. PDP
    log("\n[7/7] Partial Dependence Plots ...")
    plot_pdp(model, X_test, y_test, thin_mask, feature_names, importance_df, n_features=5)

    # Save explainability summary
    summary = {
        "shap_stability":    stability,
        "lime_mean_tau":     float(mean_tau),
        "adverse_notices":   adverse_notices,
        "lime_results":      lime_results,
    }
    out_path = os.path.join(os.path.dirname(ARTIFACTS["shap_values"]),
                            "07_explainability_summary.json")
    import json
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nExplainability summary saved to {out_path}")
    log("\nStep 07 complete.\n")


if __name__ == "__main__":
    main()
