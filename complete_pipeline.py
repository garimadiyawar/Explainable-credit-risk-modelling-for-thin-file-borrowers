"""
complete_pipeline.py
---------------------
Runs everything remaining:
  - SHAP stability H4
  - H5 alt features check  
  - SHAP beeswarm + waterfall plots
  - Fairness analysis P1, P2
  - Final results table
"""
import warnings
warnings.filterwarnings('ignore')
import sys, os, json, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr, ks_2samp

sys.path.insert(0, '/home/claude/credit_risk_pipeline')
from config import ARTIFACTS, MODELS, PLOT_DIR, RANDOM_SEED, FAIRNESS_COLS

np.random.seed(RANDOM_SEED)

# ── Load everything ──────────────────────────────────────────────────────────
print("Loading data and models...")
data    = np.load(ARTIFACTS['train_test'], allow_pickle=True)
X_te_c  = data['X_test_comb']
X_tr_c  = data['X_train_comb']
X_te_bs = data['X_test_bur_scaled']
y_te    = data['y_test']
y_tr    = data['y_train']
thin    = np.load(ARTIFACTS['thin_file_mask_test'])
fnames  = json.load(open(ARTIFACTS['feature_names']))['all_features']
sv      = np.load(ARTIFACTS['shap_values'])
ev_val  = float(np.load(ARTIFACTS['shap_expected'])[0])
imp_df  = pd.read_csv(ARTIFACTS['shap_importance'])
results = json.load(open(ARTIFACTS['cv_results']))

xgb_model = joblib.load(MODELS['xgb_combined'])
lr_model  = joblib.load(MODELS['lr_bureau'])
print(f"  Loaded. X_test shape: {X_te_c.shape}  thin={thin.sum()}")

# ── H5: Alternative features in top-15 ──────────────────────────────────────
print("\n--- H5: Alt features in top-15 SHAP ---")
alt_pfx = ('INST_', 'CC_', 'POS_', 'PREV_', 'BB_')
top15   = imp_df.head(15)['feature'].tolist()
alt15   = [f for f in top15 if any(f.startswith(p) for p in alt_pfx)]
trad15  = [f for f in top15 if f not in alt15]

print(f"  Top 15 features:")
for i, row in imp_df.head(15).iterrows():
    tag = '[ALT] ' if any(row.feature.startswith(p) for p in alt_pfx) else '[trad]'
    print(f"    {i+1:>3}. {tag} {row.feature:<44} {row.shap_mean_abs:.5f}")

print(f"\n  Alt features in top-15: {len(alt15)}")
print(f"  H5: {'SUPPORTED' if len(alt15) >= 5 else 'PARTIAL'} (need >= 5, got {len(alt15)})")


# ── H4: SHAP Stability across 5-fold CV ──────────────────────────────────────
print("\n--- H4: SHAP Stability (5-fold CV) ---")
explainer = shap.TreeExplainer(xgb_model)
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
rankings  = []

for fold, (_, vi) in enumerate(skf.split(X_tr_c, y_tr)):
    sv_fold = explainer.shap_values(X_tr_c[vi])
    rank_f  = np.abs(sv_fold).mean(axis=0).argsort()[::-1]
    rankings.append(rank_f)
    print(f"  Fold {fold+1}: top feature = {fnames[rank_f[0]]}")

corrs    = [float(spearmanr(rankings[i], rankings[j])[0])
             for i in range(5) for j in range(i+1, 5)]
mean_rho = float(np.mean(corrs))
min_rho  = float(np.min(corrs))
max_rho  = float(np.max(corrs))

print(f"\n  Pairwise Spearman rho values: {[round(r,4) for r in corrs]}")
print(f"  Mean = {mean_rho:.4f}  Min = {min_rho:.4f}  Max = {max_rho:.4f}")
print(f"  H4: {'SUPPORTED' if mean_rho > 0.85 else 'needs review'} (threshold = 0.85)")


# ── SHAP Beeswarm Plot ───────────────────────────────────────────────────────
print("\n--- SHAP Beeswarm Plot ---")
rng2    = np.random.RandomState(RANDOM_SEED)
idx_s   = rng2.choice(len(X_te_c), min(2000, len(X_te_c)), replace=False)
X_sub   = pd.DataFrame(X_te_c[idx_s], columns=fnames)
sv_sub  = sv[idx_s]

plt.figure(figsize=(12, 9))
shap.summary_plot(sv_sub, X_sub, max_display=20, show=False, plot_type="dot")
plt.title("SHAP Summary Plot (Beeswarm) — XGBoost Combined Model", fontsize=13)
plt.tight_layout()
bees_path = os.path.join(PLOT_DIR, 'shap_beeswarm.png')
plt.savefig(bees_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {bees_path}")


# ── SHAP Waterfall Plots (thin-file high-risk) ───────────────────────────────
print("\n--- SHAP Waterfall Plots ---")
proba_te = xgb_model.predict_proba(X_te_c)[:, 1]
tf_hr    = np.where(thin & (proba_te > 0.35))[0][:5]
notices  = []

for rank, idx in enumerate(tf_hr, 1):
    sv_i  = sv[idx]
    e_obj = shap.Explanation(
        values=sv_i, base_values=ev_val,
        data=X_te_c[idx], feature_names=fnames)

    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(e_obj, max_display=12, show=False)
    plt.title(f'SHAP Waterfall — Thin-File TF_{rank:03d}  '
              f'(Pred. Default Prob = {proba_te[idx]:.3f})', fontsize=11)
    plt.tight_layout()
    wf_path = os.path.join(PLOT_DIR, f'shap_waterfall_TF_{rank:03d}.png')
    plt.savefig(wf_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Extract adverse action reasons
    pos_idx  = sv_i.argsort()[::-1]
    top3     = [{'rank': k+1, 'feature': fnames[j],
                  'shap_value': round(float(sv_i[j]), 5)}
                 for k, j in enumerate(pos_idx[:3]) if sv_i[j] > 0]
    notices.append({'applicant_id': f'TF_{rank:03d}',
                    'pred_prob': round(float(proba_te[idx]), 4),
                    'adverse_reasons': top3})

    print(f"  TF_{rank:03d}  pred={proba_te[idx]:.3f}  "
          f"top reasons: {[r['feature'] for r in top3]}")

print(f"  {len(notices)} waterfall plots saved to {PLOT_DIR}/")


# ── Adverse Action Notice Display ─────────────────────────────────────────────
print("\n--- Adverse Action Notices (Regulation B format) ---")
for n in notices:
    print(f"\n  Applicant {n['applicant_id']}  (Predicted default prob: {n['pred_prob']:.3f})")
    print(f"  Principal reasons for adverse decision:")
    for r in n['adverse_reasons']:
        feat  = r['feature']
        shval = r['shap_value']
        # Plain-language translation
        translations = {
            'INST_PAYMENT_RATIO_MEAN':  'Payment amounts on prior obligations were consistently below scheduled amounts',
            'INST_DPD_MEAN':            'Prior loan instalments were frequently paid after the due date',
            'INST_SHORTPAY_COUNT':      'Multiple prior instalments were paid at less than 95% of required amount',
            'CC_UTILISATION_MEAN':      'Credit card utilisation rate is too high relative to available limit',
            'CC_DPD_COUNT':             'Credit card payment delinquencies recorded in multiple months',
            'ANNUITY_INCOME_RATIO':     'Monthly repayment obligation is too high relative to reported income',
            'CREDIT_INCOME_RATIO':      'Requested credit amount is disproportionate to reported annual income',
            'DAYS_EMPLOYED_RATIO':      'Employment tenure is insufficient relative to applicant age',
            'EXT_SOURCE_2':             'External creditworthiness score is below the required threshold',
            'EXT_SOURCE_3':             'External creditworthiness assessment indicates elevated risk',
            'AGE_YEARS':                'Applicant age corresponds to a statistically higher-risk demographic band',
        }
        reason_text = translations.get(feat, f'Feature {feat} indicates elevated default risk')
        print(f"    {r['rank']}. {reason_text}  [SHAP={shval:+.4f}]")


# ── SHAP DECOMPOSITION: Alt vs Traditional ───────────────────────────────────
print("\n--- SHAP Decomposition: Alternative vs Traditional Features ---")
all_feats   = fnames
alt_feats   = [f for f in all_feats if any(f.startswith(p) for p in alt_pfx)]
trad_feats  = [f for f in all_feats if f not in alt_feats]
bur_pfx     = ('BUREAU_', 'BB_', 'AMT_REQ_CREDIT_BUREAU')
bur_feats   = [f for f in trad_feats if any(f.startswith(p) for p in bur_pfx)]
app_feats   = [f for f in trad_feats if f not in bur_feats]

feat_idx    = {f: i for i, f in enumerate(all_feats)}

def total_shap(feat_list, sv_matrix):
    idxs = [feat_idx[f] for f in feat_list if f in feat_idx]
    return float(np.abs(sv_matrix[:, idxs]).mean()) if idxs else 0.0

def shap_pct(feat_list, sv_matrix):
    total = float(np.abs(sv_matrix).mean())
    if total == 0: return 0.0
    return 100.0 * total_shap(feat_list, sv_matrix) / total

# Full dataset
print("\n  Full dataset:")
print(f"    Application features  : {shap_pct(app_feats, sv):5.1f}%  ({len(app_feats)} features)")
print(f"    Bureau features       : {shap_pct(bur_feats, sv):5.1f}%  ({len(bur_feats)} features)")
print(f"    Alt: Instalment pmts  : {shap_pct([f for f in alt_feats if f.startswith('INST_')], sv):5.1f}%")
print(f"    Alt: Credit card      : {shap_pct([f for f in alt_feats if f.startswith('CC_')], sv):5.1f}%")
print(f"    Alt: POS Cash         : {shap_pct([f for f in alt_feats if f.startswith('POS_')], sv):5.1f}%")
print(f"    Alt: Previous apps    : {shap_pct([f for f in alt_feats if f.startswith('PREV_')], sv):5.1f}%")

# Thin-file subset
sv_thin = sv[thin]
print("\n  Thin-file sub-cohort:")
print(f"    Application features  : {shap_pct(app_feats, sv_thin):5.1f}%")
print(f"    Bureau features       : {shap_pct(bur_feats, sv_thin):5.1f}%  (structurally absent)")
print(f"    Alt: Instalment pmts  : {shap_pct([f for f in alt_feats if f.startswith('INST_')], sv_thin):5.1f}%")
print(f"    Alt: Credit card      : {shap_pct([f for f in alt_feats if f.startswith('CC_')], sv_thin):5.1f}%")


# ── Fairness Analysis ─────────────────────────────────────────────────────────
print("\n--- Fairness Analysis (P1, P2) ---")

df_raw  = pd.read_parquet(ARTIFACTS['preprocessed'])
idx_te  = data['idx_test']

lr_proba  = lr_model.predict_proba(X_te_bs)[:, 1]
xgb_proba = proba_te

LR_T  = 0.25
XGB_T = 0.30
lr_cls  = (lr_proba  >= LR_T).astype(int)
xgb_cls = (xgb_proba >= XGB_T).astype(int)

def dpd(yt, yp, grps):
    gs    = np.unique(grps)
    rates = {g: float(yp[grps == g].mean()) for g in gs if (grps == g).sum() >= 10}
    if len(rates) < 2: return 0.0
    return float(max(rates.values()) - min(rates.values()))

def eod_metric(yt, yp, grps):
    gs = np.unique(grps)
    def tpr(g):
        m = (grps == g) & (yt == 1)
        return float(yp[m].mean()) if m.sum() > 5 else 0.0
    def fpr(g):
        m = (grps == g) & (yt == 0)
        return float(yp[m].mean()) if m.sum() > 5 else 0.0
    tprs = {g: tpr(g) for g in gs}
    ref  = max(tprs, key=tprs.get)
    vals = [abs(tpr(g) - tprs[ref]) + abs(fpr(g) - fpr(ref)) for g in gs if g != ref]
    return float(max(vals)) if vals else 0.0

def bootstrap_ci(yt, yp, grps, fn, n=300):
    rng3 = np.random.RandomState(RANDOM_SEED)
    vals = []
    for _ in range(n):
        i    = rng3.choice(len(yt), len(yt), replace=True)
        vals.append(fn(yt[i], yp[i], grps[i]))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

fairness_results = []
for attr in FAIRNESS_COLS:
    src = df_raw if attr in df_raw.columns else None
    if src is None:
        print(f"  {attr}: not found, skipping")
        continue

    grps = src[attr].iloc[idx_te].fillna('Unknown').astype(str).values
    unique_g = np.unique(grps)
    print(f"\n  Attribute: {attr}  ({len(unique_g)} subgroups)")

    # Per-subgroup AUC
    for g in sorted(unique_g):
        mask = grps == g
        if mask.sum() < 20:
            continue
        n_g   = mask.sum()
        def_r = float(y_te[mask].mean())
        apr_r = float(xgb_cls[mask].mean())
        try:
            auc_g = roc_auc_score(y_te[mask], xgb_proba[mask])
        except Exception:
            auc_g = float('nan')
        print(f"    {g:<35}  n={n_g:5d}  default={def_r:.3f}  "
              f"approval={apr_r:.3f}  AUC={auc_g:.3f}")

    lr_dpd   = dpd(y_te, lr_cls,  grps)
    xgb_dpd  = dpd(y_te, xgb_cls, grps)
    lr_eod   = eod_metric(y_te, lr_cls,  grps)
    xgb_eod  = eod_metric(y_te, xgb_cls, grps)
    d_dpd    = round(xgb_dpd - lr_dpd, 4)
    d_eod    = round(xgb_eod - lr_eod, 4)

    ci_lo, ci_hi = bootstrap_ci(y_te, (xgb_cls - lr_cls).astype(float),
                                  grps, lambda a, b, g: dpd(a, b, g), n=300)
    p1 = d_dpd <= 0.02

    print(f"\n    LR  DPD={lr_dpd:.4f}  EOD={lr_eod:.4f}")
    print(f"    XGB DPD={xgb_dpd:.4f}  EOD={xgb_eod:.4f}")
    print(f"    ΔDPD={d_dpd:+.4f}  ΔEOD={d_eod:+.4f}  "
          f"95% CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    print(f"    P1: {'SUPPORTED' if p1 else 'NOT SUPPORTED'}")

    fairness_results.append({
        'attribute':    attr,
        'lr_dpd':       round(lr_dpd, 4),  'xgb_dpd': round(xgb_dpd, 4),
        'lr_eod':       round(lr_eod, 4),  'xgb_eod': round(xgb_eod, 4),
        'delta_dpd':    d_dpd,             'delta_eod': d_eod,
        'dpd_ci':       [ci_lo, ci_hi],
        'p1_supported': p1,
    })

with open(ARTIFACTS['fairness_results'], 'w') as f:
    json.dump(fairness_results, f, indent=2)
print(f"\n  Fairness results saved: {ARTIFACTS['fairness_results']}")


# ── SAVE EXPLAINABILITY SUMMARY ──────────────────────────────────────────────
expl_out = os.path.join(os.path.dirname(ARTIFACTS['shap_values']),
                         '07_explainability_summary.json')
with open(expl_out, 'w') as f:
    json.dump({
        'shap_stability': {
            'mean_spearman': round(mean_rho, 4),
            'min_spearman':  round(min_rho,  4),
            'max_spearman':  round(max_rho,  4),
            'pairwise':      [round(r, 4) for r in corrs],
        },
        'h4_supported':     mean_rho > 0.85,
        'h5_alt_in_top15':  len(alt15),
        'h5_supported':     len(alt15) >= 5,
        'adverse_notices':  notices,
    }, f, indent=2)
print(f"\n  Explainability summary saved: {expl_out}")


# ── FINAL HYPOTHESIS SUMMARY ─────────────────────────────────────────────────
print("\n" + "="*75)
print("COMPLETE HYPOTHESIS EVALUATION SUMMARY")
print("="*75)

lr_t    = results['lr_bureau']['auc_thin']
xgb_t   = results['xgb_combined']['auc_thin']
xgb_bt  = results['xgb_bureau']['auc_thin']
lr_a    = results['lr_bureau']['auc_active']
xgb_a   = results['xgb_combined']['auc_active']

h1 = xgb_t - lr_t   > 0.05
h2 = xgb_t - xgb_bt > 0.02
h3 = (xgb_t - lr_t) > (xgb_a - lr_a)
h4 = mean_rho        > 0.85
h5 = len(alt15)      >= 5
p1 = all(r['p1_supported'] for r in fairness_results) if fairness_results else False

print(f"\n  H1  XGB Combined vs LR Bureau-Only (thin-file AUC)")
print(f"      ΔAUC = {xgb_t-lr_t:+.4f}  {'✓ SUPPORTED' if h1 else '✗ NEEDS REVIEW'}")
print(f"\n  H2  Alternative feature incremental value")
print(f"      ΔAUC = {xgb_t-xgb_bt:+.4f}  {'✓ SUPPORTED' if h2 else '✗ NEEDS REVIEW'}")
print(f"\n  H3  Thin-file gain > Credit-active gain")
print(f"      thin={xgb_t-lr_t:.4f} vs active={xgb_a-lr_a:.4f}  {'✓ SUPPORTED' if h3 else '✗ NEEDS REVIEW'}")
print(f"\n  H4  SHAP stability (mean Spearman rho > 0.85)")
print(f"      rho = {mean_rho:.4f}  {'✓ SUPPORTED' if h4 else '✗ NEEDS REVIEW'}")
print(f"\n  H5  Alternative features in top-15 SHAP (need >= 5)")
print(f"      count = {len(alt15)}  {'✓ SUPPORTED' if h5 else '✗ NEEDS REVIEW'}")
for r in fairness_results:
    print(f"\n  P1  Fairness non-deterioration [{r['attribute']}]")
    print(f"      ΔDPD={r['delta_dpd']:+.4f}  {'✓ SUPPORTED' if r['p1_supported'] else '✗ NOT SUPPORTED'}")

supported = sum([h1,h2,h3,h4,h5,p1])
print(f"\n  Overall: {supported}/6 hypotheses/propositions supported")

print("\n" + "="*75)
print("OUTPUT FILES:")
print(f"  models/           — 8 trained models (.joblib)")
print(f"  outputs/05_cv_results.json          — all AUC/KS/F1 metrics")
print(f"  outputs/07_shap_values.npy          — SHAP values (n_test x n_feat)")
print(f"  outputs/07_shap_importance.csv      — ranked feature importances")
print(f"  outputs/07_explainability_summary.json")
print(f"  outputs/08_fairness_results.json")
print(f"  plots/shap_beeswarm.png")
print(f"  plots/shap_waterfall_TF_00*.png     — {len(notices)} waterfall plots")
print("="*75)
print("\nPIPELINE COMPLETE.")
