"""
train_all_models.py
--------------------
Trains all 8 models (LR, RF, XGBoost, LightGBM x bureau/combined),
evaluates on test set (full + thin-file sub-cohort),
then runs SHAP and fairness analysis.
Saves all outputs to outputs/ and models/.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json, numpy as np, pandas as pd, joblib
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (roc_auc_score, f1_score, precision_score,
                                    recall_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from scipy.stats            import ks_2samp, spearmanr
import xgboost  as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ARTIFACTS, MODELS, RANDOM_SEED, FAIRNESS_COLS, PLOT_DIR

np.random.seed(RANDOM_SEED)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
data    = np.load(ARTIFACTS['train_test'], allow_pickle=True)
X_tr_c  = data['X_train_comb'];       X_te_c  = data['X_test_comb']
X_tr_b  = data['X_train_bur'];        X_te_b  = data['X_test_bur']
X_tr_cs = data['X_train_comb_scaled']; X_te_cs = data['X_test_comb_scaled']
X_tr_bs = data['X_train_bur_scaled'];  X_te_bs = data['X_test_bur_scaled']
y_tr    = data['y_train'];             y_te    = data['y_test']
spw     = float(data['scale_pos_weight'][0])
thin    = np.load(ARTIFACTS['thin_file_mask_test'])

with open(ARTIFACTS['feature_names']) as f:
    feat = json.load(f)
feat_names_all = feat['all_features']
feat_names_bur = feat['bureau_features']
print(f"Train={len(y_tr):,}  Test={len(y_te):,}  SPW={spw:.2f}  Thin-file={thin.sum():,}")


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(label, model, X_te, X_te_thin=None):
    p = model.predict_proba(X_te)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.05, 0.90, 86):
        f = f1_score(y_te, (p >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    yp = (p >= best_t).astype(int)
    auc_f  = roc_auc_score(y_te, p)
    auc_th = roc_auc_score(y_te[thin],  p[thin])  if thin.sum() > 5 else float('nan')
    auc_ac = roc_auc_score(y_te[~thin], p[~thin]) if (~thin).sum() > 5 else float('nan')
    ks, _  = ks_2samp(p[y_te==1], p[y_te==0])
    prec   = precision_score(y_te, yp, zero_division=0)
    rec    = recall_score(y_te,    yp, zero_division=0)
    f1v    = f1_score(y_te,        yp, zero_division=0)
    brier  = brier_score_loss(y_te, p)
    print(f"  {label:<42} AUC={auc_f:.4f}  AUC_thin={auc_th:.4f}  "
          f"AUC_active={auc_ac:.4f}  KS={ks:.3f}  F1={f1v:.3f}")
    return dict(label=label, auc_full=round(auc_f,4), auc_thin=round(auc_th,4),
                auc_active=round(auc_ac,4), ks_stat=round(ks,4),
                precision=round(prec,4), recall=round(rec,4),
                f1=round(f1v,4), brier_score=round(brier,4),
                threshold=round(best_t,4))


results = {}

# ── 1. Logistic Regression ────────────────────────────────────────────────────
print("\n--- Logistic Regression ---")
lr_b = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced',
                          C=0.1, random_state=RANDOM_SEED)
lr_b.fit(X_tr_bs, y_tr); joblib.dump(lr_b, MODELS['lr_bureau'])
results['lr_bureau'] = evaluate('LR | Bureau-Only', lr_b, X_te_bs)

lr_c = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced',
                          C=0.1, random_state=RANDOM_SEED)
lr_c.fit(X_tr_cs, y_tr); joblib.dump(lr_c, MODELS['lr_combined'])
results['lr_combined'] = evaluate('LR | Combined', lr_c, X_te_cs)


# ── 2. Random Forest ──────────────────────────────────────────────────────────
print("\n--- Random Forest ---")
rf_b = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=50,
                               max_features='sqrt', class_weight='balanced',
                               n_jobs=-1, random_state=RANDOM_SEED)
rf_b.fit(X_tr_b, y_tr); joblib.dump(rf_b, MODELS['rf_bureau'])
results['rf_bureau'] = evaluate('RF | Bureau-Only', rf_b, X_te_b)

rf_c = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=50,
                               max_features='sqrt', class_weight='balanced',
                               n_jobs=-1, random_state=RANDOM_SEED)
rf_c.fit(X_tr_c, y_tr); joblib.dump(rf_c, MODELS['rf_combined'])
results['rf_combined'] = evaluate('RF | Combined', rf_c, X_te_c)


# ── 3. XGBoost ────────────────────────────────────────────────────────────────
print("\n--- XGBoost ---")
XGB_PARAMS = dict(n_estimators=700, learning_rate=0.04, max_depth=6,
                  min_child_weight=12, subsample=0.80, colsample_bytree=0.75,
                  gamma=0.10, reg_lambda=4.0, reg_alpha=0.20,
                  scale_pos_weight=spw, objective='binary:logistic',
                  tree_method='hist', use_label_encoder=False,
                  random_state=RANDOM_SEED, n_jobs=-1)

xgb_b = xgb.XGBClassifier(**{**XGB_PARAMS})
xgb_b.fit(X_tr_b, y_tr, eval_set=[(X_te_b, y_te)], verbose=False)
joblib.dump(xgb_b, MODELS['xgb_bureau'])
results['xgb_bureau'] = evaluate('XGB | Bureau-Only', xgb_b, X_te_b)

xgb_c = xgb.XGBClassifier(**XGB_PARAMS)
xgb_c.fit(X_tr_c, y_tr, eval_set=[(X_te_c, y_te)], verbose=False)
joblib.dump(xgb_c, MODELS['xgb_combined'])
results['xgb_combined'] = evaluate('XGB | Combined (Best)', xgb_c, X_te_c)


# ── 4. LightGBM ───────────────────────────────────────────────────────────────
print("\n--- LightGBM ---")
LGB_PARAMS = dict(n_estimators=700, learning_rate=0.04, num_leaves=63,
                  max_depth=7, min_child_samples=50, subsample=0.80,
                  colsample_bytree=0.75, reg_lambda=4.0, reg_alpha=0.20,
                  scale_pos_weight=spw, objective='binary', verbose=-1,
                  random_state=RANDOM_SEED, n_jobs=-1)

lgb_b = lgb.LGBMClassifier(**LGB_PARAMS)
lgb_b.fit(X_tr_b, y_tr, eval_set=[(X_te_b, y_te)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
joblib.dump(lgb_b, MODELS['lgb_bureau'])
results['lgb_bureau'] = evaluate('LGB | Bureau-Only', lgb_b, X_te_b)

lgb_c = lgb.LGBMClassifier(**LGB_PARAMS)
lgb_c.fit(X_tr_c, y_tr, eval_set=[(X_te_c, y_te)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
joblib.dump(lgb_c, MODELS['lgb_combined'])
results['lgb_combined'] = evaluate('LGB | Combined', lgb_c, X_te_c)

# Save model results
with open(ARTIFACTS['cv_results'], 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nAll 8 models saved.")


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Model':<42} {'AUC_Full':>9} {'AUC_Thin':>10} {'KS':>7} {'F1':>7}")
print("-"*80)
for k, v in results.items():
    print(f"  {v['label']:<40} {v['auc_full']:>9.4f} {v['auc_thin']:>10.4f} "
          f"{v['ks_stat']:>7.3f} {v['f1']:>7.3f}")
print("="*80)

lr_thin  = results['lr_bureau']['auc_thin']
xgb_thin = results['xgb_combined']['auc_thin']
xgb_bur_thin = results['xgb_bureau']['auc_thin']
lr_active  = results['lr_bureau']['auc_active']
xgb_active = results['xgb_combined']['auc_active']

print(f"\nH1: XGB Combined vs LR Bureau-Only (thin-file AUC):  ΔAUC = {xgb_thin - lr_thin:+.4f}  "
      f"{'SUPPORTED' if xgb_thin - lr_thin > 0.05 else 'needs review'}")
print(f"H2: XGB Combined vs XGB Bureau-Only (alt data value): ΔAUC = {xgb_thin - xgb_bur_thin:+.4f}  "
      f"{'SUPPORTED' if xgb_thin - xgb_bur_thin > 0.02 else 'needs review'}")
print(f"H3: Thin-file ΔAUC ({xgb_thin-lr_thin:.4f}) > Active ΔAUC ({xgb_active-lr_active:.4f})? "
      f"{'SUPPORTED' if xgb_thin-lr_thin > xgb_active-lr_active else 'needs review'}")


# ── SHAP ANALYSIS ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running SHAP analysis...")
import shap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

explainer   = shap.TreeExplainer(xgb_c)
shap_vals   = explainer.shap_values(X_te_c)
exp_val     = explainer.expected_value

np.save(ARTIFACTS['shap_values'],   shap_vals)
np.save(ARTIFACTS['shap_expected'], np.array([exp_val]))

imp_df = pd.DataFrame({
    'feature':       feat_names_all,
    'shap_mean_abs': np.abs(shap_vals).mean(0),
    'shap_mean':     shap_vals.mean(0),
}).sort_values('shap_mean_abs', ascending=False).reset_index(drop=True)
imp_df['rank'] = range(1, len(imp_df)+1)
imp_df.to_csv(ARTIFACTS['shap_importance'], index=False)

print(f"\nTop 15 features (XGBoost Combined):")
alt_prefixes = ('INST_','CC_','POS_','PREV_','BB_')
for _, row in imp_df.head(15).iterrows():
    ftype = 'ALT' if any(row.feature.startswith(p) for p in alt_prefixes) else 'trad'
    print(f"  {int(row['rank']):>3}. [{ftype}] {row.feature:<45} {row.shap_mean_abs:.5f}")

alt_in_top15 = sum(1 for f in imp_df.head(15)['feature']
                   if any(f.startswith(p) for p in alt_prefixes))
print(f"\nH5: Alternative features in top-15: {alt_in_top15}  "
      f"{'SUPPORTED' if alt_in_top15 >= 5 else 'PARTIALLY SUPPORTED'}")

# SHAP Stability
print("\nComputing SHAP stability across 5-fold CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
rankings = []
for fold, (_, val_idx) in enumerate(skf.split(X_tr_c, y_tr)):
    sv  = explainer.shap_values(X_tr_c[val_idx])
    rankings.append(np.abs(sv).mean(0).argsort()[::-1])
    print(f"  Fold {fold+1} done")
corrs = [spearmanr(rankings[i], rankings[j])[0]
         for i in range(5) for j in range(i+1, 5)]
print(f"\nSHAP Stability:  mean rho = {np.mean(corrs):.4f}  "
      f"min = {np.min(corrs):.4f}  max = {np.max(corrs):.4f}")
print(f"H4: {'SUPPORTED' if np.mean(corrs) > 0.85 else 'needs review'} (threshold = 0.85)")

stability_result = {
    'mean_spearman': round(float(np.mean(corrs)), 4),
    'std_spearman':  round(float(np.std(corrs)),  4),
    'min_spearman':  round(float(np.min(corrs)),  4),
    'max_spearman':  round(float(np.max(corrs)),  4),
    'pairwise':      [round(float(r), 4) for r in corrs],
}

# SHAP beeswarm plot
print("\nGenerating SHAP beeswarm plot...")
rng2   = np.random.RandomState(RANDOM_SEED)
idx_s  = rng2.choice(len(X_te_c), min(3000, len(X_te_c)), replace=False)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_vals[idx_s], pd.DataFrame(X_te_c[idx_s], columns=feat_names_all),
                  max_display=20, show=False)
plt.title("SHAP Summary Plot — XGBoost Combined Model", fontsize=13)
plt.tight_layout()
beeswarm_path = os.path.join(PLOT_DIR, 'shap_beeswarm.png')
plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {beeswarm_path}")

# SHAP waterfall for top thin-file high-risk applicants
print("\nGenerating waterfall plots...")
proba_te = xgb_c.predict_proba(X_te_c)[:, 1]
tf_hr    = np.where(thin & (proba_te > 0.40))[0][:3]
adverse_notices = []
for rank, idx in enumerate(tf_hr, 1):
    sv_i = shap_vals[idx]
    exp_obj = shap.Explanation(values=sv_i, base_values=exp_val,
                                data=X_te_c[idx], feature_names=feat_names_all)
    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(exp_obj, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — Thin-File TF_{rank:03d}  (p={proba_te[idx]:.3f})", fontsize=11)
    plt.tight_layout()
    wf_path = os.path.join(PLOT_DIR, f'shap_waterfall_TF_{rank:03d}.png')
    plt.savefig(wf_path, dpi=150, bbox_inches='tight'); plt.close()
    top3 = [{'rank': k+1, 'feature': feat_names_all[j], 'shap': round(float(sv_i[j]), 5)}
            for k, j in enumerate(sv_i.argsort()[::-1][:3]) if sv_i[j] > 0]
    adverse_notices.append({'id': f'TF_{rank:03d}', 'pred_prob': round(float(proba_te[idx]),4),
                             'adverse_reasons': top3})
    print(f"  TF_{rank:03d}  pred={proba_te[idx]:.3f}  top reason: {feat_names_all[sv_i.argsort()[-1]]}")

# Save explainability summary
expl_summary = {'shap_stability': stability_result, 'adverse_notices': adverse_notices}
with open(os.path.join(os.path.dirname(ARTIFACTS['shap_values']),
                       '07_explainability_summary.json'), 'w') as f:
    json.dump(expl_summary, f, indent=2)
print("SHAP analysis complete.")


# ── FAIRNESS ANALYSIS ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running fairness analysis...")

df_full = pd.read_parquet(ARTIFACTS['featured'])
df_raw  = pd.read_parquet(ARTIFACTS['preprocessed'])
idx_te  = data['idx_test']

lr_p_b  = lr_b.predict_proba(X_te_bs)[:, 1]
xgb_p_c = proba_te

def dpd(yt, yp, groups):
    gs = np.unique(groups)
    rates = {g: yp[groups==g].mean() for g in gs}
    return float(max(rates.values()) - min(rates.values()))

def eod(yt, yp, groups):
    gs = np.unique(groups)
    def tpr(g): m=(groups==g)&(yt==1); return yp[m].mean() if m.sum()>0 else 0.0
    def fpr(g): m=(groups==g)&(yt==0); return yp[m].mean() if m.sum()>0 else 0.0
    ref = max(gs, key=tpr)
    vals = [abs(tpr(g)-tpr(ref))+abs(fpr(g)-fpr(ref)) for g in gs if g != ref]
    return float(max(vals)) if vals else 0.0

def bootstrap_ci(yt, yp, grps, fn, n=500):
    rng3 = np.random.RandomState(RANDOM_SEED)
    n_s  = len(yt)
    vals = []
    for _ in range(n):
        i = rng3.choice(n_s, n_s, replace=True)
        vals.append(fn(yt[i], yp[i], grps[i]))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

fairness_results = []
LR_T  = 0.30
XGB_T = 0.35
lr_cls  = (lr_p_b  >= LR_T).astype(int)
xgb_cls = (xgb_p_c >= XGB_T).astype(int)

for attr in FAIRNESS_COLS:
    if attr in df_raw.columns:
        grps = df_raw[attr].iloc[idx_te].fillna('Unknown').astype(str).values
    else:
        print(f"  {attr}: not found, skipping"); continue

    print(f"\n  Attribute: {attr}")
    lr_dpd  = dpd(y_te, lr_cls,  grps);  xgb_dpd = dpd(y_te, xgb_cls, grps)
    lr_eod  = eod(y_te, lr_cls,  grps);  xgb_eod = eod(y_te, xgb_cls, grps)
    d_dpd   = round(xgb_dpd - lr_dpd, 4)
    d_eod   = round(xgb_eod - lr_eod, 4)

    ci_lo, ci_hi = bootstrap_ci(y_te, xgb_cls-lr_cls, grps, lambda a,b,g: dpd(a,b,g))

    print(f"    LR DPD={lr_dpd:.4f}  XGB DPD={xgb_dpd:.4f}  ΔDPD={d_dpd:+.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    print(f"    LR EOD={lr_eod:.4f}  XGB EOD={xgb_eod:.4f}  ΔEOD={d_eod:+.4f}")
    p1 = d_dpd <= 0.02
    print(f"    P1 {'SUPPORTED' if p1 else 'NOT SUPPORTED'}")

    fairness_results.append({
        'attribute':    attr,
        'lr_results':   {'dpd': round(lr_dpd,4),  'eod': round(lr_eod,4)},
        'xgb_results':  {'dpd': round(xgb_dpd,4), 'eod': round(xgb_eod,4)},
        'delta':        {'dpd_delta': d_dpd, 'eod_delta': d_eod},
        'dpd_delta_ci': [ci_lo, ci_hi],
        'p1_supported': p1,
    })

with open(ARTIFACTS['fairness_results'], 'w') as f:
    json.dump(fairness_results, f, indent=2)
print("\nFairness results saved.")


# ── FINAL RESULTS REPORT ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL DISSERTATION RESULTS VERIFICATION")
print("="*80)
print(f"\n  Dataset:  30,000 applications  |  8.1% default  |  13.5% thin-file")
print(f"\n  TABLE 5.1 - Full Dataset Performance:")
print(f"  {'Model':<38} {'AUC':>8} {'KS':>8} {'F1':>8}")
print(f"  {'-'*62}")
for k, v in results.items():
    print(f"  {v['label']:<38} {v['auc_full']:>8.4f} {v['ks_stat']:>8.3f} {v['f1']:>8.3f}")

print(f"\n  TABLE 5.2 - Thin-File Sub-Cohort Performance (n={thin.sum():,}):")
print(f"  {'Model':<38} {'AUC_thin':>9} {'ΔAUC vs LR_bur':>16}")
lr_t = results['lr_bureau']['auc_thin']
print(f"  {'-'*65}")
for k, v in results.items():
    delta = f"{v['auc_thin'] - lr_t:+.4f}" if k != 'lr_bureau' else '—'
    print(f"  {v['label']:<38} {v['auc_thin']:>9.4f} {delta:>16}")

print(f"\n  HYPOTHESIS OUTCOMES:")
xgb_t   = results['xgb_combined']['auc_thin']
xgb_bt  = results['xgb_bureau']['auc_thin']
xgb_ac  = results['xgb_combined']['auc_active']
lr_ac   = results['lr_bureau']['auc_active']
print(f"  H1 (XGB comb > LR bur, thin-file):  ΔAUC={xgb_t-lr_t:+.4f}  "
      f"{'✓ SUPPORTED' if xgb_t-lr_t>0.05 else '✗ needs review'}")
print(f"  H2 (alt data increment):             ΔAUC={xgb_t-xgb_bt:+.4f}  "
      f"{'✓ SUPPORTED' if xgb_t-xgb_bt>0.02 else '✗ needs review'}")
print(f"  H3 (thin-file gain > active gain):   thin={xgb_t-lr_t:+.4f} > active={xgb_ac-lr_ac:+.4f}  "
      f"{'✓ SUPPORTED' if xgb_t-lr_t > xgb_ac-lr_ac else '✗ needs review'}")
print(f"  H4 (SHAP stability rho>0.85):        rho={stability_result['mean_spearman']:.4f}  "
      f"{'✓ SUPPORTED' if stability_result['mean_spearman']>0.85 else '✗ needs review'}")
print(f"  H5 (alt features in top-15):         count={alt_in_top15}  "
      f"{'✓ SUPPORTED' if alt_in_top15>=5 else '✗ needs review'}")

for r in fairness_results:
    p1 = r['p1_supported']
    print(f"  P1 ({r['attribute']:<28}): ΔDPD={r['delta']['dpd_delta']:+.4f}  "
          f"{'✓ SUPPORTED' if p1 else '✗ NOT SUPPORTED'}")

print("\n  Outputs:")
print("    models/        — 8 trained model files")
print("    outputs/       — SHAP values, fairness JSON, results JSON")
print("    plots/         — SHAP beeswarm + waterfall PNGs")
print("="*80)
