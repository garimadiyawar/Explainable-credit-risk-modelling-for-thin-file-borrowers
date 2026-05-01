import warnings
warnings.filterwarnings('ignore')
import shap, numpy as np, pandas as pd, json, sys, joblib, os
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude/credit_risk_pipeline')
from config import ARTIFACTS, MODELS, PLOT_DIR

model  = joblib.load(MODELS['xgb_combined'])
data   = np.load(ARTIFACTS['train_test'], allow_pickle=True)
X_te   = data['X_test_comb']
X_tr   = data['X_train_comb']
y_te   = data['y_test']
y_tr   = data['y_train']
thin   = np.load(ARTIFACTS['thin_file_mask_test'])
fnames = json.load(open(ARTIFACTS['feature_names']))['all_features']
sv     = np.load(ARTIFACTS['shap_values'])
ev_val = float(np.load(ARTIFACTS['shap_expected'])[0])
imp    = pd.read_csv(ARTIFACTS['shap_importance'])

# H5 check
alt_pfx = ('INST_','CC_','POS_','PREV_','BB_')
alt15   = sum(1 for f in imp.head(15)['feature']
               if any(f.startswith(p) for p in alt_pfx))
print(f'H5: alt features in top-15 = {alt15}  {"SUPPORTED" if alt15>=5 else "PARTIAL"}')

# SHAP stability
print('Computing SHAP stability (5-fold)...')
explainer = shap.TreeExplainer(model)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rankings = []
for fold, (_, vi) in enumerate(skf.split(X_tr, y_tr)):
    sv_f = explainer.shap_values(X_tr[vi])
    rankings.append(np.abs(sv_f).mean(0).argsort()[::-1])
    print(f'  fold {fold+1}: top = {fnames[rankings[-1][0]]}')
corrs    = [float(spearmanr(rankings[i], rankings[j])[0])
             for i in range(5) for j in range(i+1, 5)]
mean_rho = float(np.mean(corrs))
print(f'Mean Spearman rho = {mean_rho:.4f}')
print(f'H4: {"SUPPORTED" if mean_rho > 0.85 else "needs review"}')

# Beeswarm plot
print('Generating beeswarm...')
rng2  = np.random.RandomState(42)
idx_s = rng2.choice(len(X_te), min(2000, len(X_te)), replace=False)
plt.figure(figsize=(12, 9))
shap.summary_plot(sv[idx_s],
                  pd.DataFrame(X_te[idx_s], columns=fnames),
                  max_display=20, show=False)
plt.title('SHAP Summary Plot - XGBoost Combined', fontsize=13)
plt.tight_layout()
bees_path = os.path.join(PLOT_DIR, 'shap_beeswarm.png')
plt.savefig(bees_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {bees_path}')

# Waterfall plots for thin-file high-risk applicants
print('Generating waterfall plots...')
proba   = model.predict_proba(X_te)[:, 1]
tf_hr   = np.where(thin & (proba > 0.35))[0][:3]
notices = []
for rank, idx in enumerate(tf_hr, 1):
    sv_i = sv[idx]
    expl_obj = shap.Explanation(
        values=sv_i, base_values=ev_val,
        data=X_te[idx], feature_names=fnames)
    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(expl_obj, max_display=12, show=False)
    plt.title(f'SHAP Waterfall TF_{rank:03d}  pred={proba[idx]:.3f}', fontsize=11)
    plt.tight_layout()
    wf_path = os.path.join(PLOT_DIR, f'shap_waterfall_TF_{rank:03d}.png')
    plt.savefig(wf_path, dpi=150, bbox_inches='tight')
    plt.close()
    top3 = [{'rank': k+1, 'feature': fnames[j], 'shap': round(float(sv_i[j]), 5)}
             for k, j in enumerate(sv_i.argsort()[::-1][:3]) if sv_i[j] > 0]
    notices.append({'id': f'TF_{rank:03d}',
                    'pred_prob': round(float(proba[idx]), 4),
                    'reasons': top3})
    print(f'  TF_{rank:03d}  pred={proba[idx]:.3f}  top: {[r["feature"] for r in top3]}')

# Save summary
expl_out = os.path.join(os.path.dirname(ARTIFACTS['shap_values']),
                         '07_explainability_summary.json')
with open(expl_out, 'w') as f:
    json.dump({
        'shap_stability': {
            'mean_spearman': round(mean_rho, 4),
            'min_spearman':  round(min(corrs), 4),
            'max_spearman':  round(max(corrs), 4),
            'pairwise':      [round(r, 4) for r in corrs],
        },
        'adverse_notices':  notices,
        'h4_supported':     mean_rho > 0.85,
        'h5_alt_in_top15':  alt15,
    }, f, indent=2)
print(f'  Saved: {expl_out}')
print('SHAP analysis DONE.')
