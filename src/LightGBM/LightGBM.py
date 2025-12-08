# Install required Python packages (run in Colab)
# pip install -q lightgbm shap optuna joblib pandas numpy scikit-learn pyarrow matplotlib

# Basic imports and seed
import os, random
import numpy as np, pandas as pd
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Setup complete. Python version:", os.sys.version.split("\n")[0])


# Mount Google Drive (uncomment when running in Colab)
from google.colab import drive
drive.mount('/content/drive')

# Set base path in Drive where dataset and outputs will be stored.
DRIVE_BASE = '/content/drive/MyDrive/IGOM_ML'  # change as needed
os.makedirs(DRIVE_BASE, exist_ok=True)
print("DRIVE_BASE set to:", DRIVE_BASE)


# %%

# Load dataset (Parquet recommended). If you uploaded directly to session, change the path.
import os
default_path = os.path.join(DRIVE_BASE, 'fused_replay.parquet')
if os.path.exists(default_path):
    df = pd.read_parquet(default_path)
    print("Loaded fused_replay.parquet from Drive, rows:", len(df))
else:
    print("No dataset at", default_path)
    # Fallback: create a tiny synthetic dataset for demo purposes (remove for real runs)
    print("Creating synthetic demo dataset (for code flow only). Replace with your real fused_replay.parquet")
    n = 2000
    rng = np.random.RandomState(SEED)
    df = pd.DataFrame({
        'cdm_id': np.arange(n),
        'object_A_id': rng.randint(1000,2000,n),
        'object_B_id': rng.randint(2000,3000,n),
        'object_pair_id': [f"{a}_{b}" for a,b in zip(rng.randint(1000,2000,n), rng.randint(2000,3000,n))],
        'timestamp': pd.Timestamp('2025-01-01') + pd.to_timedelta(rng.randint(0,86400*30,n), unit='s'),
        # physics-like features
        'range': rng.uniform(50,5000,n),  # km
        'rel_vx': rng.normal(0,0.5,n),
        'rel_vy': rng.normal(0,0.5,n),
        'rel_vz': rng.normal(0,0.2,n),
        'radar_snr': rng.uniform(0,30,n),
        'num_tracks': rng.randint(1,8,n),
        'catalog_age_hours': rng.uniform(0,720,n),
        # target (rare events)
        'collision_label': (rng.rand(n) < 0.02).astype(int)
    })
    print("Synthetic dataset created. Positive rate:", df['collision_label'].mean())


# %%

# Basic feature engineering examples. Extend these based on your available fields.
df['rel_speed'] = np.sqrt(df['rel_vx']**2 + df['rel_vy']**2 + df['rel_vz']**2)
# convert range from km to meters for intuition
df['range_m'] = df['range'] * 1000.0
# interaction features
df['range_over_tracks'] = df['range_m'] / (df['num_tracks'] + 1e-6)
df['dist_times_snr'] = df['range_m'] * (df['radar_snr'] + 1e-6)
# temporal features
df['hour'] = df['timestamp'].dt.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

# Fill missing values and choose features
df.fillna(-999, inplace=True)
feature_cols = ['range_m','rel_speed','radar_snr','num_tracks','catalog_age_hours',
                'range_over_tracks','dist_times_snr','hour_sin','hour_cos']
target_col = 'collision_label'

print("Features ready. Example row:")
display(df[feature_cols + [target_col]].head())


# %%

# Train LightGBM with GroupKFold to avoid leakage across object pairs.
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

X = df[feature_cols]
y = df[target_col].astype(int)
groups = df['object_pair_id']

gkf = GroupKFold(n_splits=5)
models = []
val_preds = []
val_trues = []
fold = 0
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    fold += 1
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        'objective':'binary',
        'metric':'auc',
        'learning_rate':0.05,
        'num_leaves':63,
        'max_depth':7,
        'min_data_in_leaf':20,
        'feature_fraction':0.8,
        'bagging_fraction':0.8,
        'bagging_freq':5,
        'seed': SEED + fold,
        'verbosity': -1
    }
    bst = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=2000, early_stopping_rounds=100, verbose_eval=100)
    p_val = bst.predict(X_val, num_iteration=bst.best_iteration)
    print(f"Fold {fold} ROC-AUC: {roc_auc_score(y_val, p_val):.4f}, PR-AUC: {average_precision_score(y_val, p_val):.4f}, Brier: {brier_score_loss(y_val, p_val):.4f}")
    models.append(bst)
    val_preds.append(pd.Series(p_val, index=val_idx))
    val_trues.append(pd.Series(y_val.values, index=val_idx))

# aggregate OOF predictions
import pandas as pd
oof_pred = pd.concat(val_preds).sort_index()
oof_true = pd.concat(val_trues).sort_index()
print("\nOverall OOF ROC-AUC:", roc_auc_score(oof_true, oof_pred))
print("Overall OOF PR-AUC:", average_precision_score(oof_true, oof_pred))


# %%

# Probability calibration using sklearn's CalibratedClassifierCV on a pooled LightGBM wrapper.
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import joblib

# Create a wrapper classifier that averages predictions from the LightGBM models
class LGBEnsembleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        preds = np.column_stack([m.predict(X, num_iteration=m.best_iteration) for m in self.models])
        mean = preds.mean(axis=1)
        # return two-column probability matrix
        return np.vstack([1-mean, mean]).T

# Fit calibrator on a holdout split (here we reuse a fraction of the dataset)
from sklearn.model_selection import train_test_split
X_train_cal, X_cal, y_train_cal, y_cal = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
ens = LGBEnsembleWrapper(models)
calibrator = CalibratedClassifierCV(base_estimator=ens, method='isotonic', cv='prefit')
# CalibratedClassifierCV with cv='prefit' expects base estimator already "fit" - our wrapper simply uses models
calibrator.fit(X_cal, y_cal)
# Evaluate calibration
probs_cal = calibrator.predict_proba(X_cal)[:,1]
from sklearn.metrics import brier_score_loss
print("Calibration set Brier score:", brier_score_loss(y_cal, probs_cal))
# Save calibrator and wrapper
artifact = {'models': models, 'features': feature_cols, 'calibrator': calibrator}
joblib.dump(artifact, os.path.join(DRIVE_BASE, 'lgb_ensemble_artifact.joblib'))
print("Saved artifact to Drive.")


# %%

# SHAP explainability for one model (tree explainer). Use ensemble average for robustness.
import shap
import matplotlib.pyplot as plt

# Use the first model's TreeExplainer for speed. For production, aggregate SHAP across ensemble or use linear approximation.
explainer = shap.TreeExplainer(models[0])
sample_idx = X.sample(min(100, len(X)), random_state=SEED).index
shap_vals = explainer.shap_values(X.loc[sample_idx])
print("Displaying SHAP summary plot (may take a moment)...")
shap.summary_plot(shap_vals, X.loc[sample_idx], show=True)

# Example: generate one-line reason for a single row
def one_line_reason(row, shap_vals_row, feature_names, topk=3):
    idxs = np.argsort(-np.abs(shap_vals_row))[:topk]
    parts = []
    for i in idxs:
        fname = feature_names[i]
        val = row[fname]
        direction = "high" if shap_vals_row[i] > 0 else "low"
        parts.append(f"{direction} {fname}={val:.2f}")
    return "Impact: " + ", ".join(parts)

# demo
i = X.sample(1, random_state=SEED).index[0]
row = X.loc[i]
shap_row = explainer.shap_values(row.values.reshape(1,-1))[0]
print("One-line reason example:", one_line_reason(row, shap_row, feature_cols, topk=3))


# %%

# Demo prediction API-like function using saved artifact
import joblib
art = joblib.load(os.path.join(DRIVE_BASE, 'lgb_ensemble_artifact.joblib'))
models_loaded = art['models']
features_loaded = art['features']
calib = art['calibrator']

def predict_event(feature_dict):
    # feature_dict maps feature name -> value
    x = pd.DataFrame([feature_dict])[features_loaded].fillna(-999)
    # ensemble mean and std
    preds = np.column_stack([m.predict(x, num_iteration=m.best_iteration) for m in models_loaded])
    prob_mean = preds.mean(axis=1)[0]
    prob_std = preds.std(axis=1)[0]
    prob_cal = calib.predict_proba(x)[:,1][0]
    return {'prob_mean': float(prob_mean), 'prob_std': float(prob_std), 'prob_calibrated': float(prob_cal)}

# Demo with a random row
sample = X.sample(1, random_state=SEED).iloc[0].to_dict()
print("Demo prediction:", predict_event(sample))



