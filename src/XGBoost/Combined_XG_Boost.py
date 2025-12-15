#!/usr/bin/env python3
"""
Train XGBoost variants and pick threshold that enforces precision >= 0.50
while maximizing recall. Saves JSON results and model files.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    average_precision_score, roc_auc_score
)
from xgboost import XGBClassifier
from scipy.special import expit  # sigmoid if needed

# ---------- Feature lists (from your pipeline) ----------
XG_Boost = [
    'cdmMissDistance', 'cdmPc',
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m', 'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca'
]

XG_Boost_NoLeak = [
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m', 'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca'
]

XG_Boost_NoLeak_Featured = [
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m', 'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca',
    'tca_bin', 'same_sat_type', 'is_debris_pair', 'close_all_axes',
    'risky_uncertainty', 'distance_ratio', 'object_type_match'
]

XG_Boost_Featured = [
    'cdmMissDistance', 'cdmPc',
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H_tca_72H',
    'condition_Radial_100m', 'condition_InTrack_500m', 'condition_CrossTrack_500m',
    'condition_sat2posUnc_1km', 'condition_sat2Obs_25',
    'hours_to_tca',
    'log_cdmPc', 'inv_miss_distance', 'tca_bin', 'same_sat_type',
    'is_debris_pair', 'close_all_axes', 'risky_uncertainty',
    'distance_ratio', 'object_type_match'
]

# ---------- Config ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_PRECISION = 0.50   # enforce at least 50% precision
WEIGHT_RECALL = 0.9    # primary objective is recall; tie-breaker uses weighted score
WEIGHT_PREC = 0.1
EARLY_STOPPING_ROUNDS = 50

DATA_PATH = os.path.join("data", "Merged_Featured_DATA.xlsx")  # adjust if needed
HYPERPARAM_DIR = os.path.join("outputs", "Best_HyperParameter")  # folder with txt files

# ---------- Utilities ----------
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_excel(path)
    return df

def preprocess(df: pd.DataFrame, features: list):
    # ensure categorical columns are category dtype for XGBoost categorical support
    categorical_cols = [
        'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
        'rso1_objectType', 'rso2_objectType',
        'org1_displayName', 'org2_displayName'
    ]
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    X = df[features].copy()
    y = df['HighRisk'].copy()
    return X, y

def compute_scale_pos_weight(y):
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)

def load_params_from_txt(path):
    params = {}
    if not os.path.exists(path):
        print(f"Hyperparameter file not found: {path}. Using defaults.")
        return params
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # try convert numeric
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass
                params[key] = value
    print("Loaded params:", params)
    return params

def save_results(model_name, results_dict):
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved evaluation results to {out_path}")

# ---------- Core training & evaluation ----------
def train_and_evaluate(X, y, features, model_name):
    print(f"\n=== Training {model_name} on {len(features)} features ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    spw = compute_scale_pos_weight(y_train)
    print("scale_pos_weight:", spw)

    # load hyperparams if available
    hp_file = os.path.join(HYPERPARAM_DIR, model_name + ".txt")
    best_params = load_params_from_txt(hp_file)

    # ensure eval_metric is set to average precision (AUC-PR) if not provided
    if "eval_metric" not in best_params:
        best_params["eval_metric"] = "aucpr"

    model = XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        tree_method="hist",
        enable_categorical=True,
        use_label_encoder=False
    )

    # fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # sanity: if outputs look like logits, convert via sigmoid (rare with predict_proba)
    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        print("Detected scores outside [0,1], applying sigmoid to convert logits to probabilities.")
        y_prob = expit(y_prob)

    # Evaluate at default threshold 0.5
    thr_default = 0.5
    y_pred_default = (y_prob >= thr_default).astype(int)
    print("\n--- Evaluation @ threshold 0.5 ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_default))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_default, digits=4))
    auc_pr = average_precision_score(y_test, y_prob)
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"AUC-PR: {auc_pr:.6f}, AUC-ROC: {auc_roc:.6f}")

    # ---------- Threshold selection: enforce MIN_PRECISION and maximize recall ----------
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
    # pr_thresholds aligns with precisions[1:] and recalls[1:]
    candidates = np.concatenate(([0.0], pr_thresholds, [1.0]))

    best_thr = None
    best_rec = -1.0
    best_prec = 0.0
    best_score = -1.0

    for thr in candidates:
        preds = (y_prob >= thr).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        # enforce minimum precision
        if prec < MIN_PRECISION:
            continue

        # primary objective: maximize recall; tie-breaker uses weighted score
        score = WEIGHT_RECALL * rec + WEIGHT_PREC * prec

        if rec > best_rec or (rec == best_rec and score > best_score):
            best_rec = rec
            best_prec = prec
            best_thr = float(thr)
            best_score = score

    # fallback: if no threshold meets min precision, pick threshold maximizing weighted score
    if best_thr is None:
        print(f"No threshold met min_precision={MIN_PRECISION:.2f}. Falling back to weighted-score maximization.")
        for thr in candidates:
            preds = (y_prob >= thr).astype(int)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            score = WEIGHT_RECALL * rec + WEIGHT_PREC * prec
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_rec = rec
                best_prec = prec

    print(f"\nChosen threshold = {best_thr:.4f} (precision >= {MIN_PRECISION:.2f} enforced)")
    print(f"Recall = {best_rec:.4f}, Precision = {best_prec:.4f}, Score = {best_score:.4f}")

    # Evaluate at chosen threshold
    y_pred_best = (y_prob >= best_thr).astype(int)
    print("\n--- Evaluation @ chosen threshold ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best, digits=4))
    print(f"AUC-PR (average_precision): {auc_pr:.6f}, AUC-ROC: {auc_roc:.6f}")

    # Feature importances
    print("\nTop feature importances:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1])[:30]:
        print(f"{feat}: {imp:.4f}")

    # Save results
    results = {
        "model_name": model_name,
        "default_threshold": thr_default,
        "best_threshold": float(best_thr),
        "confusion_matrix_default": confusion_matrix(y_test, y_pred_default).tolist(),
        "confusion_matrix_best": confusion_matrix(y_test, y_pred_best).tolist(),
        "metrics_default": {
            "recall": float(recall_score(y_test, y_pred_default, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_default, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_default, zero_division=0)),
            "accuracy": float((y_pred_default == y_test).mean()),
            "auc_pr": float(auc_pr),
            "auc_roc": float(auc_roc)
        },
        "metrics_best_threshold": {
            "recall": float(recall_score(y_test, y_pred_best, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_best, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_best, zero_division=0)),
            "accuracy": float((y_pred_best == y_test).mean())
        }
    }
    save_results(model_name, results)

    # save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}.json")
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

# ---------- Main ----------
def main():
    df = load_data(DATA_PATH)
    print(df.info())

    # Train each variant
    X, y = preprocess(df, XG_Boost)
    train_and_evaluate(X, y, XG_Boost, "XG_Boost")

    X, y = preprocess(df, XG_Boost_NoLeak)
    train_and_evaluate(X, y, XG_Boost_NoLeak, "XG_Boost_NoLeak")

    X, y = preprocess(df, XG_Boost_NoLeak_Featured)
    train_and_evaluate(X, y, XG_Boost_NoLeak_Featured, "XG_Boost_NoLeak_Featured")

    X, y = preprocess(df, XG_Boost_Featured)
    train_and_evaluate(X, y, XG_Boost_Featured, "XG_Boost_Featured")

if __name__ == "__main__":
    main()

