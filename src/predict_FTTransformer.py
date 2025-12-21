# src/predict_FTTransformer.py
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    f1_score, roc_auc_score, classification_report
)
import numpy as np
import json
import os

# Import the class (not the module)
from src.FT_Transformer.model import FTTransformer


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_data(df: pd.DataFrame):
    """Rename columns, convert conditions, encode categories, return processed df + bool cols."""
    df = df.rename(columns={
        "cdmMissDistance": "miss_distance",
        "cdmPc": "pc",
        "SAT1_CDM_TYPE": "sat1_type",
        "SAT2_CDM_TYPE": "sat2_type",
        "rso1_objectType": "obj1_type",
        "rso2_objectType": "obj2_type",
        "org1_displayName": "org1",
        "org2_displayName": "org2"
    })

    condition_cols = [
        "condition_cdmType=EPHEM:HAC",
        "condition_24H_tca_72H",
        "condition_Pc>1e-6",
        "condition_missDistance<2000m",
        "condition_Radial_100m",
        "condition_Radial<50m",
        "condition_InTrack_500m",
        "condition_CrossTrack_500m",
        "condition_sat2posUnc_1km",
        "condition_sat2Obs_25"
    ]
    for col in condition_cols:
        if col not in df.columns:
            raise ValueError(f"Missing condition column: {col}")
        df[col] = df[col].astype(int)
    for i, col in enumerate(condition_cols):
        df[f"bool_{i}"] = df[col]
    bool_cols = [f"bool_{i}" for i in range(10)]
    df[bool_cols] = df[bool_cols].astype("int64")

    # numeric
    for col in ["miss_distance", "pc", "hours_to_tca"]:
        if col not in df.columns:
            raise ValueError(f"Missing numeric column: {col}")
        df[col] = df[col].astype(float)

    # categorical (Note: for production you should reuse training LabelEncoders)
    cat_cols = ["sat1_type", "sat2_type", "obj1_type", "obj2_type", "org1", "org2"]
    for col in cat_cols:
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df, bool_cols


# -----------------------------
# Model Loader
# -----------------------------
def load_ft_model(cardinalities, bool_cols, model_path="models/ft_transformer.pth"):
    """Load FTTransformer with correct cardinalities."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cpu")
    model = FTTransformer(
        num_categories_sat1=cardinalities["sat1"],
        num_categories_sat2=cardinalities["sat2"],
        num_categories_obj1=cardinalities["obj1"],
        num_categories_obj2=cardinalities["obj2"],
        num_categories_org1=cardinalities["org1"],
        num_categories_org2=cardinalities["org2"],
        num_boolean_features=len(bool_cols)
    ).to(device)

    # load state dict (no weights_only argument)
    state_dict = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# -----------------------------
# Prediction
# -----------------------------
def predict_ft(model, df: pd.DataFrame, bool_cols, device="cpu"):
    """Run predictions row-by-row and return DataFrame of results."""
    results = []
    for _, row in df.iterrows():
        miss = torch.tensor([[row["miss_distance"]]], dtype=torch.float32)
        pc = torch.tensor([[row["pc"]]], dtype=torch.float32)
        hours_to_tca = torch.tensor([[row["hours_to_tca"]]], dtype=torch.float32)

        sat1 = torch.tensor([row["sat1_type"]], dtype=torch.long)
        sat2 = torch.tensor([row["sat2_type"]], dtype=torch.long)
        obj1 = torch.tensor([row["obj1_type"]], dtype=torch.long)
        obj2 = torch.tensor([row["obj2_type"]], dtype=torch.long)
        org1 = torch.tensor([row["org1"]], dtype=torch.long)
        org2 = torch.tensor([row["org2"]], dtype=torch.long)

        bools = torch.tensor(row[bool_cols].values.astype("float32")).unsqueeze(0)

        with torch.no_grad():
            pc_pred, class_pred = model(
                miss, pc, hours_to_tca,
                sat1, sat2, obj1, obj2, org1, org2,
                bools
            )

        highrisk_probability = float(class_pred.item())
        risk_label = "HighRisk" if highrisk_probability >= 0.5 else "LowRisk"

        results.append({
            "pc_pred": float(pc_pred.item()),
            "highrisk_prob": highrisk_probability,
            "risk_label": risk_label
        })

    return pd.DataFrame(results)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_predictions(results_df: pd.DataFrame, df: pd.DataFrame, save_dir="results"):
    """Evaluate predictions and save metrics + outputs."""
    if "HighRisk" not in df.columns:
        raise ValueError("Ground-truth column 'HighRisk' not found in df for evaluation.")

    results_df["true_label"] = df["HighRisk"].values
    y_true = results_df["true_label"].values
    y_prob = results_df["highrisk_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float((y_pred == y_true).mean()),
        "auc_pr": float(roc_auc_score(y_true, y_prob)),
        "auc_roc": float(roc_auc_score(y_true, y_prob))
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "FT_Transformer_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    results_df.to_excel(os.path.join(save_dir, "predictions.xlsx"), index=False)
    return metrics


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    # load data
    df = pd.read_csv("data/sample_Featured_Data.csv")

    # preprocess (overwrites df)
    df, bool_cols = preprocess_data(df)

    # Prefer loading saved training cardinalities if available
    card_path = "models/ft_transformer_cardinalities.json"
    
    cardinalities = {
            "sat1": int(2),
            "sat2": int(1),
            "obj1": int(4),
            "obj2": int(2),
            "org1": int(15),
            "org2": int(5)
        }
    # load model
    model, device = load_ft_model(cardinalities, bool_cols, model_path="models/ft_transformer.pth")

    # predict
    results_df = predict_ft(model, df, bool_cols, device)
    print(results_df)
    # evaluate (requires 'HighRisk' column in df)
    # try:
    #     metrics = evaluate_predictions(results_df, df)
    #     print("Evaluation metrics:", metrics)
    # except Exception as e:
    #     print("Evaluation skipped or failed:", e)
    #     # still save predictions
    #     os.makedirs("results", exist_ok=True)
    #     results_df.to_excel(os.path.join("results", "predictions.xlsx"), index=False)
    #     print("Predictions saved to results/predictions.xlsx")