# train_fttransformer_fixed.py
# Complete, self-contained training module (fixed and runnable)
# Place this file in your project and run from project root:
# python -m src.FT_Transformer.train_fttransformer_fixed
# (or run directly: python train_fttransformer_fixed.py if imports adjusted)

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, f1_score,
    roc_auc_score, classification_report, average_precision_score,
    precision_recall_curve, brier_score_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from datetime import datetime

# Adjust this import to match your package layout
from src.FT_Transformer.model import FTTransformer


# -----------------------------
# Dataset
# -----------------------------
class SdcDataset(Dataset):
    """Dataset that returns tensors for one row of the DataFrame."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        miss = torch.tensor([row["miss_distance"]], dtype=torch.float32)
        pc = torch.tensor([row["pc"]], dtype=torch.float32)
        hours_to_tca = torch.tensor([row["hours_to_tca"]], dtype=torch.float32)

        sat1 = torch.tensor(row["sat1_type"], dtype=torch.long)
        sat2 = torch.tensor(row["sat2_type"], dtype=torch.long)
        obj1 = torch.tensor(row["obj1_type"], dtype=torch.long)
        obj2 = torch.tensor(row["obj2_type"], dtype=torch.long)
        org1 = torch.tensor(row["org1"], dtype=torch.long)
        org2 = torch.tensor(row["org2"], dtype=torch.long)

        bools = torch.tensor([row[f"bool_{i}"] for i in range(10)], dtype=torch.float32)

        pc_label = torch.tensor([row["pc"]], dtype=torch.float32)
        class_label = torch.tensor([row["HighRisk"]], dtype=torch.float32)

        return (miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2,
                bools, pc_label, class_label)


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(df: pd.DataFrame):
    """Rename, create bools, encode categories. Returns processed df and bool_cols."""
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

    # Ensure numeric columns exist
    for col in ["miss_distance", "pc", "hours_to_tca"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["hours_to_tca"] = df["hours_to_tca"].astype(float)
    df["miss_distance"] = df["miss_distance"].astype(float)
    df["pc"] = df["pc"].astype(float)

    # Condition columns -> bool_0..bool_9
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

    # Categorical encoding
    cat_cols = ["sat1_type", "sat2_type", "obj1_type", "obj2_type", "org1", "org2"]
    for col in cat_cols:
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Ensure label exists
    if "HighRisk" not in df.columns:
        raise ValueError("Missing label column 'HighRisk' (0/1).")

    return df, bool_cols


# -----------------------------
# Dataloader helper
# -----------------------------
def make_dataloaders(df: pd.DataFrame, batch_size: int = 256, seed: int = 42):
    """Shuffle, split and return train/val DataLoaders."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df = df[:split]
    val_df = df[split:]

    train_dataset = SdcDataset(train_df)
    val_dataset = SdcDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, train_df, val_df


# -----------------------------
# Model builder
# -----------------------------
def build_model_from_df(df: pd.DataFrame, num_boolean_features: int = 10, device=None):
    """Build FTTransformer using cardinalities inferred from df (must match training ideally)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cardinalities = {
        "sat1": int(df["sat1_type"].nunique()),
        "sat2": int(df["sat2_type"].nunique()),
        "obj1": int(df["obj1_type"].nunique()),
        "obj2": int(df["obj2_type"].nunique()),
        "org1": int(df["org1"].nunique()),
        "org2": int(df["org2"].nunique())
    }
    with open("models/ft_transformer_cardinalities.json", "w") as f:
        json.dump(cardinalities, f)

    model = FTTransformer(
        num_categories_sat1=cardinalities["sat1"],
        num_categories_sat2=cardinalities["sat2"],
        num_categories_obj1=cardinalities["obj1"],
        num_categories_obj2=cardinalities["obj2"],
        num_categories_org1=cardinalities["org1"],
        num_categories_org2=cardinalities["org2"],
        num_boolean_features=num_boolean_features,
    ).to(device)

    return model, device, cardinalities


# -----------------------------
# Training loop (fixed)
# -----------------------------
def train_model(model, train_loader, val_loader, device, bool_cols, val_df=None, epochs=10, lr=1e-4, save_dir="models"):
    loss_pc_fn = nn.MSELoss()
    loss_class_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            (miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2,
             bools, pc_label, class_label) = batch

            miss, pc, hours_to_tca = miss.to(device), pc.to(device), hours_to_tca.to(device)
            sat1, sat2, obj1, obj2, org1, org2 = (
                sat1.to(device), sat2.to(device), obj1.to(device),
                obj2.to(device), org1.to(device), org2.to(device)
            )
            bools, pc_label, class_label = bools.to(device), pc_label.to(device), class_label.to(device)

            pc_pred, class_pred = model(miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2, bools)

            loss_pc = loss_pc_fn(pc_pred, pc_label)
            loss_class = loss_class_fn(class_pred, class_label)
            loss = loss_pc + loss_class

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / max(batch_count, 1)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss Avg: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_losses = 0.0
        all_class_labels = []
        all_class_preds = []
        all_pc_labels = []
        all_pc_preds = []

        with torch.no_grad():
            for batch in val_loader:
                (miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2,
                 bools, pc_label, class_label) = batch

                miss, pc, hours_to_tca = miss.to(device), pc.to(device), hours_to_tca.to(device)
                sat1, sat2, obj1, obj2, org1, org2 = (
                    sat1.to(device), sat2.to(device), obj1.to(device),
                    obj2.to(device), org1.to(device), org2.to(device)
                )
                bools, pc_label, class_label = bools.to(device), pc_label.to(device), class_label.to(device)

                pc_pred, class_pred = model(miss, pc, hours_to_tca, sat1, sat2, obj1, obj2, org1, org2, bools)

                loss_pc = loss_pc_fn(pc_pred, pc_label)
                loss_class = loss_class_fn(class_pred, class_label)
                val_losses += (loss_pc + loss_class).item()

                all_class_labels.append(class_label.cpu())
                all_class_preds.append(class_pred.cpu())
                all_pc_labels.append(pc_label.cpu())
                all_pc_preds.append(pc_pred.cpu())

        # Concatenate and convert to numpy arrays
        all_class_labels = torch.cat(all_class_labels).numpy().ravel()
        all_class_preds = torch.cat(all_class_preds).numpy().ravel()
        all_pc_labels = torch.cat(all_pc_labels).numpy().ravel()
        all_pc_preds = torch.cat(all_pc_preds).numpy().ravel()

        val_loss_avg = val_losses / max(len(val_loader), 1)
        class_preds_binary = (all_class_preds >= 0.5).astype(int)

        auc_pr = average_precision_score(all_class_labels, all_class_preds)
        try:
            auc_roc = roc_auc_score(all_class_labels, all_class_preds)
        except ValueError:
            auc_roc = float("nan")
        brier = brier_score_loss(all_class_labels, all_class_preds)

        # Threshold sweep
        precisions, recalls, thrs = precision_recall_curve(all_class_labels, all_class_preds)
        candidates = np.append(thrs, 1.0)

        best_f1 = 0.0
        best_f1_thr = 0.5
        best_recall = 0.0
        best_recall_thr = 0.5
        min_precision_for_recall = 0.50

        for thr in np.append(0.0, candidates):
            preds_thr = (all_class_preds >= thr).astype(int)
            prec = precision_score(all_class_labels, preds_thr, zero_division=0)
            rec = recall_score(all_class_labels, preds_thr, zero_division=0)
            f1 = f1_score(all_class_labels, preds_thr, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thr = thr
            if (prec >= min_precision_for_recall) and (rec > best_recall):
                best_recall = rec
                best_recall_thr = thr

        preds_best_f1 = (all_class_preds >= best_f1_thr).astype(int)
        preds_best_recall = (all_class_preds >= best_recall_thr).astype(int)

        # Confusion matrices and reports
        cm_default = confusion_matrix(all_class_labels, class_preds_binary)
        cm_best_f1 = confusion_matrix(all_class_labels, preds_best_f1)
        cm_best_recall = confusion_matrix(all_class_labels, preds_best_recall)

        report_default = classification_report(all_class_labels, class_preds_binary, digits=4)
        report_best_f1 = classification_report(all_class_labels, preds_best_f1, digits=4)
        report_best_recall = classification_report(all_class_labels, preds_best_recall, digits=4)

        # Brier and regression metrics
        brier = brier_score_loss(all_class_labels, all_class_preds)
        try:
            tn, fp, fn, tp = cm_best_recall.ravel()
        except ValueError:
            tn = fp = fn = tp = 0

        mse = mean_squared_error(all_pc_labels, all_pc_preds)
        mae = mean_absolute_error(all_pc_labels, all_pc_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_pc_labels, all_pc_preds)

        # Print summary
        print("\n===== FT-Transformer Validation Summary =====")
        print(f"Val Loss Avg: {val_loss_avg:.6f}")
        print(f"AUC-PR: {auc_pr:.6f}")
        print(f"AUC-ROC: {auc_roc:.6f}")
        print(f"Brier score: {brier:.6f}")
        print(f"Best F1: {best_f1:.4f} at thr {best_f1_thr:.3f}")
        print(f"Best recall (prec >= {min_precision_for_recall}): {best_recall:.4f} at thr {best_recall_thr:.3f}")
        print("\n--- Default classification report ---")
        print(report_default)
        print("\n--- Best-recall classification report ---")
        print(report_best_recall)
        print("\n--- Best-F1 classification report ---")
        print(report_best_f1)
        print("Confusion matrices (default, best_recall, best_f1):")
        print(cm_default)
        print(cm_best_recall)
        print(cm_best_f1)
        print(f"\nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"\nRegression (Pc) metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}")

        # Save JSON summary for this epoch
        results_summary = {
            "model": "FT-Transformer",
            "epoch": int(epoch + 1),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "validation": {
                "val_loss_avg": float(val_loss_avg),
                "auc_pr": float(auc_pr),
                "auc_roc": None if np.isnan(auc_roc) else float(auc_roc),
                "brier": float(brier)
            },
            "thresholds": {
                "default_threshold": 0.5,
                "best_f1_threshold": float(best_f1_thr),
                "best_f1": float(best_f1),
                "best_recall_threshold": float(best_recall_thr),
                "best_recall": float(best_recall),
                "min_precision_for_recall": float(min_precision_for_recall)
            },
            "classification_reports": {
                "default_report": report_default,
                "best_recall_report": report_best_recall,
                "best_f1_report": report_best_f1
            },
            "confusion_matrices": {
                "default": cm_default.tolist(),
                "best_recall": cm_best_recall.tolist(),
                "best_f1": cm_best_f1.tolist()
            },
            "counts_best_recall": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "metrics_best_f1": {
                "precision": float(precision_score(all_class_labels, preds_best_f1, zero_division=0)),
                "recall": float(recall_score(all_class_labels, preds_best_f1)),
                "f1": float(f1_score(all_class_labels, preds_best_f1))
            },
            "regression_pc_metrics": {"mse": float(mse), "mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
        }

        out_file = os.path.join("results", f"ft_transformer_validation_epoch_{epoch+1}.json")
        os.makedirs("results", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results_summary, f, indent=4)
        print(f"Validation summary saved to: {out_file}")

        # Optionally show failure examples if val_df provided
        if val_df is not None:
            try:
                val_df_reset = val_df.reset_index(drop=True)
                inspect_df = val_df_reset.copy()
                inspect_df["ft_prob"] = all_class_preds
                inspect_df["label"] = all_class_labels
                fp_examples = inspect_df[(inspect_df["label"] == 0)].sort_values("ft_prob", ascending=False).head(5)
                fn_examples = inspect_df[(inspect_df["label"] == 1)].sort_values("ft_prob", ascending=True).head(5)
                print("\nTop 5 False Positives (high prob, label=0):")
                print(fp_examples[["ft_prob", "label"] + bool_cols + ["miss_distance", "pc", "hours_to_tca"]].to_string(index=False))
                print("\nTop 5 False Negatives (low prob, label=1):")
                print(fn_examples[["ft_prob", "label"] + bool_cols + ["miss_distance", "pc", "hours_to_tca"]].to_string(index=False))
            except Exception as e:
                print("Could not show failure examples (need val_df in scope). Error:", e)

        # Save model checkpoint each epoch
        model_path = os.path.join(save_dir, f"ft_transformer_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model checkpoint to: {model_path}\n")

    # final save
    final_path = os.path.join(save_dir, "ft_transformer_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/sample_Featured_Data.csv")

    # Preprocess (overwrites df variable with processed df)
    df, bool_cols = preprocess(df)

    # Create dataloaders
    train_loader, val_loader, train_df, val_df = make_dataloaders(df, batch_size=256)

    # Build model (cardinalities inferred from df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device, cardinalities = build_model_from_df(df, num_boolean_features=len(bool_cols), device=device)
    print("Model built with cardinalities:", cardinalities)

    # Train
    train_model(model, train_loader, val_loader, device, bool_cols=bool_cols, val_df=val_df, epochs=10, lr=1e-4, save_dir="models")