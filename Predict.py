import pandas as pd
from src.Ensemble_All import load_xgb_probs
from src.Ensemble_All import load_lgb_probs

from src.LightGBM.Light_GBM import (
    LGB_Boost, LGB_Boost_NoLeak,
    LGB_Boost_Featured, LGB_Boost_NoLeak_Featured,
    CATEGORICAL_COLS
)

from src.XGBoost.Combined_XG_Boost import (
    XG_Boost, XG_Boost_NoLeak,
    XG_Boost_Featured, XG_Boost_NoLeak_Featured
)

MODEL_LIST = [
    ("XG_Boost", XG_Boost, "xgb"),
    ("XG_Boost_NoLeak", XG_Boost_NoLeak, "xgb"),
    ("XG_Boost_Featured", XG_Boost_Featured, "xgb"),
    ("XG_Boost_NoLeak_Featured", XG_Boost_NoLeak_Featured, "xgb"),

    ("LGB_Boost", LGB_Boost, "lgb"),
    ("LGB_Boost_NoLeak", LGB_Boost_NoLeak, "lgb"),
    ("LGB_Boost_Featured", LGB_Boost_Featured, "lgb"),
    ("LGB_Boost_NoLeak_Featured", LGB_Boost_NoLeak_Featured, "lgb"),
]
best_w = [0.10271423501250435,
        0.17716131369096302,
        0.1458275541414128,
        0.0983209236751832,
        0.20656566757547107,
        0.047094069010400016,
        0.09863000546995927,
        0.12368623142410624]

best_thr= 0.315

test_probs = {}

def Predict_Ensemble(test_data):
    for name, feats, mtype in MODEL_LIST:
        
        test_data[CATEGORICAL_COLS] = test_data[CATEGORICAL_COLS].astype("category")
        if mtype == "xgb":
            test_probs[name] = load_xgb_probs(name, feats, test_data)
        else:
            test_probs[name] = load_lgb_probs(name, feats, test_data)
    
    test_probs_df = pd.DataFrame(test_probs)
    test_avg = test_probs_df.values.dot(best_w)
    test_pred = (test_avg >= best_thr).astype(int)
    
    return test_pred



###########--------------------     FTT Transformer Prediction ---------------------#################



from src.predict_FTTransformer import preprocess_data, load_ft_model, predict_ft
import pandas as pd

def Predict_FTT(sample_test):
    """
    Run FT-Transformer prediction on `sample_test`.

    Args:
      sample_test (pd.DataFrame or str): DataFrame to predict on or path to CSV/Excel file.

    Returns:
      pd.DataFrame: predictions with columns ['pc_pred','highrisk_prob','risk_label'].
    """
    # Accept a file path or DataFrame
    if isinstance(sample_test, str):
        path = sample_test
        if path.lower().endswith((".csv", ".txt")):
            sample_test = pd.read_csv(path)
        elif path.lower().endswith((".xls", ".xlsx")):
            sample_test = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Provide a pandas DataFrame or a CSV/XLSX path.")

    if not isinstance(sample_test, pd.DataFrame):
        raise ValueError("sample_test must be a pandas DataFrame or a path to a CSV/XLSX file.")

    # Preprocess (returns processed df and bool column names)
    sample_test, bool_cols = preprocess_data(sample_test)

    # Hardcoded training cardinalities (must match the checkpoint used for load)
    cardinalities = {
            "sat1": int(2),
            "sat2": int(1),
            "obj1": int(4),
            "obj2": int(2),
            "org1": int(15),
            "org2": int(5)
        }

    # Load model (will raise if model file or shapes mismatch)
    model, device = load_ft_model(cardinalities, bool_cols)

    # Run predictions and return results DataFrame
    results_df = predict_ft(model, sample_test, bool_cols, device)

    # Optionally attach original index to results for easy merging
    results_df.index = sample_test.index
    return results_df
