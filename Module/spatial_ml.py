"""Spatial Machine Learning helper module for EO Change Detection.

Provides reusable utilities for spatial block generation, sensitivity analysis,
GroupShuffleSplit partitioning, spatial leakage verification, metric evaluation,
multi-sensor ablation benchmarking, and GIS handoff validation.
"""

import json, joblib, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    roc_auc_score
)

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

def load_ml_config(json_path=None):
    """Load machine learning and project configuration from root config.json."""
    if json_path is None:
        json_path = CONFIG_PATH
    json_path = Path(json_path)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "RANDOM_STATE": 42,
        "DEFAULT_BLOCK_SIZE": 0.02,
        "CANDIDATE_BLOCK_SIZES": [0.01, 0.02, 0.03],
        "SPLIT_RATIOS": {"DEV_SIZE": 0.8, "TEST_SIZE": 0.2, "TRAIN_SIZE": 0.75, "VAL_SIZE": 0.25},
        "FEATURE_GROUPS": {
            "S1_FEATURES": ["VV_pre", "VH_pre", "VV_post", "VH_post", "dVV", "dVH"],
            "S2_FEATURES": ["NDVI_pre", "NDWI_pre", "NDBI_pre", "NDVI_post", "NDWI_post", "NDBI_post", "dNDVI", "dNDWI", "dNDBI"],
            "TERRAIN_FEATURES": ["DEM", "slope"]
        },
        "LOGISTIC_REGRESSION": {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"},
        "RANDOM_FOREST": {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        "PATHS": {
            "INPUT_FILE": "Data/tables/EO_feature_table_labelled.csv",
            "SPATIAL_SPLIT_FILE": "Data/tables/EO_feature_table_spatial_split.csv",
            "PREDICTION_FILE": "Data/tables/RF_spatial_test_predictions.csv",
            "MODEL_FILE": "Data/models/random_forest_final.joblib",
            "ABLATION_FILE": "Data/tables/ablation_results.csv"
        }
    }

load_config = load_ml_config

def construct_spatial_blocks(df, block_size=0.02, lon_col="longitude", lat_col="latitude"):
    """
    Construct discrete regular spatial blocks based on longitude and latitude.
    
    Args:
        df (pd.DataFrame): Input dataframe containing coordinates.
        block_size (float): Dimension of the spatial grid cell in degrees.
        lon_col (str): Column name for longitude.
        lat_col (str): Column name for latitude.
        
    Returns:
        pd.DataFrame: Dataframe with added 'lon_block', 'lat_block', and 'spatial_block'.
    """
    df_out = df.copy()
    df_out["lon_block"] = np.floor(df_out[lon_col] / block_size).astype(int)
    df_out["lat_block"] = np.floor(df_out[lat_col] / block_size).astype(int)
    df_out["spatial_block"] = (
        df_out["lon_block"].astype(str) + "_" + df_out["lat_block"].astype(str)
    )
    return df_out

def evaluate_block_size_sensitivity(df, candidate_sizes=[0.01, 0.02, 0.03],
                                   lon_col="longitude", lat_col="latitude", target_col="Y"):
    """
    Evaluate candidate spatial block sizes for number of blocks and pixel density distribution.
    
    Args:
        df (pd.DataFrame): Dataframe with coordinate and target columns.
        candidate_sizes (list): Sequence of grid sizes in degrees to evaluate.
        lon_col (str): Longitude column.
        lat_col (str): Latitude column.
        target_col (str): Target label column.
        
    Returns:
        pd.DataFrame: Summary table comparing block sizes.
    """
    records = []
    for size in candidate_sizes:
        temp_df = construct_spatial_blocks(df, block_size=size, lon_col=lon_col, lat_col=lat_col)
        block_counts = temp_df["spatial_block"].value_counts()
        
        # Calculate class 1 proportion per block (for blocks with >= 50 pixels)
        pos_props = temp_df.groupby("spatial_block")[target_col].mean()
        records.append({
            "block_size_deg": size,
            "approx_km": round(size * 111.0, 2),
            "num_blocks": temp_df["spatial_block"].nunique(),
            "min_pixels_per_block": int(block_counts.min()),
            "median_pixels_per_block": float(block_counts.median()),
            "max_pixels_per_block": int(block_counts.max()),
            "mean_class1_ratio": float(pos_props.mean()),
            "std_class1_ratio": float(pos_props.std())
        })
    return pd.DataFrame(records)

def spatial_train_val_test_split(df, group_col="spatial_block",
                                 dev_size=0.80, test_size=0.20,
                                 train_size=0.75, val_size=0.25,
                                 random_state=42):
    """
    Perform a two-stage spatial block partition (Train ~60%, Val ~20%, Test ~20%).
    
    Args:
        df (pd.DataFrame): Dataframe containing spatial group identifiers.
        group_col (str): Column name containing block identifiers.
        dev_size (float): Proportion of data allocated to development (Train + Val).
        test_size (float): Proportion allocated to held-out test.
        train_size (float): Proportion of dev data allocated to train.
        val_size (float): Proportion of dev data allocated to validation.
        random_state (int): Random seed for reproducible partitioning.
        
    Returns:
        tuple: (train_df, val_df, test_df, full_df_with_split)
    """
    df_out = df.copy()
    groups = df_out[group_col]
    
    # Split into Dev (Train + Val) and Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(gss_test.split(df_out, groups=groups))
    
    dev_df = df_out.iloc[dev_idx].copy()
    test_df = df_out.iloc[test_idx].copy()
    
    # Split Dev into Train and Validation
    dev_groups = dev_df[group_col]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_sub_idx, val_sub_idx = next(gss_val.split(dev_df, groups=dev_groups))
    
    train_df = dev_df.iloc[train_sub_idx].copy()
    val_df = dev_df.iloc[val_sub_idx].copy()
    
    # Tag split column in full dataframe
    df_out["split"] = ""
    df_out.loc[train_df.index, "split"] = "train"
    df_out.loc[val_df.index, "split"] = "validation"
    df_out.loc[test_df.index, "split"] = "test"
    
    # Add split tag in subsets
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    
    return train_df, val_df, test_df, df_out

def verify_split_leakage(train_df, val_df, test_df, group_col="spatial_block"):
    """
    Assert zero overlap between spatial blocks across all split combinations.
    
    Returns:
        dict: Diagnostics showing unique block counts and overlap status.
    """
    train_blocks = set(train_df[group_col].unique())
    val_blocks = set(val_df[group_col].unique())
    test_blocks = set(test_df[group_col].unique())
    
    train_val_overlap = train_blocks & val_blocks
    train_test_overlap = train_blocks & test_blocks
    val_test_overlap = val_blocks & test_blocks
    
    if len(train_val_overlap) > 0:
        raise AssertionError(f"Spatial leakage detected: Train and Val share {len(train_val_overlap)} blocks: {train_val_overlap}")
    if len(train_test_overlap) > 0:
        raise AssertionError(f"Spatial leakage detected: Train and Test share {len(train_test_overlap)} blocks: {train_test_overlap}")
    if len(val_test_overlap) > 0:
        raise AssertionError(f"Spatial leakage detected: Val and Test share {len(val_test_overlap)} blocks: {val_test_overlap}")
        
    return {
        "train_blocks": len(train_blocks),
        "val_blocks": len(val_blocks),
        "test_blocks": len(test_blocks),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "leakage_status": "PASSED (Zero spatial block leakage)"
    }

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1, IoU (Jaccard), and ROC-AUC.
    
    Args:
        y_true (array-like): Ground truth binary targets.
        y_pred (array-like): Predicted binary classes.
        y_prob (array-like, optional): Predicted probabilities for positive class (Y=1).
        
    Returns:
        dict: Computed metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    
    roc_auc = np.nan
    if y_prob is not None:
        # ROC-AUC requires both classes present in y_true
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            
    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "IoU": float(iou),
        "ROC-AUC": float(roc_auc) if not np.isnan(roc_auc) else None
    }

def spatial_group_kfold_cv(dev_df, model_or_pipeline, feature_cols, target_col="Y",
                           group_col="spatial_block", n_splits=4):
    """
    Perform Spatial GroupKFold Cross-Validation on development data.
    
    Guarantees entire spatial blocks are held out in each fold.
    Returns:
        fold_results (pd.DataFrame): Metrics for each individual fold.
        summary (dict): Mean and std for each metric, plus formatted string.
    """
    from sklearn.base import clone
    gkf = GroupKFold(n_splits=n_splits)
    groups = dev_df[group_col].values
    X = dev_df[feature_cols].values
    y = dev_df[target_col].values
    
    records = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        # Assert strict spatial block separation between fold train and fold val
        train_blocks = set(dev_df.iloc[train_idx][group_col])
        val_blocks = set(dev_df.iloc[val_idx][group_col])
        assert len(train_blocks & val_blocks) == 0, f"Spatial leakage detected in fold {fold_idx}"
        
        clf = clone(model_or_pipeline)
        clf.fit(X[train_idx], y[train_idx])
        y_val_pred = clf.predict(X[val_idx])
        y_val_prob = clf.predict_proba(X[val_idx])[:, 1] if hasattr(clf, "predict_proba") else None
        
        metrics = calculate_classification_metrics(y[val_idx], y_val_pred, y_val_prob)
        metrics["Fold"] = f"Fold {fold_idx}"
        metrics["Train_Blocks"] = len(train_blocks)
        metrics["Val_Blocks"] = len(val_blocks)
        metrics["Val_Rows"] = len(val_idx)
        records.append(metrics)
        
    df_folds = pd.DataFrame(records)
    
    summary = {}
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "IoU", "ROC-AUC"]
    for m in metric_cols:
        if m in df_folds.columns and df_folds[m].notna().any():
            mean_v = df_folds[m].mean()
            std_v = df_folds[m].std()
            summary[f"{m}_mean"] = float(mean_v)
            summary[f"{m}_std"] = float(std_v)
            summary[m] = f"{mean_v:.4f} ± {std_v:.4f}"
            
    return df_folds, summary

def run_spatial_cv_ablation(dev_df, feature_groups_dict, target_col="Y",
                            group_col="spatial_block", rf_params=None, n_splits=4):
    """
    Execute Random Forest ablation study using Spatial GroupKFold Cross-Validation on development data.
    
    Args:
        dev_df (pd.DataFrame): Development subset containing spatial groups.
        feature_groups_dict (dict): Mapping of experiment names to feature column lists.
        target_col (str): Target column name.
        group_col (str): Column name with spatial blocks.
        rf_params (dict): Random Forest hyperparameter dictionary.
        n_splits (int): Number of spatial folds.
        
    Returns:
        tuple: (summary_df, fold_details_dict)
    """
    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        
    cv_summary_records = []
    fold_details = {}
    
    for exp_name, feat_cols in feature_groups_dict.items():
        print(f"Running Spatial CV Ablation: {exp_name} ({len(feat_cols)} features)...")
        clf = RandomForestClassifier(**rf_params)
        df_folds, summary = spatial_group_kfold_cv(
            dev_df=dev_df,
            model_or_pipeline=clf,
            feature_cols=feat_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits
        )
        fold_details[exp_name] = df_folds
        
        record = {
            "Experiment": exp_name,
            "Num_Features": len(feat_cols),
            "Accuracy": summary.get("Accuracy"),
            "Precision": summary.get("Precision"),
            "Recall": summary.get("Recall"),
            "F1": summary.get("F1"),
            "IoU": summary.get("IoU"),
            "ROC-AUC": summary.get("ROC-AUC")
        }
        cv_summary_records.append(record)
        
    cols_order = ["Experiment", "Num_Features", "Accuracy", "Precision", "Recall", "F1", "IoU", "ROC-AUC"]
    summary_df = pd.DataFrame(cv_summary_records)[cols_order]
    return summary_df, fold_details

def run_ablation_experiments(train_df, test_df, feature_groups_dict,
                             target_col="Y", rf_params=None):
    """
    Execute Random Forest ablation study on the exact same spatial test partition.
    
    Args:
        train_df (pd.DataFrame): Training subset.
        test_df (pd.DataFrame): Unseen spatial test subset.
        feature_groups_dict (dict): Mapping of experiment names to feature column lists.
        target_col (str): Target column name.
        rf_params (dict): Random Forest hyperparameter dictionary.
        
    Returns:
        pd.DataFrame: Ablation metrics comparison table.
    """
    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    results = []
    for exp_name, feat_cols in feature_groups_dict.items():
        print(f"Running Ablation Experiment: {exp_name} ({len(feat_cols)} features)...")
        X_train = train_df[feat_cols].values
        X_test = test_df[feat_cols].values
        
        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        metrics["Experiment"] = exp_name
        metrics["Num_Features"] = len(feat_cols)
        results.append(metrics)
        
    cols_order = ["Experiment", "Num_Features", "Accuracy", "Precision", "Recall", "F1", "IoU", "ROC-AUC"]
    return pd.DataFrame(results)[cols_order]

def validate_and_export_predictions(test_df, y_pred, y_prob, output_path,
                                    target_col="Y", lon_col="longitude", lat_col="latitude"):
    """
    Verify test predictions and export to CSV formatted for downstream GIS mapping.
    
    Required output format: [longitude, latitude, Y, Y_pred, P_change]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(y_pred) != len(test_df):
        raise ValueError(f"Length mismatch: {len(y_pred)} predictions vs {len(test_df)} test rows")
        
    preds_df = pd.DataFrame({
        "longitude": test_df[lon_col].values,
        "latitude": test_df[lat_col].values,
        "Y": test_df[target_col].values.astype(int),
        "Y_pred": np.asarray(y_pred).astype(int),
        "P_change": np.asarray(y_prob).astype(float)
    })
    
    # Rigorous sanity assertions
    assert not preds_df["longitude"].isna().any(), "longitude contains null values"
    assert not preds_df["latitude"].isna().any(), "latitude contains null values"
    assert not preds_df["Y"].isna().any(), "Y target contains null values"
    assert not preds_df["Y_pred"].isna().any(), "Y_pred contains null values"
    assert not preds_df["P_change"].isna().any(), "P_change contains null values"
    assert set(preds_df["Y_pred"].unique()).issubset({0, 1}), f"Invalid Y_pred values: {preds_df['Y_pred'].unique()}"
    assert (preds_df["P_change"] >= 0.0).all() and (preds_df["P_change"] <= 1.0).all(), "P_change outside [0, 1] range"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(output_path, index=False)
    print(f"Predictions successfully validated and exported to: {output_path}")
    print(f"Exported row count: {len(preds_df):,}")
    return preds_df

def export_model_with_metadata(model, model_path, features_list, rf_params, block_size, random_state):
    """
    Serialize trained model with an accompanying JSON metadata file.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path, compress=3)
    print(f"Model saved to: {model_path} (compressed)")
    
    metadata = {
        "model_type": type(model).__name__,
        "export_timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "block_size": block_size,
        "features": list(features_list),
        "hyperparameters": rf_params
    }
    
    meta_path = model_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {meta_path}")
