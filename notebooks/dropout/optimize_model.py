"""
Dropout Model Optimization
==========================
Runs RandomizedSearchCV on XGBClassifier, sweeps thresholds,
and saves the best model to models/dropout/dropout_xgb_optimized.joblib.
Updates config/model_config.json with the best threshold.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES = ["total_clicks", "active_days", "relative_engagement",
            "avg_score", "avg_lateness", "num_of_prev_attempts", "studied_credits"]


def load_data():
    path = BASE_DIR / "data" / "processed" / "dropout" / "dropout_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {path}")
    df = pd.read_csv(path)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")
    X = df[FEATURES].copy()
    y = df["target"].copy()
    return X, y


def sweep_threshold(model, X_val, y_val):
    """Find threshold where precision >= 0.70 AND recall >= 0.75."""
    proba = model.predict_proba(X_val)[:, 1]
    best = {"threshold": 0.42, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    for t in np.arange(0.25, 0.76, 0.01):
        preds = (proba >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f = f1_score(y_val, preds, zero_division=0)
        if p >= 0.70 and r >= 0.75 and f > best["f1"]:
            best = {"threshold": round(t, 2), "f1": f, "precision": p, "recall": r}

    if best["f1"] == 0.0:
        # Fallback: best f1 regardless of constraints
        for t in np.arange(0.25, 0.76, 0.01):
            preds = (proba >= t).astype(int)
            f = f1_score(y_val, preds, zero_division=0)
            if f > best["f1"]:
                p = precision_score(y_val, preds, zero_division=0)
                r = recall_score(y_val, preds, zero_division=0)
                best = {"threshold": round(t, 2), "f1": f, "precision": p, "recall": r}

    return best


def main():
    print("=" * 60)
    print("Dropout Model Optimization")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    X, y = load_data()
    print(f"    Dataset shape: {X.shape}, Positive rate: {y.mean():.2%}")

    # 2. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. SMOTE on training only
    print("\n[2] Applying SMOTE on training set...")
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    After SMOTE: {X_res.shape}, Positive rate: {y_res.mean():.2%}")

    # 4. RandomizedSearchCV
    print("\n[3] Running RandomizedSearchCV (this may take a few minutes)...")
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.5],
        "scale_pos_weight": [max(ratio, 1.0)],
    }

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=50,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_res, y_res)
    best_model = search.best_estimator_
    print(f"    Best params: {search.best_params_}")

    # 5. Threshold sweep
    print("\n[4] Sweeping thresholds...")
    best_thresh_info = sweep_threshold(best_model, X_val, y_val)
    threshold = best_thresh_info["threshold"]
    print(f"    Best threshold: {threshold}")
    print(f"    Precision: {best_thresh_info['precision']:.3f}, Recall: {best_thresh_info['recall']:.3f}, F1: {best_thresh_info['f1']:.3f}")

    # 6. Evaluate old vs new
    print("\n[5] Comparing old vs new model...")
    old_model_path = BASE_DIR / "models" / "dropout" / "oula_ews_model.pkl"
    old_proba = load(old_model_path).predict_proba(X_val)[:, 1]
    old_preds = (old_proba >= 0.5).astype(int)
    new_proba = best_model.predict_proba(X_val)[:, 1]
    new_preds = (new_proba >= threshold).astype(int)

    print("\n    OLD MODEL (threshold=0.5):")
    print(f"      Precision: {precision_score(y_val, old_preds, zero_division=0):.3f}")
    print(f"      Recall:    {recall_score(y_val, old_preds, zero_division=0):.3f}")
    print(f"      F1:        {f1_score(y_val, old_preds, zero_division=0):.3f}")
    print(f"      AUC:       {roc_auc_score(y_val, old_proba):.3f}")

    new_precision = precision_score(y_val, new_preds, zero_division=0)
    new_recall = recall_score(y_val, new_preds, zero_division=0)
    new_f1 = f1_score(y_val, new_preds, zero_division=0)
    new_auc = roc_auc_score(y_val, new_proba)

    print(f"\n    NEW MODEL (threshold={threshold}):")
    print(f"      Precision: {new_precision:.3f}")
    print(f"      Recall:    {new_recall:.3f}")
    print(f"      F1:        {new_f1:.3f}")
    print(f"      AUC:       {new_auc:.3f}")

    # 7. Save optimized model
    out_path = BASE_DIR / "models" / "dropout" / "dropout_xgb_optimized.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out_path)
    print(f"\n[6] Saved optimized model to {out_path}")

    # 8. Update config
    config_path = BASE_DIR / "config" / "model_config.json"
    with open(config_path) as f:
        config = json.load(f)

    high_thresh = round(threshold + 0.15, 2)
    config["dropout"]["threshold"] = threshold
    config["dropout"]["risk_levels"] = {"high": high_thresh, "medium": threshold}
    config["dropout"]["metrics"] = {
        "precision": round(new_precision, 3),
        "recall": round(new_recall, 3),
        "f1": round(new_f1, 3),
        "auc": round(new_auc, 3),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[7] Updated config at {config_path}")
    print("\nOptimization complete.")


if __name__ == "__main__":
    main()
