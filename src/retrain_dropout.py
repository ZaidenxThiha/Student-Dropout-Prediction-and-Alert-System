"""Retrain the dropout XGBoost model with expanded features and tuned hyperparameters."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE_DIR / "data/processed/dropout/dropout_preprocessed.csv")
print(f"Dataset: {df.shape[0]} rows")

# ── Feature engineering ────────────────────────────────────────────────────────
# Ordinal encode imd_band (deprivation index — clear ordinal meaning)
imd_order = ["?", "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
             "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
df["imd_band_enc"] = df["imd_band"].map(
    {v: i for i, v in enumerate(imd_order)}
).fillna(0)

# Ordinal encode education level
edu_order = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
             "HE Qualification", "Post Graduate Qualification"]
df["education_enc"] = df["highest_education"].map(
    {v: i for i, v in enumerate(edu_order)}
).fillna(0)

# Ordinal encode age band
age_order = ["0-35", "35-55", "55<="]
df["age_enc"] = df["age_band"].map({v: i for i, v in enumerate(age_order)}).fillna(0)

# Binary flags
df["is_female"] = (df["gender"] == "F").astype(int)
df["has_disability"] = (df["disability"] == "Y").astype(int)

# Engagement ratio (guard against zero active_days)
df["clicks_per_day"] = df["total_clicks"] / (df["active_days"] + 1)

# Interaction: low engagement + poor score is a strong signal
df["low_engage_fail"] = (
    (df["relative_engagement"] < 0) & (df["avg_score"] < 50)
).astype(int)

FEATURES = [
    # Original
    "total_clicks", "active_days", "relative_engagement",
    "avg_score", "avg_lateness", "num_of_prev_attempts", "studied_credits",
    # Added
    "avg_clicks_per_day", "clicks_per_day",
    "imd_band_enc", "education_enc", "age_enc",
    "is_female", "has_disability",
    "low_engage_fail",
]

X = df[FEATURES].fillna(0)
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(y_train)}  Test: {len(y_test)}")

# ── Train optimized XGBoost ────────────────────────────────────────────────────
# Classes are nearly balanced (17K vs 15K), so scale_pos_weight≈1
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {pos_weight:.3f}")

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=30,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
THRESHOLD = 0.36
proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= THRESHOLD).astype(int)

acc  = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec  = recall_score(y_test, preds, zero_division=0)
f1   = f1_score(y_test, preds, zero_division=0)
auc  = roc_auc_score(y_test, proba)

print("\n=== Results at threshold 0.36 ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
print()
print(classification_report(y_test, preds, target_names=["On-Track", "At-Risk"]))

# ── Save artifacts ─────────────────────────────────────────────────────────────
model_dir = BASE_DIR / "models/dropout"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "dropout_xgb_optimized.joblib")
joblib.dump(FEATURES, model_dir / "model_features.pkl")
print("Saved model and features.")

# ── Update config metrics ──────────────────────────────────────────────────────
config_path = BASE_DIR / "config/model_config.json"
with open(config_path) as f:
    config = json.load(f)

config["dropout"]["metrics"] = {
    "precision": round(prec, 4),
    "recall":    round(rec, 4),
    "f1":        round(f1, 4),
    "auc":       round(auc, 4),
    "accuracy":  round(acc, 4),
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print("Updated config/model_config.json with new metrics.")
