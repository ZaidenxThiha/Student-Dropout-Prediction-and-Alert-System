"""Model loading and prediction utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    config_path = BASE_DIR / "config" / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        "performance": {"threshold": 0.5, "risk_levels": {"high": 0.65, "medium": 0.5}},
        "dropout": {"threshold": 0.42, "risk_levels": {"high": 0.57, "medium": 0.42}},
    }


@st.cache_resource
def load_model(model_type: str):
    """Load a trained model. 'performance' -> RF, 'dropout' -> XGBoost."""
    from joblib import load as jload

    config = _load_config()

    if model_type == "performance":
        path = BASE_DIR / config["performance"]["model_path"]
        if not path.exists():
            st.error(f"Performance model not found at {path}")
            return None
        return jload(path)

    if model_type == "dropout":
        # Try optimized first, fall back to original
        path = BASE_DIR / config["dropout"]["model_path"]
        if not path.exists():
            fallback = BASE_DIR / config["dropout"].get("model_path_fallback", "models/dropout/oula_ews_model.pkl")
            if not fallback.exists():
                st.error(f"Dropout model not found at {path} or {fallback}")
                return None
            return jload(fallback)
        return jload(path)

    raise ValueError(f"Unknown model_type: {model_type}")


@st.cache_resource
def load_dropout_features() -> list[str]:
    """Load the ordered dropout feature list from saved model_features.pkl."""
    from joblib import load as jload
    path = BASE_DIR / "models" / "dropout" / "model_features.pkl"
    if path.exists():
        return jload(path)
    return ["total_clicks", "active_days", "relative_engagement",
            "avg_score", "avg_lateness", "num_of_prev_attempts", "studied_credits"]


def get_risk_level(probability: float, model_type: str) -> str:
    """Map probability to High/Medium/Low using config thresholds."""
    config = _load_config()
    levels = config.get(model_type, {}).get("risk_levels", {"high": 0.65, "medium": 0.5})
    high = levels.get("high", 0.65)
    medium = levels.get("medium", 0.5)
    if probability >= high:
        return "High"
    if probability >= medium:
        return "Medium"
    return "Low"


def _engineer_dropout_row(row: pd.DataFrame) -> pd.DataFrame:
    """Compute derived dropout features for a single-row DataFrame."""
    row = row.copy()
    imd_order = ["?", "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
                 "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    edu_order = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                 "HE Qualification", "Post Graduate Qualification"]
    age_order = ["0-35", "35-55", "55<="]
    if "imd_band_enc" not in row.columns:
        row["imd_band_enc"] = row.get("imd_band", pd.Series(["?"])).map(
            {v: i for i, v in enumerate(imd_order)}
        ).fillna(0)
    if "education_enc" not in row.columns:
        row["education_enc"] = row.get("highest_education", pd.Series(["No Formal quals"])).map(
            {v: i for i, v in enumerate(edu_order)}
        ).fillna(0)
    if "age_enc" not in row.columns:
        row["age_enc"] = row.get("age_band", pd.Series(["0-35"])).map(
            {v: i for i, v in enumerate(age_order)}
        ).fillna(0)
    if "is_female" not in row.columns:
        row["is_female"] = (row.get("gender", pd.Series(["M"])) == "F").astype(int)
    if "has_disability" not in row.columns:
        row["has_disability"] = (row.get("disability", pd.Series(["N"])) == "Y").astype(int)
    if "clicks_per_day" not in row.columns:
        total = float(row["total_clicks"].iloc[0]) if "total_clicks" in row.columns else 0
        days = float(row["active_days"].iloc[0]) if "active_days" in row.columns else 0
        row["clicks_per_day"] = total / (days + 1)
    if "low_engage_fail" not in row.columns:
        rel_eng = float(row.get("relative_engagement", pd.Series([0])).iloc[0])
        score = float(row.get("avg_score", pd.Series([100])).iloc[0])
        row["low_engage_fail"] = int(rel_eng < 0 and score < 50)
    return row


def predict_single(student_data: dict, model_type: str) -> dict:
    """Predict for a single student. Returns prediction, probability, risk_level."""
    model = load_model(model_type)
    if model is None:
        return {"prediction": "Unknown", "probability": 0.0, "risk_level": "Unknown"}

    features = load_dropout_features() if model_type == "dropout" else None
    data = dict(student_data)
    # Performance model was trained with a 'dataset' column — inject default if missing
    if model_type == "performance" and "dataset" not in data:
        data["dataset"] = "math"
    row = pd.DataFrame([data])

    try:
        if model_type == "performance":
            proba = model.predict_proba(row)[0]
            prob_fail = 1.0 - proba[1]
            pred = "Pass" if proba[1] >= 0.5 else "Fail"
            return {
                "prediction": pred,
                "probability": float(prob_fail),
                "risk_level": get_risk_level(prob_fail, model_type),
            }
        else:
            row = _engineer_dropout_row(row)
            feat_row = row[features].fillna(0) if features else row
            prob_at_risk = float(model.predict_proba(feat_row)[0][1])
            config = _load_config()
            threshold = config["dropout"].get("threshold", 0.36)
            pred = "At-Risk" if prob_at_risk >= threshold else "On-Track"
            return {
                "prediction": pred,
                "probability": prob_at_risk,
                "risk_level": get_risk_level(prob_at_risk, model_type),
            }
    except Exception as e:
        return {"prediction": "Error", "probability": 0.0, "risk_level": "Unknown", "error": str(e)}


def predict_batch(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Add predicted_outcome, probability, risk_level columns to dataframe."""
    model = load_model(model_type)
    if model is None:
        df = df.copy()
        df["predicted_outcome"] = "Unknown"
        df["probability"] = 0.0
        df["risk_level"] = "Unknown"
        return df

    features = load_dropout_features() if model_type == "dropout" else None

    try:
        if model_type == "performance":
            proba = model.predict_proba(df)[:, 1]
            prob_fail = 1.0 - proba
            result = df.copy()
            result["probability"] = prob_fail
            result["predicted_outcome"] = ["Pass" if p >= 0.5 else "Fail" for p in proba]
            result["risk_level"] = [get_risk_level(p, model_type) for p in prob_fail]
        else:
            feat_df = df[features] if features else df
            proba = model.predict_proba(feat_df)[:, 1]
            config = _load_config()
            threshold = config["dropout"].get("threshold", 0.42)
            result = df.copy()
            result["probability"] = proba
            result["predicted_outcome"] = ["At-Risk" if p >= threshold else "On-Track" for p in proba]
            result["risk_level"] = [get_risk_level(p, model_type) for p in proba]
        return result
    except Exception as e:
        df = df.copy()
        df["predicted_outcome"] = "Error"
        df["probability"] = 0.0
        df["risk_level"] = "Unknown"
        return df
