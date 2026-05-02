"""Data loading utilities for performance and dropout models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

PERFORMANCE_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "dataset",
]

DROPOUT_FEATURES = [
    "total_clicks", "active_days", "relative_engagement",
    "avg_score", "avg_lateness", "num_of_prev_attempts", "studied_credits",
    "avg_clicks_per_day", "clicks_per_day",
    "imd_band_enc", "education_enc", "age_enc",
    "is_female", "has_disability",
    "low_engage_fail",
]


@st.cache_data
def load_performance_data() -> pd.DataFrame:
    """Load processed UCI dataset with all features and target column 'pass'."""
    path = BASE_DIR / "data" / "processed" / "performance" / "student_predictions.csv"
    if not path.exists():
        st.error(f"Performance data not found at {path}. Run `python src/predict.py --task performance`.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "actual_pass" in df.columns and "pass" not in df.columns:
        df = df.rename(columns={"actual_pass": "pass"})
    return df


@st.cache_data
def load_dropout_data() -> pd.DataFrame:
    """Load processed OULA dataset with features and target column 'target'."""
    path = BASE_DIR / "data" / "processed" / "dropout" / "Student_risk_report.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "Risk_Probability_Value" not in df.columns:
            if "Risk_Probability" in df.columns:
                df["Risk_Probability_Value"] = (
                    df["Risk_Probability"].astype(str).str.rstrip("%").astype(float) / 100.0
                )
            else:
                df["Risk_Probability_Value"] = 0.0
        if "Dropout_Risk_Level" not in df.columns:
            # Fallback only — Dropout_Risk_Level should now be in the CSV from predict.py
            _cfg_path = BASE_DIR / "config" / "model_config.json"
            _cfg = json.load(open(_cfg_path)) if _cfg_path.exists() else {}
            _levels = _cfg.get("dropout", {}).get("risk_levels", {"high": 0.51, "medium": 0.36})
            _high = _levels.get("high", 0.51)
            _medium = _levels.get("medium", 0.36)
            df["Dropout_Risk_Level"] = df["Risk_Probability_Value"].apply(
                lambda p: "High" if p >= _high else "Medium" if p >= _medium else "Low"
            )
        return df

    # Fallback: engineer from preprocessed features
    preprocessed = BASE_DIR / "data" / "processed" / "dropout" / "dropout_preprocessed.csv"
    if not preprocessed.exists():
        st.error("Dropout data not found. Please run the preprocessing notebook first.")
        return pd.DataFrame()
    return pd.read_csv(preprocessed)


def _engineer_dropout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features added by the retrained dropout model."""
    df = df.copy()
    imd_order = ["?", "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
                 "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    edu_order = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                 "HE Qualification", "Post Graduate Qualification"]
    age_order = ["0-35", "35-55", "55<="]
    if "imd_band_enc" not in df.columns and "imd_band" in df.columns:
        df["imd_band_enc"] = df["imd_band"].map({v: i for i, v in enumerate(imd_order)}).fillna(0)
    if "education_enc" not in df.columns and "highest_education" in df.columns:
        df["education_enc"] = df["highest_education"].map({v: i for i, v in enumerate(edu_order)}).fillna(0)
    if "age_enc" not in df.columns and "age_band" in df.columns:
        df["age_enc"] = df["age_band"].map({v: i for i, v in enumerate(age_order)}).fillna(0)
    if "is_female" not in df.columns and "gender" in df.columns:
        df["is_female"] = (df["gender"] == "F").astype(int)
    if "has_disability" not in df.columns and "disability" in df.columns:
        df["has_disability"] = (df["disability"] == "Y").astype(int)
    if "clicks_per_day" not in df.columns:
        df["clicks_per_day"] = df["total_clicks"] / (df.get("active_days", pd.Series(0, index=df.index)) + 1)
    if "low_engage_fail" not in df.columns:
        df["low_engage_fail"] = (
            (df.get("relative_engagement", pd.Series(0, index=df.index)) < 0) &
            (df.get("avg_score", pd.Series(100, index=df.index)) < 50)
        ).astype(int)
    return df


@st.cache_data
def load_dropout_preprocessed() -> pd.DataFrame:
    """Load the engineered feature set used to train the dropout model."""
    path = BASE_DIR / "data" / "processed" / "dropout" / "dropout_preprocessed.csv"
    if not path.exists():
        return pd.DataFrame()
    return _engineer_dropout_features(pd.read_csv(path))


@st.cache_data
def load_actionable_report() -> pd.DataFrame:
    """Load the actionable (high-risk-only) dropout report."""
    path = BASE_DIR / "data" / "processed" / "dropout" / "actionable_weekly_risk_report.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_feature_names(model_type: str) -> list[str]:
    """Return feature column names for 'performance' or 'dropout' model."""
    if model_type == "performance":
        return PERFORMANCE_FEATURES
    if model_type == "dropout":
        return DROPOUT_FEATURES
    raise ValueError(f"Unknown model_type: {model_type}")


@st.cache_data
def get_population_stats(model_type: str) -> dict:
    """Return {feature: {mean, median, std}} for each feature. Used for SHAP comparisons."""
    if model_type == "performance":
        df = load_performance_data()
        features = PERFORMANCE_FEATURES
    elif model_type == "dropout":
        df = load_dropout_preprocessed()
        features = DROPOUT_FEATURES
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    stats: dict = {}
    for feat in features:
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors="coerce").dropna()
            stats[feat] = {
                "mean": float(col.mean()) if len(col) else 0.0,
                "median": float(col.median()) if len(col) else 0.0,
                "std": float(col.std()) if len(col) else 1.0,
            }
    return stats


def load_config() -> dict:
    """Load model configuration from config/model_config.json."""
    config_path = BASE_DIR / "config" / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        "performance": {"threshold": 0.5, "risk_levels": {"high": 0.65, "medium": 0.5}},
        "dropout": {"threshold": 0.42, "risk_levels": {"high": 0.57, "medium": 0.42}},
    }


def deduplicate_dropout(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per student (highest risk), since OULA has multiple presentations."""
    if "Student_ID" not in df.columns:
        return df
    df = df.copy()
    df["Risk_Probability_Value"] = pd.to_numeric(df.get("Risk_Probability_Value", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["Student_ID", "Risk_Probability_Value"], ascending=[True, False], kind="stable")
    return df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
