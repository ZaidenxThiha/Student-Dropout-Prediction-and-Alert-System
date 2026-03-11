"""Prediction utilities for the student pass/fail classifier.

Loads the trained pipeline, runs predictions, attaches fail-risk probabilities
and risk levels, and optionally saves results. Assumes inputs use the same
schema as the combined training data (see data/processed/student_all_cleaned.csv).
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import load


def load_model(model_path: Optional[Path] = None):
    """Load the trained pipeline."""
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "pass_classifier_rf.joblib"
    return load(model_path)


def risk_level(prob_fail: float) -> str:
    """Map fail probability to risk bucket."""
    if prob_fail < 0.3:
        return "Low"
    if prob_fail < 0.6:
        return "Medium"
    return "High"


def predict_students(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Predict pass/fail, fail-risk score, and risk level for given students."""
    if model is None:
        model = load_model()

    preds = model.predict(df)  # 1 = pass, 0 = fail
    prob_pass = model.predict_proba(df)[:, 1]
    prob_fail = 1.0 - prob_pass

    result = df.copy()
    result["prediction"] = preds
    result["risk_score"] = prob_fail  # higher = more likely to fail
    result["risk_level"] = result["risk_score"].apply(risk_level)
    return result


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    """Save predictions dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Example: load combined cleaned data, run predictions, save to processed.
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "student_all_cleaned.csv"
    out_path = base_dir / "data" / "processed" / "student_predictions.csv"

    data = pd.read_csv(data_path)
    feature_df = data.drop(columns=["G3", "pass"])
    preds_df = predict_students(feature_df)

    # Attach a lightweight identifier
    preds_df.insert(0, "student_id", range(1, len(preds_df) + 1))
    save_predictions(preds_df, out_path)
    print(f"Saved predictions to {out_path}")
