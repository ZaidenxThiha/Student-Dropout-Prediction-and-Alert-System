"""Prediction utilities for performance and dropout models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import load


def load_model(model_path: Optional[Path] = None):
    """Load the trained performance pipeline."""
    if model_path is None:
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "performance"
            / "pass_classifier_rf.joblib"
        )
    return load(model_path)


def load_dropout_model(model_path: Optional[Path] = None):
    """Load the trained dropout model."""
    if model_path is None:
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "dropout"
            / "rf_dropout_model.joblib"
        )
    return load(model_path)


def risk_level(prob_fail: float) -> str:
    """Map performance fail probability to risk bucket."""
    if prob_fail < 0.3:
        return "Low"
    if prob_fail < 0.6:
        return "Medium"
    return "High"


def derive_performance_risk_factors(row: pd.Series) -> str:
    """Create a compact explanation of major performance risk signals."""
    factors = []

    if row.get("G2", 20) <= 9:
        factors.append("Low G2 Score")
    if row.get("G1", 20) <= 9:
        factors.append("Low G1 Score")
    if row.get("failures", 0) >= 1:
        factors.append("Prior Class Failures")
    if row.get("studytime", 4) <= 1:
        factors.append("Low Study Time")
    if row.get("absences", 0) >= 10:
        factors.append("High Absences")
    if row.get("goout", 1) >= 4:
        factors.append("High Social Time")
    if row.get("Walc", 1) >= 4 or row.get("Dalc", 1) >= 3:
        factors.append("High Alcohol Use")
    if str(row.get("higher", "yes")).lower() == "no":
        factors.append("Low Higher-Education Intent")

    if not factors:
        return "General Academic Risk Profile"
    return " | ".join(factors[:3])


def derive_dropout_risk_factors(row: pd.Series) -> str:
    """Create a compact explanation of dropout risk signals."""
    factors = []

    if row.get("sum_click", 0.0) < -0.2:
        factors.append("Low Online Engagement")
    if row.get("date_registration", 0.0) > 0.1:
        factors.append("Late Registration")
    if row.get("studied_credits", 0.0) > 0.5:
        factors.append("High Course Load")
    if row.get("num_of_prev_attempts", 0.0) > 0:
        factors.append("Previous Failures")

    if not factors:
        return "General Risk Profile"
    return " | ".join(factors[:3])


def predict_students(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Predict pass/fail, fail-risk score, and risk level for given students."""
    if model is None:
        model = load_model()

    preds = model.predict(df)  # 1 = pass, 0 = fail
    prob_pass = model.predict_proba(df)[:, 1]
    prob_fail = 1.0 - prob_pass

    result = df.copy()
    result["prediction"] = preds
    result["risk_score"] = prob_fail
    result["risk_level"] = result["risk_score"].apply(risk_level)
    result["Primary_Risk_Factors"] = result.apply(derive_performance_risk_factors, axis=1)
    return result


def predict_dropout_students(
    df: pd.DataFrame,
    student_ids: Optional[pd.Series] = None,
    model=None,
    threshold: float = 0.4,
) -> pd.DataFrame:
    """Predict dropout risk and create a full dropout risk report."""
    if model is None:
        model = load_dropout_model()

    risk_probabilities = model.predict_proba(df)[:, 1]
    predictions = (risk_probabilities >= threshold).astype(int)

    result = df.copy().reset_index(drop=True)
    if student_ids is None:
        result.insert(0, "Student_ID", range(1, len(result) + 1))
    else:
        result["Student_ID"] = pd.Series(student_ids).reset_index(drop=True)

    result["Risk_Probability_Value"] = risk_probabilities
    result["Risk_Probability"] = (result["Risk_Probability_Value"] * 100).round(1).astype(str) + "%"
    result["ML_Prediction"] = predictions
    result["Prediction_Band"] = pd.Series(predictions).map({1: "band_1", 0: "band_2"})
    result["Primary_Risk_Factors"] = result.apply(derive_dropout_risk_factors, axis=1)
    result["Intervention_Status"] = pd.Series(predictions).map(
        {1: "ACTION REQUIRED: High Risk", 0: "Safe: On Track"}
    )
    result.loc[result["Intervention_Status"] == "Safe: On Track", "Primary_Risk_Factors"] = "N/A"
    result = result.sort_values(by="Risk_Probability_Value", ascending=False).reset_index(drop=True)
    return result


def build_actionable_dropout_report(report_df: pd.DataFrame) -> pd.DataFrame:
    """Create the compact actionable dropout intervention report."""
    actionable = report_df[report_df["ML_Prediction"] == 1].copy()
    columns = [
        "Student_ID",
        "Prediction_Band",
        "Primary_Risk_Factors",
        "Intervention_Status",
        "Risk_Probability",
    ]
    return actionable[columns].reset_index(drop=True)


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    """Save predictions dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_combined_performance_data(base_dir: Path) -> pd.DataFrame:
    """Load the combined cleaned UCI dataset, rebuilding it if missing."""
    processed_dir = base_dir / "data" / "processed" / "performance"
    combined_path = processed_dir / "student_all_cleaned.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    mat_path = processed_dir / "student_mat_cleaned.csv"
    por_path = processed_dir / "student_por_cleaned.csv"
    mat_df = pd.read_csv(mat_path).assign(dataset="math")
    por_df = pd.read_csv(por_path).assign(dataset="portuguese")
    combined_df = pd.concat([mat_df, por_df], ignore_index=True)
    combined_df.to_csv(combined_path, index=False)
    return combined_df


def load_dropout_inputs(base_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load dropout feature data and student ids."""
    processed_dir = base_dir / "data" / "processed" / "dropout"
    feature_df = pd.read_csv(processed_dir / "X_test.csv")
    student_ids = pd.read_csv(processed_dir / "student_ids.csv").squeeze()
    return feature_df, student_ids


def export_performance_predictions(base_dir: Path) -> Path:
    """Export the full-cohort performance predictions used by Streamlit."""
    out_path = base_dir / "data" / "processed" / "performance" / "student_predictions.csv"
    data = load_combined_performance_data(base_dir)
    feature_df = data.drop(columns=["G3", "pass"])
    preds_df = predict_students(feature_df)
    preds_df.insert(0, "student_id", range(1, len(preds_df) + 1))
    preds_df["actual_pass"] = data["pass"].reset_index(drop=True)
    preds_df["G3"] = data["G3"].reset_index(drop=True)
    preds_df["predicted_outcome"] = preds_df["prediction"].map({0: "Fail", 1: "Pass"})
    save_predictions(preds_df, out_path)
    return out_path


def export_dropout_predictions(base_dir: Path) -> tuple[Path, Path]:
    """Export the full and actionable dropout reports used by Streamlit."""
    processed_dir = base_dir / "data" / "processed" / "dropout"
    report_path = processed_dir / "Student_risk_report.csv"
    actionable_path = processed_dir / "actionable_weekly_risk_report.csv"

    feature_df, student_ids = load_dropout_inputs(base_dir)
    report_df = predict_dropout_students(feature_df, student_ids=student_ids)
    actionable_df = build_actionable_dropout_report(report_df)

    save_predictions(report_df, report_path)
    save_predictions(actionable_df, actionable_path)
    return report_path, actionable_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export performance and dropout predictions.")
    parser.add_argument(
        "--task",
        choices=["performance", "dropout", "all"],
        default="all",
        help="Which prediction export to run.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent

    if args.task in {"performance", "all"}:
        out_path = export_performance_predictions(base_dir)
        print(f"Saved performance predictions to {out_path}")

    if args.task in {"dropout", "all"}:
        report_path, actionable_path = export_dropout_predictions(base_dir)
        print(f"Saved dropout report to {report_path}")
        print(f"Saved actionable dropout report to {actionable_path}")


if __name__ == "__main__":
    main()
