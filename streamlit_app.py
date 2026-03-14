from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import derive_performance_risk_factors


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_pass_fail_predictions(base_dir: Path) -> pd.DataFrame:
    df = load_csv(base_dir / "data" / "processed" / "performance" / "student_predictions.csv").copy()
    if "Primary_Risk_Factors" not in df.columns:
        df["Primary_Risk_Factors"] = df.apply(derive_performance_risk_factors, axis=1)
    return df


def prepare_pass_fail_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df["predicted_outcome"] = display_df["prediction"].map({0: "Fail", 1: "Pass"})
    display_df = display_df.rename(columns={"risk_score": "fail_risk_score"})
    return display_df


def load_dropout_predictions(base_dir: Path) -> pd.DataFrame:
    df = load_csv(base_dir / "data" / "processed" / "dropout" / "Student_risk_report.csv").copy()
    df["Risk_Probability_Value"] = (
        df["Risk_Probability"].astype(str).str.rstrip("%").astype(float) / 100.0
    )
    df["Dropout_Risk_Level"] = pd.cut(
        df["Risk_Probability_Value"],
        bins=[-0.01, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return df


def load_actionable_dropout_report(base_dir: Path) -> pd.DataFrame:
    return load_csv(base_dir / "data" / "processed" / "dropout" / "actionable_weekly_risk_report.csv")


def render_bar_chart(series: pd.Series, title: str, colors: list[str]) -> None:
    fig, ax = plt.subplots()
    ax.bar(series.index.astype(str), series.values, color=colors[: len(series)])
    ax.set_xlabel("Category")
    ax.set_ylabel("Students")
    ax.set_title(title)
    plt.xticks(rotation=15)
    st.pyplot(fig, clear_figure=True)


def render_pass_fail_tab(df: pd.DataFrame) -> None:
    st.subheader("Pass/Fail Risk")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    high_risk = df[df["risk_level"] == "High"]
    metric_col1.metric("Total Students", len(df))
    metric_col2.metric("High Risk Students", len(high_risk))
    metric_col3.metric("Average Fail Risk", f"{df['risk_score'].mean():.1%}")

    st.markdown("### Risk Level Distribution")
    counts = (
        df["risk_level"]
        .astype(str)
        .value_counts()
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
        .astype(int)
    )
    render_bar_chart(counts, "Students by Pass/Fail Risk Level", ["#d62728", "#ffbf00", "#1f77b4"])

    st.markdown("### Search by Student ID")
    student_id = st.text_input("Enter Student ID", "", key="pass_fail_student_id")
    if student_id.strip():
        try:
            sid = int(student_id)
            student = df[df["student_id"] == sid]
            if student.empty:
                st.warning("No student found with that ID.")
            else:
                st.dataframe(prepare_pass_fail_display(student))
        except ValueError:
            st.warning("Please enter a numeric Student ID.")

    st.markdown("### Students at Risk")
    selected_level = st.selectbox("Choose risk level", ["High", "Medium", "Low"], key="pass_fail_level")
    filtered = df[df["risk_level"] == selected_level].copy()
    if filtered.empty:
        st.info(f"No {selected_level} risk students found.")
    else:
        display_columns = [
            "student_id",
            "predicted_outcome",
            "fail_risk_score",
            "Primary_Risk_Factors",
            "risk_level",
        ]
        filtered_display = prepare_pass_fail_display(filtered)
        available_columns = [column for column in display_columns if column in filtered_display.columns]
        st.dataframe(filtered_display[available_columns])

    st.markdown("### All Pass/Fail Predictions")
    st.dataframe(prepare_pass_fail_display(df))


def render_dropout_tab(report_df: pd.DataFrame, actionable_df: pd.DataFrame) -> None:
    st.subheader("Dropout Prevention")

    high_risk_mask = report_df["Intervention_Status"].astype(str).str.contains("High Risk", na=False)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total Students", len(report_df))
    metric_col2.metric("High Dropout Risk", int(high_risk_mask.sum()))
    metric_col3.metric("Average Dropout Risk", f"{report_df['Risk_Probability_Value'].mean():.1%}")
    metric_col4.metric("Actionable Cases", len(actionable_df))

    st.markdown("### Intervention Status Distribution")
    status_counts = report_df["Intervention_Status"].astype(str).value_counts()
    render_bar_chart(status_counts, "Students by Intervention Status", ["#d62728", "#2ca02c", "#1f77b4"])

    st.markdown("### Dropout Risk Level Distribution")
    risk_counts = (
        report_df["Dropout_Risk_Level"]
        .astype(str)
        .value_counts()
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
        .astype(int)
    )
    render_bar_chart(risk_counts, "Students by Dropout Risk Level", ["#d62728", "#ffbf00", "#1f77b4"])

    st.markdown("### Search by Student ID")
    student_id = st.text_input("Enter Dropout Student ID", "", key="dropout_student_id")
    if student_id.strip():
        try:
            sid = int(student_id)
            student = report_df[report_df["Student_ID"] == sid]
            if student.empty:
                st.warning("No student found with that ID.")
            else:
                st.dataframe(student)
        except ValueError:
            st.warning("Please enter a numeric Student ID.")

    st.markdown("### Filter Dropout Cases")
    selected_status = st.selectbox(
        "Choose intervention status",
        sorted(report_df["Intervention_Status"].astype(str).unique()),
        key="dropout_status",
    )
    filtered = report_df[report_df["Intervention_Status"] == selected_status].copy()
    if filtered.empty:
        st.info(f"No students found for status: {selected_status}")
    else:
        display_columns = [
            "Student_ID",
            "Prediction_Band",
            "Risk_Probability_Value",
            "Primary_Risk_Factors",
            "Intervention_Status",
        ]
        available_columns = [column for column in display_columns if column in filtered.columns]
        st.dataframe(filtered[available_columns])

    st.markdown("### Weekly Actionable Interventions")
    if actionable_df.empty:
        st.info("No actionable weekly intervention report found.")
    else:
        st.dataframe(actionable_df)

    st.markdown("### Full Dropout Risk Report")
    display_df = report_df.rename(
        columns={
            "Risk_Probability": "dropout_risk_probability",
            "ML_Prediction": "dropout_prediction",
        }
    )
    st.dataframe(display_df)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pass_fail_path = base_dir / "data" / "processed" / "performance" / "student_predictions.csv"
    dropout_path = base_dir / "data" / "processed" / "dropout" / "Student_risk_report.csv"
    actionable_path = base_dir / "data" / "processed" / "dropout" / "actionable_weekly_risk_report.csv"

    st.title("Student Success Dashboard")
    st.caption("Combined academic performance and dropout prevention dashboard.")

    missing_paths = [
        str(path)
        for path in [pass_fail_path, dropout_path, actionable_path]
        if not path.exists()
    ]
    if missing_paths:
        st.error(
            "Required processed files are missing:\n- " + "\n- ".join(missing_paths)
        )
        return

    pass_fail_df = load_pass_fail_predictions(base_dir)
    dropout_df = load_dropout_predictions(base_dir)
    actionable_df = load_actionable_dropout_report(base_dir)

    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Pass/Fail Records", len(pass_fail_df))
    overview_col2.metric("Dropout Records", len(dropout_df))
    overview_col3.metric(
        "Combined High Risk Cases",
        int((pass_fail_df["risk_level"] == "High").sum())
        + int(dropout_df["Intervention_Status"].astype(str).str.contains("High Risk", na=False).sum()),
    )

    pass_fail_tab, dropout_tab = st.tabs(["Pass/Fail Model", "Dropout Prevention Model"])

    with pass_fail_tab:
        render_pass_fail_tab(pass_fail_df)

    with dropout_tab:
        render_dropout_tab(dropout_df, actionable_df)


if __name__ == "__main__":
    main()
