from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import derive_performance_risk_factors


st.set_page_config(
    page_title="Student Risk Monitoring System",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_pass_fail_predictions(base_dir: Path) -> pd.DataFrame:
    df = load_csv(base_dir / "data" / "processed" / "performance" / "student_predictions.csv").copy()
    if "Primary_Risk_Factors" not in df.columns:
        df["Primary_Risk_Factors"] = df.apply(derive_performance_risk_factors, axis=1)
    if "risk_score" not in df.columns:
        df["risk_score"] = 0.0
    if "risk_level" not in df.columns:
        df["risk_level"] = "Unknown"
    return df


def deduplicate_dropout_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if "Student_ID" not in df.columns:
        return df

    deduped = df.copy()
    deduped["Risk_Probability_Value"] = pd.to_numeric(
        deduped.get("Risk_Probability_Value", 0.0),
        errors="coerce",
    ).fillna(0.0)

    # OULA can contain multiple records per learner across module presentations.
    # Keep one dashboard row per student, preserving the highest-risk record.
    deduped = deduped.sort_values(
        by=["Student_ID", "Risk_Probability_Value"],
        ascending=[True, False],
        kind="stable",
    )
    deduped = deduped.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
    return deduped


def load_dropout_predictions(base_dir: Path) -> pd.DataFrame:
    df = load_csv(base_dir / "data" / "processed" / "dropout" / "Student_risk_report.csv").copy()

    if "Risk_Probability_Value" not in df.columns:
        if "Risk_Probability" in df.columns:
            df["Risk_Probability_Value"] = (
                df["Risk_Probability"].astype(str).str.rstrip("%").astype(float) / 100.0
            )
        else:
            df["Risk_Probability_Value"] = 0.0

    if "Dropout_Risk_Level" not in df.columns:
        df["Dropout_Risk_Level"] = pd.cut(
            df["Risk_Probability_Value"],
            bins=[-0.01, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
        )
    return deduplicate_dropout_predictions(df)


def load_actionable_dropout_report(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "data" / "processed" / "dropout" / "actionable_weekly_risk_report.csv"
    if not path.exists():
        return pd.DataFrame()
    return load_csv(path)


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif !important;
        }

        .block-container {
            padding-top: 0.45rem;
            padding-bottom: 0.6rem;
            max-width: 1400px;
        }

        h1, h2, h3, h4 {
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"] {
            background: var(--secondary-background-color) !important;
            border: 1px solid #d7e6fb !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 0.7rem;
            border-radius: 12px;
            box-shadow: none;
            transition: none;
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 500;
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: var(--primary-color) !important;
        }

        /* Tabs Styling */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            border: none !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
            padding-bottom: 0.5rem !important;
            transition: color 0.3s ease !important;
        }

        div[data-baseweb="tab-highlight"] {
            background-color: var(--primary-color) !important;
            height: 3px !important;
            border-radius: 3px 3px 0 0 !important;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            background: var(--secondary-background-color) !important;
            border: 1px solid #d7e6fb !important;
            border-radius: 12px !important;
            margin-bottom: 0.45rem !important;
        }

        /* Buttons */
        .stDownloadButton > button {
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        .stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #d7e6fb !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }

        h3 {
            margin-top: 0.1rem !important;
            margin-bottom: 0.55rem !important;
            font-size: 1.05rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric_value:.0%}"


def normalize_risk_label(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    if "high" in text.lower():
        return "High"
    if "medium" in text.lower():
        return "Medium"
    if "low" in text.lower() or "safe" in text.lower():
        return "Low"
    return text


def simplify_risk_factors(raw_factors: object) -> str:
    if pd.isna(raw_factors):
        return "Attendance, performance, or engagement concerns need review."

    factors = []
    replacements = {
        "Low G2 Score": "recent classroom performance is lower than expected",
        "Low G1 Score": "earlier term performance needs support",
        "Prior Class Failures": "previous course difficulties may still be affecting progress",
        "Low Early Assessment Score": "early assessment results suggest the student may need extra help",
        "Low Total Clicks": "online learning activity has been limited",
        "Low Active Days": "regular participation has been inconsistent",
        "High Absences": "attendance patterns may be affecting progress",
    }

    for factor in str(raw_factors).split("|"):
        cleaned = factor.strip()
        if not cleaned or cleaned.upper() == "N/A":
            continue
        factors.append(replacements.get(cleaned, cleaned.replace("_", " ").lower()))

    if not factors:
        return "Attendance, performance, or engagement concerns need review."
    return "; ".join(factors)


def academic_parent_message(row: pd.Series) -> str:
    risk_label = normalize_risk_label(row.get("risk_level", "Unknown")).lower()
    factors = simplify_risk_factors(row.get("Primary_Risk_Factors"))
    return (
        f"Student {row.get('student_id', 'Unknown')} is currently at risk academically and may need support. "
        f"Current concern level: {risk_label}. Key concerns include {factors}. "
        "Early intervention recommended through teacher check-ins, home encouragement, and review of current study habits."
    )


def dropout_parent_message(row: pd.Series) -> str:
    risk_label = normalize_risk_label(row.get("Intervention_Status", row.get("Dropout_Risk_Level", "Unknown"))).lower()
    factors = simplify_risk_factors(row.get("Primary_Risk_Factors"))
    return (
        f"Student {row.get('Student_ID', 'Unknown')} is showing signs of dropout risk and needs support. "
        f"Current concern level: {risk_label}. Key concerns include {factors}. "
        "Early intervention recommended through outreach, attendance follow-up, and regular progress check-ins."
    )


def build_academic_alerts(df: pd.DataFrame) -> pd.DataFrame:
    risk_labels = df.get("risk_level", pd.Series("Unknown", index=df.index, dtype="object")).astype(str)
    alerts = pd.DataFrame(
        {
            "Student_ID": df.get("student_id", pd.Series(index=df.index, dtype="object")),
            "Alert_Type": "Academic Risk",
            "Risk_Value": pd.to_numeric(df.get("risk_score", 0.0), errors="coerce").fillna(0.0),
            "Risk_Label": risk_labels,
            "Risk_Factors": df.get(
                "Primary_Risk_Factors",
                pd.Series("Not available", index=df.index, dtype="object"),
            ).astype(str),
        }
    )
    alerts["Parent_Message"] = df.apply(academic_parent_message, axis=1)
    alerts["Recommended_Action"] = (
        "Review recent grades, coordinate with teachers, and reinforce a regular study routine."
    )
    return alerts


def build_dropout_alerts(df: pd.DataFrame) -> pd.DataFrame:
    risk_label = df.get("Intervention_Status", pd.Series(index=df.index, dtype="object")).apply(
        normalize_risk_label
    )
    alerts = pd.DataFrame(
        {
            "Student_ID": df.get("Student_ID", pd.Series(index=df.index, dtype="object")),
            "Alert_Type": "Dropout Risk",
            "Risk_Value": pd.to_numeric(df.get("Risk_Probability_Value", 0.0), errors="coerce").fillna(0.0),
            "Risk_Label": risk_label,
            "Risk_Factors": df.get(
                "Primary_Risk_Factors",
                pd.Series("Not available", index=df.index, dtype="object"),
            ).astype(str),
        }
    )
    alerts["Parent_Message"] = df.apply(dropout_parent_message, axis=1)
    alerts["Recommended_Action"] = (
        "Follow up on attendance and participation, connect with support staff, and maintain regular family-school communication."
    )
    return alerts


def build_parent_alerts(pass_fail_df: pd.DataFrame, dropout_df: pd.DataFrame) -> pd.DataFrame:
    academic_alerts = build_academic_alerts(pass_fail_df)
    dropout_alerts = build_dropout_alerts(dropout_df)
    combined = pd.concat([academic_alerts, dropout_alerts], ignore_index=True)
    combined["Risk_Label"] = combined["Risk_Label"].apply(normalize_risk_label)
    combined["Student_ID"] = combined["Student_ID"].astype(str)
    combined = combined.sort_values(by="Risk_Value", ascending=False).reset_index(drop=True)
    return combined


def render_bar_chart(series: pd.Series, title: str, color: str) -> None:
    if series.empty:
        st.info("No data available")
        return

    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    ax.bar(series.index.astype(str), series.values, color=color, alpha=0.9, edgecolor='none', width=0.6)
    ax.set_title(title, pad=15, fontsize=12, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=15)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def extract_top_risk_factors(series: pd.Series, limit: int = 6) -> pd.Series:
    factors: list[str] = []
    for value in series.dropna().astype(str):
        for factor in value.split("|"):
            cleaned = factor.strip()
            if cleaned and cleaned.upper() != "N/A":
                factors.append(cleaned)

    if not factors:
        return pd.Series(dtype="int64")

    return pd.Series(factors).value_counts().head(limit)


def render_section_header(title: str, description: str) -> None:
    st.subheader(title)
    st.caption(description)


def render_academic_tab(df: pd.DataFrame) -> None:
    render_section_header(
        "Academic Risk Analysis",
        "Review pass/fail model outputs, identify learners who need support, and focus on the highest-risk students first.",
    )

    with st.container(border=True):
        st.markdown("### 📊 Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Students", f"{len(df):,}")
        metric_col2.metric("High Risk Count", int((df["risk_level"] == "High").sum()))
        metric_col3.metric("Average Risk Score", format_percent(df["risk_score"].mean()))

    with st.container():
        left_col, right_col = st.columns(2, gap="medium")
        with left_col:
            with st.container(border=True):
                st.markdown("### 📋 Data")
                filter_col1, filter_col2 = st.columns([1, 1])
                risk_levels = ["All", "High", "Medium", "Low"]
                selected_level = filter_col1.selectbox("Risk Level", risk_levels, key="academic_risk_level")
                student_query = filter_col2.text_input("Search Student_ID", "", key="academic_student_search")

                filtered = df.copy()
                if selected_level != "All":
                    filtered = filtered[filtered["risk_level"].astype(str) == selected_level]
                if student_query.strip():
                    filtered = filtered[
                        filtered["student_id"].astype(str).str.contains(student_query.strip(), case=False, na=False)
                    ]

                filtered = filtered.sort_values(by="risk_score", ascending=False).copy()
                if filtered.empty:
                    st.info("No data available")
                else:
                    academic_table = filtered.rename(
                        columns={
                            "student_id": "Student_ID",
                            "risk_level": "Risk_Level",
                            "Primary_Risk_Factors": "Key Risk Factors",
                        }
                    )
                    academic_table["Risk_Score"] = academic_table["risk_score"].apply(format_percent)
                    display_columns = ["Student_ID", "Risk_Level", "Risk_Score", "Key Risk Factors"]
                    available_columns = [col for col in display_columns if col in academic_table.columns]
                    st.dataframe(
                        academic_table[available_columns],
                        width="stretch",
                        hide_index=True,
                        height=300,
                    )
        with right_col:
            with st.container(border=True):
                st.markdown("### ⚠️ Alerts")
                distribution = (
                    df["risk_level"].astype(str).value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
                )
                render_bar_chart(distribution, "Academic risk level distribution", "#d9534f")
                top_factors = extract_top_risk_factors(df.get("Primary_Risk_Factors", pd.Series(dtype="object")))
                render_bar_chart(top_factors, "Top academic risk factors", "#5b8def")


def render_dropout_tab(report_df: pd.DataFrame, actionable_df: pd.DataFrame) -> None:
    render_section_header(
        "Dropout Risk Analysis",
        "Track engagement-related risk indicators and prioritize students who may benefit from early intervention.",
    )

    with st.container(border=True):
        st.markdown("### 📊 Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        high_risk_mask = report_df["Dropout_Risk_Level"].astype(str) == "High"
        metric_col1.metric("Total Students", f"{len(report_df):,}")
        metric_col2.metric("High Risk Count", int(high_risk_mask.sum()))
        metric_col3.metric("Average Probability", format_percent(report_df["Risk_Probability_Value"].mean()))

    with st.container():
        left_col, right_col = st.columns(2, gap="medium")
        with left_col:
            with st.container(border=True):
                st.markdown("### 📋 Data")
                filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.1])
                all_statuses = ["All"] + sorted(report_df["Intervention_Status"].dropna().astype(str).unique().tolist())
                selected_status = filter_col1.selectbox("Intervention Status", all_statuses, key="dropout_status")
                selected_level = filter_col2.selectbox(
                    "Risk Level",
                    ["All", "High", "Medium", "Low"],
                    key="dropout_risk_level",
                )
                student_query = filter_col3.text_input("Search Student_ID", "", key="dropout_student_search")

                filtered = report_df.copy()
                if selected_status != "All":
                    filtered = filtered[filtered["Intervention_Status"].astype(str) == selected_status]
                if selected_level != "All":
                    filtered = filtered[filtered["Dropout_Risk_Level"].astype(str) == selected_level]
                if student_query.strip():
                    filtered = filtered[
                        filtered["Student_ID"].astype(str).str.contains(student_query.strip(), case=False, na=False)
                    ]

                filtered = filtered.sort_values(by="Risk_Probability_Value", ascending=False).copy()
                if filtered.empty:
                    st.info("No data available")
                else:
                    dropout_table = filtered.rename(
                        columns={
                            "Primary_Risk_Factors": "Key Risk Factors",
                            "Intervention_Status": "Intervention Status",
                        }
                    )
                    dropout_table["Risk Probability"] = dropout_table["Risk_Probability_Value"].apply(format_percent)
                    display_columns = [
                        "Student_ID",
                        "Risk Probability",
                        "Intervention Status",
                        "Key Risk Factors",
                    ]
                    available_columns = [col for col in display_columns if col in dropout_table.columns]
                    st.dataframe(
                        dropout_table[available_columns],
                        width="stretch",
                        hide_index=True,
                        height=300,
                    )
        with right_col:
            with st.container(border=True):
                st.markdown("### ⚠️ Alerts")
                risk_counts = (
                    report_df["Dropout_Risk_Level"]
                    .astype(str)
                    .value_counts()
                    .reindex(["High", "Medium", "Low"])
                    .fillna(0)
                    .astype(int)
                )
                render_bar_chart(risk_counts, "Dropout risk level distribution", "#f0ad4e")
                status_counts = report_df["Intervention_Status"].astype(str).value_counts()
                render_bar_chart(status_counts, "Intervention status distribution", "#5bc0de")

                st.markdown("#### Weekly Actionable Interventions")
                if actionable_df.empty:
                    st.info("No data available")
                else:
                    st.dataframe(actionable_df, width="stretch", hide_index=True, height=110)


def render_parent_alerts_tab(alerts_df: pd.DataFrame) -> None:
    render_section_header(
        "Parent Alert System",
        "Unified alerts generated from predictive models. This view combines message outputs only and does not merge student records across datasets.",
    )

    with st.container(border=True):
        st.markdown("### 📊 Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Total Alerts", f"{len(alerts_df):,}")
        metric_col2.metric("High Risk Alerts", int((alerts_df["Risk_Label"] == "High").sum()))
        metric_col3.metric("Academic Alerts", int((alerts_df["Alert_Type"] == "Academic Risk").sum()))
        metric_col4.metric("Dropout Alerts", int((alerts_df["Alert_Type"] == "Dropout Risk").sum()))

    with st.container():
        left_col, right_col = st.columns(2, gap="medium")
        with left_col:
            with st.container(border=True):
                st.markdown("### 📋 Data")
                filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.1])
                selected_type = filter_col1.selectbox(
                    "Alert Type",
                    ["All", "Academic Risk", "Dropout Risk"],
                    key="parent_alert_type",
                )
                selected_level = filter_col2.selectbox(
                    "Risk Level",
                    ["All", "High", "Medium", "Low"],
                    key="parent_alert_risk_level",
                )
                student_query = filter_col3.text_input("Search Student_ID", "", key="parent_alert_student_search")

                filtered = alerts_df.copy()
                if selected_type != "All":
                    filtered = filtered[filtered["Alert_Type"] == selected_type]
                if selected_level != "All":
                    filtered = filtered[filtered["Risk_Label"] == selected_level]
                if student_query.strip():
                    filtered = filtered[
                        filtered["Student_ID"].astype(str).str.contains(student_query.strip(), case=False, na=False)
                    ]

                filtered = filtered.sort_values(by="Risk_Value", ascending=False).copy()
                if filtered.empty:
                    st.info("No data available")
                else:
                    display_df = filtered.copy()
                    display_df["Risk_Value"] = display_df["Risk_Value"].apply(format_percent)
                    st.dataframe(
                        display_df[["Student_ID", "Alert_Type", "Risk_Label", "Risk_Value", "Parent_Message"]],
                        width="stretch",
                        hide_index=True,
                        height=250,
                    )

                    st.download_button(
                        "Download Parent Alerts CSV",
                        data=filtered.to_csv(index=False).encode("utf-8"),
                        file_name="parent_alerts.csv",
                        mime="text/csv",
                    )
        with right_col:
            with st.container(border=True):
                st.markdown("### ⚠️ Alerts")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    type_counts = filtered["Alert_Type"].value_counts()
                    render_bar_chart(type_counts, "Alerts by type", "#5cb85c")
                with chart_col2:
                    level_counts = (
                        filtered["Risk_Label"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
                    )
                    render_bar_chart(level_counts, "Alerts by risk level", "#337ab7")

                st.markdown("#### Alert Details")
                if filtered.empty:
                    st.info("No data available")
                else:
                    detail_options = [
                        f"{row.Student_ID} | {row.Alert_Type} | {row.Risk_Label} | {format_percent(row.Risk_Value)}"
                        for row in filtered.head(12).itertuples(index=False)
                    ]
                    selected_detail = st.selectbox("Select alert", detail_options, key="parent_alert_detail")
                    selected_row = filtered.head(12).reset_index(drop=True).iloc[detail_options.index(selected_detail)]
                    st.caption(f"Key concerns: {simplify_risk_factors(selected_row['Risk_Factors'])}")
                    st.write(selected_row["Parent_Message"])
                    st.write(f"**Recommended Action:** {selected_row['Recommended_Action']}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pass_fail_path = base_dir / "data" / "processed" / "performance" / "student_predictions.csv"
    dropout_path = base_dir / "data" / "processed" / "dropout" / "Student_risk_report.csv"

    apply_global_styles()

    st.title("Student Risk Monitoring Dashboard")
    st.caption("AI-powered early warning system for academic and engagement risks")

    required_paths = [pass_fail_path, dropout_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        st.error("Required processed files are missing:\n- " + "\n- ".join(missing_paths))
        return

    pass_fail_df = load_pass_fail_predictions(base_dir)
    dropout_df = load_dropout_predictions(base_dir)
    actionable_df = load_actionable_dropout_report(base_dir)
    parent_alerts_df = build_parent_alerts(pass_fail_df, dropout_df)

    with st.container(border=True):
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        overview_col1.metric("Academic Records", f"{len(pass_fail_df):,}")
        overview_col2.metric("Dropout Records", f"{len(dropout_df):,}")
        overview_col3.metric("Unified Alerts", f"{len(parent_alerts_df):,}")

    academic_tab, dropout_tab, parent_alerts_tab = st.tabs(
        ["Academic Risk", "Dropout Risk", "Parent Alerts"]
    )

    with academic_tab:
        render_academic_tab(pass_fail_df)

    with dropout_tab:
        render_dropout_tab(dropout_df, actionable_df)

    with parent_alerts_tab:
        render_parent_alerts_tab(parent_alerts_df)


if __name__ == "__main__":
    main()
