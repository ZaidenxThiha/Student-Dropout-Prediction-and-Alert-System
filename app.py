"""Main entry point for the Student Dropout Prediction & Alert System dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_performance_data, load_dropout_data, deduplicate_dropout, load_config

st.set_page_config(
    page_title="Student Dropout Prediction & Alert System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }

h1, h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.02em; }

div[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid #d7e6fb !important;
    padding: 0.8rem 1rem;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--primary-color) !important;
}

.risk-badge-high   { background:#fee2e2; color:#dc2626; padding:2px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.risk-badge-medium { background:#fef9c3; color:#ca8a04; padding:2px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.risk-badge-low    { background:#dcfce7; color:#16a34a; padding:2px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }

.stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #d7e6fb !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Student Risk System")
    st.caption("AI-powered early warning system")
    st.divider()
    st.markdown("### Navigation")
    st.markdown("""
- **Overview** — KPIs & risk summary
- **Performance** — UCI academic model
- **Dropout Alerts** — OULA engagement
- **Student Profile** — Individual deep-dive
- **Model Insights** — Metrics & SHAP
- **Analytics** — What-If & correlations
""")
    st.divider()
    st.markdown("### System Status")

    BASE_DIR = Path(__file__).resolve().parent
    perf_ok = (BASE_DIR / "data/processed/performance/student_predictions.csv").exists()
    drop_ok = (BASE_DIR / "data/processed/dropout/Student_risk_report.csv").exists()
    model_ok = (BASE_DIR / "models/dropout/dropout_xgb_optimized.joblib").exists()

    st.markdown(f"{'[OK]' if perf_ok else '[MISSING]'} Performance data")
    st.markdown(f"{'[OK]' if drop_ok else '[MISSING]'} Dropout data")
    st.markdown(f"{'[OK]' if model_ok else '[MISSING]'} Optimized model")

    config = load_config()
    st.caption(f"Dropout threshold: **{config['dropout']['threshold']}**")
    metrics = config["dropout"].get("metrics", {})
    if metrics:
        st.caption(f"Model — P: {metrics.get('precision','?')} | R: {metrics.get('recall','?')} | AUC: {metrics.get('auc','?')}")


# ─── Main Landing Page ────────────────────────────────────────────────────────
st.title("Student Dropout Prediction & Alert System")
st.caption("Early warning dashboard for educators — academic failure & dropout risk monitoring")

st.divider()

# Load data for landing metrics
try:
    with st.spinner("Loading data..."):
        perf_df = load_performance_data()
        drop_df = load_dropout_data()
        drop_df_dedup = deduplicate_dropout(drop_df)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if perf_df.empty or drop_df.empty:
    st.warning("Some data files are missing. Run `python src/predict.py --task all` to regenerate them.")
    st.stop()

# ─── KPI Cards ───────────────────────────────────────────────────────────────
total_students = len(drop_df_dedup) + len(perf_df)

drop_high = int((drop_df_dedup["Dropout_Risk_Level"].astype(str) == "High").sum())
drop_med  = int((drop_df_dedup["Dropout_Risk_Level"].astype(str) == "Medium").sum())
drop_low  = int((drop_df_dedup["Dropout_Risk_Level"].astype(str) == "Low").sum())

perf_high = int((perf_df["risk_level"].astype(str) == "High").sum()) if "risk_level" in perf_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Student Records", f"{total_students:,}")
col2.metric("High Risk — Dropout",  f"{drop_high:,}", delta=f"{drop_high/max(len(drop_df_dedup),1):.0%} of cohort", delta_color="inverse")
col3.metric("Medium Risk — Dropout", f"{drop_med:,}")
col4.metric("High Risk — Academic",  f"{perf_high:,}", delta=f"{perf_high/max(len(perf_df),1):.0%} of cohort", delta_color="inverse")

st.divider()

# ─── Quick Summary Panels ─────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("### Dropout Risk Summary")
        st.caption(f"OULA dataset — {len(drop_df_dedup):,} unique students monitored")
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("High", drop_high)
        dcol2.metric("Medium", drop_med)
        dcol3.metric("Low", drop_low)

        if "code_module" in drop_df_dedup.columns:
            top_modules = (
                drop_df_dedup[drop_df_dedup["Dropout_Risk_Level"].astype(str) == "High"]
                ["code_module"].value_counts().head(3)
            )
            if not top_modules.empty:
                st.caption("Top at-risk modules: " + ", ".join(f"**{m}** ({c})" for m, c in top_modules.items()))

with col_b:
    with st.container(border=True):
        st.markdown("### Academic Risk Summary")
        st.caption(f"UCI dataset — {len(perf_df):,} student records")
        if "risk_level" in perf_df.columns:
            acol1, acol2, acol3 = st.columns(3)
            acol1.metric("High",   int((perf_df["risk_level"] == "High").sum()))
            acol2.metric("Medium", int((perf_df["risk_level"] == "Medium").sum()))
            acol3.metric("Low",    int((perf_df["risk_level"] == "Low").sum()))
        if "dataset" in perf_df.columns:
            st.caption("Courses: " + ", ".join(perf_df["dataset"].unique()))

st.divider()

# ─── Getting Started ──────────────────────────────────────────────────────────
st.markdown("### Getting Started")
st.markdown("""
Use the **sidebar navigation** to explore the system:

| Page | Description |
|------|-------------|
| Overview | System-wide KPIs, risk distributions, and top-risk student list |
| Performance | Academic pass/fail predictions from UCI student data |
| Dropout Alerts | Dropout risk flags from OULA engagement data |
| Student Profile | Deep-dive per-student explanation with SHAP |
| Model Insights | Confusion matrices, ROC curves, threshold tuning |
| Analytics | Correlation heatmaps, What-If simulator, course analysis |
""")
