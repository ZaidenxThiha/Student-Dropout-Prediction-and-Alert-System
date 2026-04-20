"""Main entry point for the Student Dropout Prediction & Alert System dashboard."""

from pathlib import Path

import streamlit as st

from src.data_loader import load_config

st.set_page_config(
    page_title="Student Dropout Prediction & Alert System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

pages = {
    "Dashboard": [
        st.Page("pages/1_Overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("pages/2_Performance.py", title="Performance", icon=":material/school:"),
        st.Page("pages/3_Dropout_Alerts.py", title="Dropout Alerts", icon=":material/notification_important:"),
        st.Page("pages/4_Student_Profile.py", title="Student Profile", icon=":material/person_search:"),
        st.Page("pages/5_Model_Insights.py", title="Model Insights", icon=":material/insights:"),
        st.Page("pages/6_Analytics.py", title="Analytics", icon=":material/analytics:"),
    ],
}

with st.sidebar:
    st.markdown("## Student Risk System")
    st.caption("AI-powered early warning system")
    st.divider()
    st.markdown("### System Status")

    base_dir = Path(__file__).resolve().parent
    perf_ok = (base_dir / "data/processed/performance/student_predictions.csv").exists()
    drop_ok = (base_dir / "data/processed/dropout/Student_risk_report.csv").exists()
    model_ok = (base_dir / "models/dropout/dropout_xgb_optimized.joblib").exists()

    st.markdown("**Academic Model (UCI)**")
    st.markdown(f"{'[OK]' if perf_ok else '[MISSING]'} Performance data")
    st.markdown("")
    st.markdown("**Dropout Model (OULA)**")
    st.markdown(f"{'[OK]' if drop_ok else '[MISSING]'} Dropout data")
    st.markdown(f"{'[OK]' if model_ok else '[MISSING]'} Optimized XGBoost")

    config = load_config()
    st.caption(f"Dropout threshold: **{config['dropout']['threshold']}**")
    metrics = config["dropout"].get("metrics", {})
    if metrics:
        st.caption(
            f"Model — P: {metrics.get('precision', '?')} | "
            f"R: {metrics.get('recall', '?')} | "
            f"AUC: {metrics.get('auc', '?')}"
        )

navigation = st.navigation(pages, position="sidebar")
navigation.run()
