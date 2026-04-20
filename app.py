"""Main entry point for the Student Dropout Prediction & Alert System dashboard."""

import streamlit as st

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

navigation = st.navigation(pages, position="sidebar")
navigation.run()
