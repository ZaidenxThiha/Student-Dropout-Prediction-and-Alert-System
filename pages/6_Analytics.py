"""Analytics page — correlations, course-level analysis, and What-If simulator."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_performance_data, load_dropout_data, load_dropout_preprocessed,
    deduplicate_dropout, get_feature_names, get_population_stats,
)
from src.predictor import load_model, load_dropout_features, predict_single

st.title("Analytics")
st.caption("Deep-dive analytics: risk factor analysis, course-level stats, correlations, and What-If simulator")

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    perf_df = load_performance_data()
    drop_raw = load_dropout_data()
    drop_df = deduplicate_dropout(drop_raw)
    preprocessed = load_dropout_preprocessed()

tabs = st.tabs(["Risk Factor Analysis", "Course Analysis", "Correlation Heatmap", "What-If Simulator"])

# ─── Tab 1: Risk Factor Analysis ──────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### Risk Factor Frequency by Risk Level")

    for df_src, risk_col, label in [
        (drop_df, "Dropout_Risk_Level", "Dropout"),
        (perf_df, "risk_level", "Academic"),
    ]:
        if df_src.empty or "Primary_Risk_Factors" not in df_src.columns:
            continue

        st.markdown(f"**{label} Model**")
        rows = []
        for _, row in df_src.iterrows():
            rl = str(row.get(risk_col, "Unknown"))
            for f in str(row.get("Primary_Risk_Factors", "")).split("|"):
                f = f.strip()
                if f and f.upper() != "N/A":
                    rows.append({"risk_level": rl, "factor": f})

        if rows:
            fdf = pd.DataFrame(rows)
            top_factors = fdf["factor"].value_counts().head(8).index.tolist()
            fdf = fdf[fdf["factor"].isin(top_factors)]
            grouped = fdf.groupby(["factor", "risk_level"]).size().reset_index(name="count")
            fig = px.bar(
                grouped, x="factor", y="count", color="risk_level", barmode="group",
                color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
            )
            fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                               xaxis_tickangle=-30, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# ─── Tab 2: Course Analysis ────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Dropout Rate and Avg Engagement by Course Module")

    if not drop_raw.empty and "code_module" in drop_raw.columns:
        module_stats = drop_raw.groupby("code_module").agg(
            total_students=("Student_ID", "count"),
            dropout_rate=("target", "mean") if "target" in drop_raw.columns else ("ML_Prediction", "mean"),
            avg_clicks=("total_clicks", "mean") if "total_clicks" in drop_raw.columns else ("total_clicks", lambda x: np.nan),
            avg_active_days=("active_days", "mean") if "active_days" in drop_raw.columns else ("active_days", lambda x: np.nan),
        ).reset_index()

        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                fig1 = px.bar(
                    module_stats, x="code_module", y="dropout_rate",
                    color="dropout_rate", color_continuous_scale="Reds",
                    labels={"dropout_rate": "Dropout/At-Risk Rate", "code_module": "Module"},
                    title="At-Risk Rate by Module",
                )
                fig1.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10),
                                    coloraxis_showscale=False,
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig1, use_container_width=True)

        with c2:
            with st.container(border=True):
                fig2 = px.bar(
                    module_stats, x="code_module", y="avg_clicks",
                    color="avg_clicks", color_continuous_scale="Blues",
                    labels={"avg_clicks": "Avg Total Clicks", "code_module": "Module"},
                    title="Avg VLE Engagement by Module",
                )
                fig2.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10),
                                    coloraxis_showscale=False,
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

        with st.container(border=True):
            st.markdown("**Module Summary Table**")
            st.dataframe(module_stats, hide_index=True, use_container_width=True)
    else:
        st.info("No course-level data available.")

# ─── Tab 3: Correlation Heatmap ────────────────────────────────────────────────
with tabs[2]:
    heatmap_choice = st.selectbox("Dataset", ["Dropout (OULA)", "Academic (UCI)"])

    if heatmap_choice == "Dropout (OULA)":
        numeric_feats = get_feature_names("dropout")
        source = preprocessed if not preprocessed.empty else drop_df
    else:
        numeric_feats = ["G1", "G2", "G3", "absences", "failures", "studytime",
                          "goout", "Dalc", "Walc", "health", "freetime"]
        source = perf_df

    available = [f for f in numeric_feats if f in source.columns]
    if available and len(source) > 5:
        corr = source[available].corr()
        fig_heat = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto",
        )
        fig_heat.update_layout(height=450, margin=dict(t=30, b=10, l=10, r=10),
                                title="Feature Correlation Matrix",
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Insufficient data for correlation matrix.")

# ─── Tab 4: What-If Simulator ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### What-If Risk Simulator")
    st.caption("Adjust feature values to see real-time risk prediction updates")

    sim_model = st.selectbox("Model", ["Dropout (OULA)", "Academic (UCI)"], key="sim_model")

    if sim_model == "Dropout (OULA)":
        pop_stats = get_population_stats("dropout")
        feat_names = load_dropout_features()
        model_type = "dropout"

        defaults = {
            "total_clicks": pop_stats.get("total_clicks", {}).get("mean", 150),
            "active_days": pop_stats.get("active_days", {}).get("mean", 20),
            "relative_engagement": pop_stats.get("relative_engagement", {}).get("mean", 0.0),
            "avg_score": pop_stats.get("avg_score", {}).get("mean", 65.0),
            "avg_lateness": pop_stats.get("avg_lateness", {}).get("mean", 2.0),
            "num_of_prev_attempts": 0,
            "studied_credits": 60,
        }

        scol1, scol2 = st.columns(2)
        with scol1:
            total_clicks = st.slider("Total VLE Clicks", 0, 2000, int(defaults["total_clicks"]))
            active_days = st.slider("Active Days", 0, 100, int(defaults["active_days"]))
            avg_score = st.slider("Avg Assessment Score", 0.0, 100.0, float(defaults["avg_score"]))
            num_prev = st.slider("Prior Attempts", 0, 5, int(defaults["num_of_prev_attempts"]))
        with scol2:
            avg_lateness = st.slider("Avg Submission Lateness (days)", -30.0, 30.0, float(defaults["avg_lateness"]))
            studied_credits = st.slider("Studied Credits", 30, 240, int(defaults["studied_credits"]))
            highest_education = st.selectbox("Highest Education", [
                "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                "HE Qualification", "Post Graduate Qualification"
            ], index=2)
            age_band = st.selectbox("Age Band", ["0-35", "35-55", "55<="], index=0)

        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            imd_band = st.selectbox("Deprivation Band (IMD)", [
                "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
                "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
            ], index=5)
        with dcol2:
            gender = st.selectbox("Gender", ["M", "F"], index=0)
        with dcol3:
            disability = st.selectbox("Disability", ["N", "Y"], index=0)

        mean_clicks = pop_stats.get("total_clicks", {}).get("mean", 150)
        relative_engagement = (total_clicks - mean_clicks) / max(mean_clicks, 1)

        student_data = {
            "total_clicks": total_clicks,
            "active_days": active_days,
            "relative_engagement": relative_engagement,
            "avg_score": avg_score,
            "avg_lateness": avg_lateness,
            "num_of_prev_attempts": num_prev,
            "studied_credits": studied_credits,
            "avg_clicks_per_day": total_clicks / max(active_days, 1),
            "highest_education": highest_education,
            "imd_band": imd_band,
            "age_band": age_band,
            "gender": gender,
            "disability": disability,
        }
    else:
        pop_stats = get_population_stats("performance")
        model_type = "performance"
        scol1, scol2 = st.columns(2)
        with scol1:
            g1 = st.slider("G1 (first period grade)", 0, 20, 10)
            g2 = st.slider("G2 (second period grade)", 0, 20, 10)
            failures = st.slider("Prior failures", 0, 4, 0)
        with scol2:
            absences = st.slider("Absences", 0, 93, 5)
            studytime = st.slider("Study time (1-4)", 1, 4, 2)
            higher = st.selectbox("Aims for higher education?", ["yes", "no"])

        student_data = {
            "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
            "Pstatus": "T", "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other",
            "reason": "course", "guardian": "mother", "traveltime": 1,
            "studytime": studytime, "failures": failures, "schoolsup": "no",
            "famsup": "yes", "paid": "no", "activities": "no", "nursery": "yes",
            "higher": higher, "internet": "yes", "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1,
            "health": 3, "absences": absences, "G1": g1, "G2": g2,
        }

    with st.container(border=True):
        result = predict_single(student_data, model_type)
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", result["prediction"])
        r2.metric("Risk Probability", f"{result['probability']:.1%}")
        r3.metric("Risk Level", result["risk_level"])

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability"] * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626" if result["risk_level"] == "High" else "#f59e0b" if result["risk_level"] == "Medium" else "#16a34a"},
                "steps": [
                    {"range": [0, 36], "color": "#dcfce7"},
                    {"range": [36, 51], "color": "#fef9c3"},
                    {"range": [51, 100], "color": "#fee2e2"},
                ],
            },
            title={"text": "Live Risk Score"},
        ))
        fig_g.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20),
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

    # Download report
    st.download_button(
        "Download What-If Report",
        data=pd.DataFrame([{**student_data, **result}]).to_csv(index=False).encode(),
        file_name="whatif_report.csv",
        mime="text/csv",
    )
