"""Performance page — UCI academic pass/fail model."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_performance_data, get_feature_names, get_population_stats
from src.predictor import load_model, predict_single

st.title("Academic Performance Model")
st.caption("Academic failure risk from the UCI Student Performance dataset — independent from the Dropout Engagement model")

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading performance data..."):
    df = load_performance_data()

if df.empty:
    st.error("Performance data not found. Run `python src/predict.py --task performance`.")
    st.stop()

# ─── Summary Metrics ──────────────────────────────────────────────────────────
risk_col = "risk_level" if "risk_level" in df.columns else None
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Students", f"{len(df):,}")
if risk_col:
    m2.metric("High Risk", int((df[risk_col] == "High").sum()))
    m3.metric("Medium Risk", int((df[risk_col] == "Medium").sum()))
    m4.metric("Low Risk", int((df[risk_col] == "Low").sum()))

# ─── Charts ───────────────────────────────────────────────────────────────────
st.divider()
RISK_COLORS = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}
RISK_ORDER = ["Low", "Medium", "High"]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    with st.container(border=True):
        st.markdown("#### Fail Probability Distribution")
        if "risk_score" in df.columns:
            fig = px.histogram(
                df, x="risk_score", nbins=30,
                color=risk_col if risk_col else None,
                color_discrete_map=RISK_COLORS,
                labels={"risk_score": "Fail Probability"},
            )
            fig.add_vline(x=0.50, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Medium ≥ 0.50", annotation_position="top")
            fig.add_vline(x=0.65, line_dash="dash", line_color="#dc2626",
                          annotation_text="High ≥ 0.65", annotation_position="top")
            fig.update_layout(height=320, margin=dict(t=56, b=10, l=10, r=10),
                              bargap=0.05, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Risk")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("risk_score unavailable")

with chart_col2:
    with st.container(border=True):
        st.markdown("#### Grade Trajectory (G1 → G2) by Risk")
        if {"G1", "G2"}.issubset(df.columns):
            fig2 = px.scatter(
                df, x="G1", y="G2",
                color=risk_col if risk_col else None,
                color_discrete_map=RISK_COLORS,
                opacity=0.65,
                hover_data=[c for c in ["student_id", "failures", "absences", "risk_score"] if c in df.columns],
                labels={"G1": "First Period Grade", "G2": "Second Period Grade"},
            )
            g_max = float(max(df["G1"].max(), df["G2"].max()))
            fig2.add_shape(type="line", x0=0, y0=0, x1=g_max, y1=g_max,
                           line=dict(color="#94a3b8", dash="dot", width=1))
            fig2.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
            fig2.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               legend_title_text="Risk")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("G1/G2 unavailable")

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    with st.container(border=True):
        st.markdown("#### G2 Grade vs Absences by Risk")
        if {"G2", "absences"}.issubset(df.columns) and risk_col:
            fig3 = px.scatter(
                df, x="absences", y="G2",
                color=risk_col, category_orders={risk_col: RISK_ORDER},
                color_discrete_map=RISK_COLORS, opacity=0.65,
                hover_data=[c for c in ["student_id", "G1", "failures", "risk_score"] if c in df.columns],
                labels={"absences": "Absences", "G2": "Second Period Grade"},
            )
            fig3.add_hline(y=10, line_dash="dot", line_color="#94a3b8",
                           annotation_text="Passing (G2 = 10)", annotation_position="bottom right")
            fig3.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
            fig3.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               legend_title_text="Risk")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("G2 / absences unavailable")

with chart_col4:
    with st.container(border=True):
        st.markdown("#### Absences by Risk Level")
        if "absences" in df.columns and risk_col:
            fig4 = px.box(
                df, x=risk_col, y="absences",
                color=risk_col, category_orders={risk_col: RISK_ORDER},
                color_discrete_map=RISK_COLORS, points="outliers",
                labels={risk_col: "Risk Level", "absences": "Absences"},
            )
            fig4.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                               showlegend=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("absences column unavailable")

bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    with st.container(border=True):
        st.markdown("#### Feature Correlation Heatmap")
        corr_cols = [c for c in ["G1", "G2", "absences", "failures", "studytime",
                                   "goout", "Dalc", "Walc", "health", "age", "risk_score"]
                     if c in df.columns]
        if len(corr_cols) >= 3:
            corr = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
            fig5 = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                labels=dict(color="Corr"),
            )
            fig5.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation")

with bottom_col2:
    with st.container(border=True):
        st.markdown("#### Feature Importance")
        with st.spinner("Loading model for feature importance..."):
            model = load_model("performance")
        if model is not None:
            try:
                rf = None
                if hasattr(model, "named_steps"):
                    rf = model.named_steps.get("model") or model.named_steps.get("classifier")
                else:
                    rf = model

                if rf and hasattr(rf, "feature_importances_"):
                    try:
                        feat_names_out = model.named_steps["preprocess"].get_feature_names_out()
                        importances = rf.feature_importances_
                        imp_df = pd.DataFrame({"feature": feat_names_out, "importance": importances})
                        imp_df["feature"] = imp_df["feature"].str.replace(r"^(cat__|num__)", "", regex=True)
                        imp_df = imp_df.groupby("feature")["importance"].sum().reset_index()
                        imp_df = imp_df.sort_values("importance", ascending=False).head(12)
                    except Exception:
                        importances = rf.feature_importances_
                        imp_df = pd.DataFrame({"feature": [f"feature_{i}" for i in range(len(importances))], "importance": importances})
                        imp_df = imp_df.sort_values("importance", ascending=False).head(12)

                    fig3 = px.bar(
                        imp_df, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="Blues",
                        labels={"importance": "Importance", "feature": "Feature"},
                    )
                    fig3.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                                       showlegend=False, coloraxis_showscale=False,
                                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.info(f"Feature importance unavailable: {e}")
        else:
            st.info("Model not loaded.")

# ─── Student Table ────────────────────────────────────────────────────────────
st.divider()
with st.expander("Student Predictions Table", expanded=False):
    filt_col1, filt_col2, filt_col3 = st.columns([1, 1, 2])
    risk_sel = filt_col1.selectbox("Risk Level", ["All", "High", "Medium", "Low"], key="perf_risk")
    dataset_sel = filt_col2.selectbox("Dataset", ["All"] + sorted(df["dataset"].unique().tolist()) if "dataset" in df.columns else ["All"])
    student_q = filt_col3.text_input("Search Student ID", key="perf_search")

    filtered = df.copy()
    if risk_sel != "All" and risk_col:
        filtered = filtered[filtered[risk_col] == risk_sel]
    if dataset_sel != "All" and "dataset" in filtered.columns:
        filtered = filtered[filtered["dataset"] == dataset_sel]
    if student_q.strip() and "student_id" in filtered.columns:
        filtered = filtered[filtered["student_id"].astype(str).str.contains(student_q.strip(), case=False)]

    if risk_col:
        filtered = filtered.sort_values("risk_score", ascending=False)

    show_cols = [c for c in ["student_id", "dataset", "G1", "G2", "G3", "absences",
                              "failures", "predicted_outcome", "risk_score", risk_col,
                              "Primary_Risk_Factors"] if c and c in filtered.columns]

    display = filtered[show_cols].copy()
    if "risk_score" in display.columns:
        display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.1%}")

    st.dataframe(display, hide_index=True, use_container_width=True, height=320)
    st.download_button(
        "Download All Predictions CSV",
        data=filtered.to_csv(index=False).encode(),
        file_name="performance_predictions.csv",
        mime="text/csv",
    )

# ─── Manual Prediction Form ───────────────────────────────────────────────────
st.divider()
with st.expander("Manual Prediction — Enter Student Data"):
    st.caption("Input student attributes to get a live pass/fail prediction")
    f1, f2, f3 = st.columns(3)
    g1_val = f1.number_input("G1 (first period grade)", 0, 20, 10)
    g2_val = f2.number_input("G2 (second period grade)", 0, 20, 10)
    abs_val = f3.number_input("Absences", 0, 93, 5)
    f4, f5, f6 = st.columns(3)
    fail_val = f4.number_input("Prior failures", 0, 4, 0)
    study_val = f5.selectbox("Study time (1=<2h, 4=>10h)", [1, 2, 3, 4], index=1)
    higher_val = f6.selectbox("Aims for higher education?", ["yes", "no"])
    dataset_val = st.selectbox("Course dataset", ["math", "portuguese"])

    if st.button("Predict"):
        student = {
            "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
            "Pstatus": "T", "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other",
            "reason": "course", "guardian": "mother", "traveltime": 1,
            "studytime": study_val, "failures": fail_val, "schoolsup": "no",
            "famsup": "yes", "paid": "no", "activities": "no", "nursery": "yes",
            "higher": higher_val, "internet": "yes", "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1,
            "health": 3, "absences": abs_val, "G1": g1_val, "G2": g2_val,
            "dataset": dataset_val,
        }
        with st.spinner("Predicting..."):
            result = predict_single(student, "performance")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Prediction", result["prediction"])
        rc2.metric("Fail Probability", f"{result['probability']:.1%}")
        rc3.metric("Risk Level", result["risk_level"])
