"""Performance page — UCI academic pass/fail model."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_performance_data, get_feature_names, get_population_stats
from src.predictor import load_model, predict_single

st.title("Academic Performance Model")
st.caption("Pass/fail predictions from UCI student performance dataset")

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

st.divider()

# ─── Full Student Table ───────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Student Predictions Table")
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

# ─── Charts ───────────────────────────────────────────────────────────────────
st.divider()
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    with st.container(border=True):
        st.markdown("#### G1 vs Fail Probability")
        if "G1" in df.columns and "risk_score" in df.columns:
            fig = px.scatter(
                df, x="G1", y="risk_score",
                color=risk_col if risk_col else None,
                color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
                opacity=0.6,
                labels={"G1": "First Period Grade", "risk_score": "Fail Probability"},
            )
            fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    with st.container(border=True):
        st.markdown("#### G2 vs Fail Probability")
        if "G2" in df.columns and "risk_score" in df.columns:
            fig2 = px.scatter(
                df, x="G2", y="risk_score",
                color=risk_col if risk_col else None,
                color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
                opacity=0.6,
                labels={"G2": "Second Period Grade", "risk_score": "Fail Probability"},
            )
            fig2.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

# ─── Feature Importance ───────────────────────────────────────────────────────
st.divider()
with st.container(border=True):
    st.markdown("#### Feature Importance (Model)")
    with st.spinner("Loading model for feature importance..."):
        model = load_model("performance")
    if model is not None:
        try:
            # Pipeline — extract RF step
            rf = None
            if hasattr(model, "named_steps"):
                rf = model.named_steps.get("model") or model.named_steps.get("classifier")
            else:
                rf = model

            if rf and hasattr(rf, "feature_importances_"):
                # Get feature names from the pipeline's preprocessor output
                try:
                    feat_names_out = model.named_steps["preprocess"].get_feature_names_out()
                    importances = rf.feature_importances_
                    imp_df = pd.DataFrame({"feature": feat_names_out, "importance": importances})
                    # Clean up OHE prefixes for readability
                    imp_df["feature"] = imp_df["feature"].str.replace(r"^(cat__|num__)", "", regex=True)
                    imp_df = imp_df.groupby("feature")["importance"].sum().reset_index()
                    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
                except Exception:
                    importances = rf.feature_importances_
                    imp_df = pd.DataFrame({"feature": [f"feature_{i}" for i in range(len(importances))], "importance": importances})
                    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

                fig3 = px.bar(
                    imp_df, x="importance", y="feature", orientation="h",
                    color="importance", color_continuous_scale="Blues",
                    labels={"importance": "Importance", "feature": "Feature"},
                )
                fig3.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10),
                                   showlegend=False, coloraxis_showscale=False,
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")
    else:
        st.info("Model not loaded.")

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
