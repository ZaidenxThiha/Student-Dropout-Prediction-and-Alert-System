"""Student Profile page — per-student deep-dive with SHAP explanation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from src.data_loader import (
    load_performance_data, load_dropout_data, load_dropout_preprocessed,
    deduplicate_dropout, get_feature_names, get_population_stats,
)
from src.predictor import load_model, load_dropout_features
from src.explainability import ModelExplainer

PERFORMANCE_EXPLANATION_FEATURES = {
    "G1",
    "G2",
    "failures",
    "absences",
    "studytime",
    "goout",
    "Dalc",
    "Walc",
    "Medu",
    "Fedu",
    "health",
    "freetime",
    "age",
}

st.set_page_config(page_title="Student Profile", layout="wide")
st.title("Student Profile")
st.caption("Individual risk deep-dive with SHAP explanation and peer comparison")

# ─── Dataset / Student Selector ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Student Selection")
    dataset_choice = st.selectbox("Dataset", ["Dropout (OULA)", "Academic (UCI)"])

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    if dataset_choice == "Dropout (OULA)":
        raw_df = load_dropout_data()
        df = deduplicate_dropout(raw_df)
        id_col = "Student_ID"
        prob_col = "Risk_Probability_Value"
        risk_col = "Dropout_Risk_Level"
        features = get_feature_names("dropout")
        model_type = "dropout"
        preprocessed = load_dropout_preprocessed()
    else:
        df = load_performance_data()
        id_col = "student_id"
        prob_col = "risk_score"
        risk_col = "risk_level"
        features = get_feature_names("performance")
        model_type = "performance"
        preprocessed = df.copy()

if df.empty:
    st.error("No data available.")
    st.stop()

# ─── Student Selector ──────────────────────────────────────────────────────────
student_ids = df[id_col].astype(str).tolist()
with st.sidebar:
    selected_id = st.selectbox("Student ID", student_ids)

student_row = df[df[id_col].astype(str) == selected_id].iloc[0]
prob = float(student_row.get(prob_col, 0.0)) if pd.notna(student_row.get(prob_col)) else 0.0
risk_level = str(student_row.get(risk_col, "Unknown"))

# ─── Header ───────────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns([2, 1, 1])
with h1:
    st.markdown(f"### Student {selected_id}")
    if "code_module" in student_row.index:
        st.caption(f"Module: **{student_row.get('code_module', 'N/A')}** | Presentation: **{student_row.get('code_presentation', 'N/A')}**")
    if "gender" in student_row.index:
        st.caption(f"Gender: {student_row.get('gender', 'N/A')} | Age Band: {student_row.get('age_band', 'N/A')} | Region: {student_row.get('region', 'N/A')}")

with h2:
    st.metric("Risk Level", risk_level)

with h3:
    st.metric("Risk Probability", f"{prob:.1%}")

st.divider()

# ─── Gauge Chart ──────────────────────────────────────────────────────────────
gauge_col, explain_col = st.columns([1, 2])

with gauge_col:
    with st.container(border=True):
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626" if risk_level == "High" else "#f59e0b" if risk_level == "Medium" else "#16a34a"},
                "steps": [
                    {"range": [0, 36], "color": "#dcfce7"},
                    {"range": [36, 51], "color": "#fef9c3"},
                    {"range": [51, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": prob * 100},
            },
            title={"text": "Risk Score"},
        ))
        fig_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20),
                                 paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─── SHAP Explanation ──────────────────────────────────────────────────────────
with explain_col:
    with st.container(border=True):
        st.markdown("#### Risk Explanation")
        with st.spinner("Computing explanation..."):
            pop_stats = get_population_stats(model_type)
            model = load_model(model_type)
            feat_names = load_dropout_features() if model_type == "dropout" else features

            explanation = None
            explainer = None
            student_row_for_explainer = student_row  # may be overridden for performance pipeline
            shap_class_index = 1 if model_type == "dropout" else 0
            allowed_features = None if model_type == "dropout" else PERFORMANCE_EXPLANATION_FEATURES

            if model and not preprocessed.empty:
                feat_data = preprocessed[[f for f in feat_names if f in preprocessed.columns]].dropna()
                try:
                    if model_type == "performance" and hasattr(model, "named_steps"):
                        # Unwrap sklearn Pipeline: extract RF and transform data
                        preprocessor_step = (
                            model.named_steps.get("preprocess")
                            or model.named_steps.get("preprocessor")
                            or model.named_steps.get("prep")
                        )
                        rf_step = (
                            model.named_steps.get("model")
                            or model.named_steps.get("classifier")
                            or model.named_steps.get("clf")
                        )
                        if preprocessor_step is not None and rf_step is not None:
                            X_transformed = preprocessor_step.transform(feat_data)
                            # ColumnTransformer with OHE returns sparse — convert to dense
                            if hasattr(X_transformed, "toarray"):
                                X_transformed = X_transformed.toarray()
                            X_transformed = np.array(X_transformed, dtype=float)

                            raw_names = list(preprocessor_step.get_feature_names_out())
                            # Strip num__/cat__ prefixes added by ColumnTransformer
                            clean_names = [
                                n.replace("num__", "").replace("cat__", "") for n in raw_names
                            ]
                            X_shap = pd.DataFrame(X_transformed, columns=clean_names)

                            # Transform the selected student's raw features
                            student_raw = {f: student_row.get(f, None) for f in feat_names}
                            student_df = pd.DataFrame([student_raw])
                            student_transformed = preprocessor_step.transform(student_df)
                            if hasattr(student_transformed, "toarray"):
                                student_transformed = student_transformed.toarray()
                            student_transformed = np.array(student_transformed, dtype=float)
                            student_row_shap = pd.Series(student_transformed[0], index=clean_names)
                            student_row_for_explainer = student_row_shap

                            # Build population stats for transformed features
                            pop_stats_shap = {
                                cn: {"mean": float(X_shap[cn].mean()), "std": float(X_shap[cn].std())}
                                for cn in clean_names
                            }

                            explainer = ModelExplainer(
                                rf_step,
                                X_shap,
                                clean_names,
                                class_index=shap_class_index,
                                allowed_features=allowed_features,
                            )
                            explanation = explainer.get_student_explanation(student_row_shap, pop_stats_shap)
                        else:
                            # Pipeline structure unknown — try direct
                            explainer = ModelExplainer(
                                model,
                                feat_data,
                                feat_names,
                                class_index=shap_class_index,
                                allowed_features=allowed_features,
                            )
                            explanation = explainer.get_student_explanation(student_row, pop_stats)
                    else:
                        explainer = ModelExplainer(
                            model,
                            feat_data,
                            feat_names,
                            class_index=shap_class_index,
                            allowed_features=allowed_features,
                        )
                        explanation = explainer.get_student_explanation(student_row, pop_stats)
                except Exception as e:
                    st.info(f"Explanation unavailable: {e}")

            if explanation:
                st.markdown(f"**Summary:** {explanation['summary']}")
                st.divider()

                if explanation["risk_factors"]:
                    st.markdown("**Areas of Concern:**")
                    for rf in explanation["risk_factors"][:4]:
                        st.markdown(f"- {rf['explanation']}")

                if explanation["protective_factors"]:
                    st.markdown("**Positive Signals:**")
                    for pf in explanation["protective_factors"][:2]:
                        st.markdown(f"- {pf['explanation']}")

                if explanation["interventions"]:
                    st.markdown("**Recommended Actions:**")
                    for i, action in enumerate(explanation["interventions"], 1):
                        st.markdown(f"{i}. {action}")
            elif not model:
                st.info("Model not available for explanation.")

# ─── SHAP Waterfall ────────────────────────────────────────────────────────────
st.divider()
wf_col, radar_col = st.columns(2)

with wf_col:
    with st.container(border=True):
        st.markdown("#### SHAP Feature Impact")
        if explainer is not None:
            try:
                fig_wf = explainer.plot_waterfall(student_row_for_explainer)
                if fig_wf:
                    st.plotly_chart(fig_wf, use_container_width=True)
                else:
                    st.info("Waterfall chart unavailable")
            except Exception:
                st.info("Waterfall chart unavailable")
        else:
            st.info("Model not available")

# ─── Radar Chart: Student vs Class Average ─────────────────────────────────────
with radar_col:
    with st.container(border=True):
        st.markdown("#### Student vs Class Average (Numeric Features)")
        numeric_feats = [f for f in feat_names if f in student_row.index]
        pop_stats = get_population_stats(model_type)

        radar_feats = [f for f in numeric_feats if f in pop_stats][:7]
        if radar_feats:
            student_vals, avg_vals = [], []
            for f in radar_feats:
                sv = student_row.get(f, pop_stats[f]["mean"])
                try:
                    sv = float(sv)
                except (TypeError, ValueError):
                    sv = pop_stats[f]["mean"]
                student_vals.append(sv)
                avg_vals.append(pop_stats[f]["mean"])

            labels = [f.replace("_", " ").title() for f in radar_feats]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=student_vals, theta=labels, fill="toself",
                                                  name="This Student", line_color="#6366f1"))
            fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=labels, fill="toself",
                                                  name="Class Average", line_color="#94a3b8",
                                                  fillcolor="rgba(148,163,184,0.2)"))
            fig_radar.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10),
                                     polar=dict(radialaxis=dict(visible=True)),
                                     paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("No numeric features available for radar chart")

# ─── Parent Notification ──────────────────────────────────────────────────────
st.divider()
with st.container(border=True):
    st.markdown("#### Parent / Guardian Notification")
    st.caption("Review and send this message to the student's parent or guardian.")

    if explanation and explanation.get("parent_message"):
        message = explanation["parent_message"]
        edited_message = st.text_area(
            "Message (editable before sending)",
            value=message,
            height=380,
            key="parent_msg",
        )
        st.download_button(
            "Download as Text File",
            data=edited_message.encode(),
            file_name=f"parent_notification_{selected_id}.txt",
            mime="text/plain",
        )
    else:
        st.info("No explanation available to generate a parent notification.")

# ─── Similar Students ─────────────────────────────────────────────────────────
st.divider()
with st.container(border=True):
    st.markdown("#### Similar Students (Nearest Neighbours)")
    numeric_cols = [f for f in feat_names if f in preprocessed.columns]
    if numeric_cols and len(preprocessed) > 5:
        try:
            feat_mat = preprocessed[numeric_cols].fillna(0).values
            student_vec_df = df[df[id_col].astype(str) == selected_id]
            student_numeric = student_vec_df[[f for f in numeric_cols if f in student_vec_df.columns]].fillna(0).values

            if student_numeric.shape[1] == feat_mat.shape[1]:
                knn = NearestNeighbors(n_neighbors=6)
                knn.fit(feat_mat)
                _, indices = knn.kneighbors(student_numeric)
                neighbours = preprocessed.iloc[indices[0][1:6]]

                show_cols = [c for c in [id_col if id_col in neighbours.columns else None,
                                          "code_module", "final_result", "total_clicks",
                                          "active_days", "avg_score"] if c and c in neighbours.columns]
                if show_cols:
                    st.dataframe(neighbours[show_cols], hide_index=True, use_container_width=True)
                else:
                    st.dataframe(neighbours[numeric_cols], hide_index=True, use_container_width=True)
            else:
                st.info("Feature mismatch — cannot compute neighbours.")
        except Exception as e:
            st.info(f"Similar students unavailable: {e}")
    else:
        st.info("Insufficient data for neighbour comparison.")
