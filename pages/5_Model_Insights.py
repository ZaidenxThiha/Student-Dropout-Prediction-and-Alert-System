"""Model Insights page — metrics, ROC curves, confusion matrices, SHAP beeswarm, threshold tuning."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)

from src.data_loader import (
    load_performance_data, load_dropout_preprocessed, load_dropout_data,
    deduplicate_dropout, get_feature_names, load_config,
)
from src.predictor import load_model, load_dropout_features

st.title("Model Insights")
st.caption("Evaluation metrics, ROC curves, confusion matrices, and interactive threshold analysis")

config = load_config()

# ─── Load Test Data ────────────────────────────────────────────────────────────
@st.cache_data
def get_dropout_eval():
    """Use dropout_preprocessed.csv with a stratified split — X_test.csv has old feature schema."""
    from sklearn.model_selection import train_test_split
    preprocessed = load_dropout_preprocessed()
    if preprocessed.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    features = get_feature_names("dropout")
    available = [f for f in features if f in preprocessed.columns]
    X = preprocessed[available].fillna(0)
    y = preprocessed["target"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


@st.cache_data
def get_performance_eval():
    """Use a stratified split so both classes appear in the test set."""
    from sklearn.model_selection import train_test_split
    df = load_performance_data()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    features = get_feature_names("performance")
    available = [f for f in features if f in df.columns]
    y_col = "pass" if "pass" in df.columns else "actual_pass" if "actual_pass" in df.columns else None
    if y_col is None:
        return pd.DataFrame(), pd.Series(dtype=int)
    X = df[available]
    y = df[y_col].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# ─── Model Comparison Table ────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### Model Comparison")
    metrics_data = {
        "Model": ["Academic (Random Forest)", "Dropout (XGBoost — Optimized)"],
        "Accuracy": ["~0.82", "~0.80"],
        "Precision": ["~0.83", f"{config['dropout'].get('metrics', {}).get('precision', 0.727):.3f}"],
        "Recall": ["~0.87", f"{config['dropout'].get('metrics', {}).get('recall', 0.827):.3f}"],
        "F1": ["~0.85", f"{config['dropout'].get('metrics', {}).get('f1', 0.774):.3f}"],
        "AUC": ["~0.89", f"{config['dropout'].get('metrics', {}).get('auc', 0.843):.3f}"],
        "Dataset": ["UCI Student Performance", "OULA Virtual Learning"],
        "Threshold": [config["performance"].get("threshold", 0.5), config["dropout"].get("threshold", 0.36)],
    }
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

insight_tabs = st.tabs(["Dropout Model", "Academic Model", "SHAP"])

with insight_tabs[0]:
    with st.spinner("Loading dropout model evaluation..."):
        X_d, y_d = get_dropout_eval()
        dropout_model = load_model("dropout")
        drop_features = load_dropout_features()

    if dropout_model is not None and not X_d.empty and len(y_d) > 0:
        available_feats = [f for f in drop_features if f in X_d.columns]
        X_d_feat = X_d[available_feats].fillna(0)
        try:
            d_proba = dropout_model.predict_proba(X_d_feat)[:, 1]
            threshold = config["dropout"].get("threshold", 0.36)
            d_preds = (d_proba >= threshold).astype(int)

            dcol1, dcol2 = st.columns(2)

            with dcol1:
                with st.container(border=True):
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_d, d_preds, labels=[0, 1])
                    fig_cm = px.imshow(
                        cm, text_auto=True, aspect="auto",
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["On-Track", "At-Risk"], y=["On-Track", "At-Risk"],
                    )
                    fig_cm.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10),
                                          paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cm, use_container_width=True)

            with dcol2:
                with st.container(border=True):
                    st.markdown("#### ROC Curve")
                    fpr, tpr, _ = roc_curve(y_d, d_proba)
                    auc_val = roc_auc_score(y_d, d_proba)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC AUC={auc_val:.3f}", line_color="#6366f1"))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False))
                    fig_roc.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10),
                                           xaxis_title="FPR", yaxis_title="TPR",
                                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_roc, use_container_width=True)

            with st.container(border=True):
                st.markdown("#### Interactive Threshold Tuning")
                slider_thresh = st.slider("Classification Threshold", 0.10, 0.80, float(threshold), 0.01, key="dropout_thresh")
                live_preds = (d_proba >= slider_thresh).astype(int)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Precision", f"{precision_score(y_d, live_preds, zero_division=0):.3f}")
                s2.metric("Recall",    f"{recall_score(y_d, live_preds, zero_division=0):.3f}")
                s3.metric("F1",        f"{f1_score(y_d, live_preds, zero_division=0):.3f}")
                s4.metric("Accuracy",  f"{accuracy_score(y_d, live_preds):.3f}")

                prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_d, d_proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec_arr, y=prec_arr, name="PR Curve", line_color="#6366f1"))
                if len(thresh_arr) > 0:
                    idx = np.argmin(np.abs(thresh_arr - slider_thresh))
                    fig_pr.add_trace(go.Scatter(
                        x=[rec_arr[idx]], y=[prec_arr[idx]],
                        mode="markers", marker=dict(size=12, color="#dc2626"),
                        name=f"t={slider_thresh:.2f}",
                    ))
                fig_pr.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10),
                                      xaxis_title="Recall", yaxis_title="Precision",
                                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pr, use_container_width=True)

        except Exception as e:
            st.error(f"Error evaluating dropout model: {e}")
    else:
        st.info("Dropout model or test data not available.")

with insight_tabs[1]:
    with st.spinner("Loading academic model evaluation..."):
        X_p, y_p = get_performance_eval()
        perf_model = load_model("performance")

    if perf_model is not None and not X_p.empty and len(y_p) > 0:
        try:
            p_proba = perf_model.predict_proba(X_p)[:, 1]
            p_preds = (p_proba >= 0.5).astype(int)

            pc1, pc2 = st.columns(2)

            with pc1:
                with st.container(border=True):
                    st.markdown("#### Confusion Matrix")
                    cm_p = confusion_matrix(y_p, p_preds, labels=[0, 1])
                    fig_cmp = px.imshow(
                        cm_p, text_auto=True, aspect="auto",
                        color_continuous_scale="Blues",
                        x=["Fail", "Pass"], y=["Fail", "Pass"],
                    )
                    fig_cmp.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10),
                                           paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cmp, use_container_width=True)

            with pc2:
                with st.container(border=True):
                    st.markdown("#### ROC Curve")
                    fpr_p, tpr_p, _ = roc_curve(y_p, p_proba)
                    auc_p = roc_auc_score(y_p, p_proba)
                    fig_roc_p = go.Figure()
                    fig_roc_p.add_trace(go.Scatter(x=fpr_p, y=tpr_p, name=f"ROC AUC={auc_p:.3f}", line_color="#10b981"))
                    fig_roc_p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False))
                    fig_roc_p.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10),
                                             xaxis_title="FPR", yaxis_title="TPR",
                                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_roc_p, use_container_width=True)

        except Exception as e:
            st.error(f"Error evaluating performance model: {e}")
    else:
        st.info("Academic model or test data not available.")

with insight_tabs[2]:
    from src.explainability import ModelExplainer, SHAP_AVAILABLE

    if not SHAP_AVAILABLE:
        st.warning("Install `shap` to enable beeswarm plots.")
    else:
        bee_model_choice = st.selectbox("Model", ["Dropout", "Academic"], key="bee_model")
        if st.button("Generate Beeswarm"):
            if bee_model_choice == "Dropout":
                m = load_model("dropout")
                shap_model = m  # XGBoost — TreeExplainer works directly
                feats = load_dropout_features()
                data = load_dropout_preprocessed()
                available = [f for f in feats if f in data.columns]
                X_bee = data[available].dropna().head(300)
            else:
                pipeline = load_model("performance")
                # TreeExplainer needs the underlying RF, not the sklearn Pipeline
                try:
                    shap_model = pipeline.named_steps.get("model") or pipeline.named_steps.get("classifier")
                    # Transform X through the preprocessor first
                    feats = get_feature_names("performance")
                    data = load_performance_data()
                    available = [f for f in feats if f in data.columns]
                    X_raw = data[available].head(300)
                    preprocessor = pipeline.named_steps["preprocess"]
                    X_transformed = preprocessor.transform(X_raw)
                    X_bee = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
                    available = list(X_bee.columns)
                    m = shap_model
                except Exception as e:
                    st.error(f"Could not extract pipeline steps: {e}")
                    st.stop()

            if m and not X_bee.empty:
                with st.spinner("Computing SHAP values..."):
                    try:
                        exp = ModelExplainer(m, X_bee, available)
                        fig_bee = exp.plot_beeswarm(X_bee)
                        if fig_bee:
                            st.plotly_chart(fig_bee, use_container_width=True)
                        else:
                            st.info("Beeswarm not available.")
                    except Exception as e:
                        st.error(f"SHAP error: {e}")
            else:
                st.info("Model or data not available.")
