"""Overview page — system-wide KPIs, distributions, and top-risk students."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_performance_data, load_dropout_data, deduplicate_dropout

st.title("Student Risk System Dashboard")
st.caption("Two complementary early-warning models monitoring academic failure risk and dropout engagement risk")

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    dataset_sel = st.selectbox("Dataset", ["Both", "Dropout (OULA)", "Academic (UCI)"])
    risk_filter = st.multiselect("Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    perf_df = load_performance_data()
    drop_df_raw = load_dropout_data()
    drop_df = deduplicate_dropout(drop_df_raw)

if perf_df.empty and drop_df.empty:
    st.error("No data available. Run `python src/predict.py --task all` first.")
    st.stop()

show_academic = dataset_sel in ("Both", "Academic (UCI)")
show_dropout = dataset_sel in ("Both", "Dropout (OULA)")

# Apply risk filter
if "risk_level" in perf_df.columns:
    perf_filt = perf_df[perf_df["risk_level"].astype(str).isin(risk_filter)]
else:
    perf_filt = perf_df

if "Dropout_Risk_Level" in drop_df.columns:
    drop_filt = drop_df[drop_df["Dropout_Risk_Level"].astype(str).isin(risk_filter)]
else:
    drop_filt = drop_df

# Module filter (dropout only — OULA has code_module)
if "code_module" in drop_df.columns:
    all_modules = sorted(drop_df["code_module"].dropna().unique().tolist())
    with st.sidebar:
        module_sel = st.multiselect("Course Module", all_modules, default=all_modules)
    if module_sel:
        drop_filt = drop_filt[drop_filt["code_module"].isin(module_sel)]
    else:
        drop_filt = drop_filt.iloc[0:0]

# Dataset selector hides the non-selected side everywhere downstream
if not show_academic:
    perf_filt = perf_filt.iloc[0:0]
if not show_dropout:
    drop_filt = drop_filt.iloc[0:0]

# ─── KPI Row ──────────────────────────────────────────────────────────────────
drop_high  = int((drop_filt["Dropout_Risk_Level"].astype(str) == "High").sum()) if "Dropout_Risk_Level" in drop_filt.columns else 0
drop_med   = int((drop_filt["Dropout_Risk_Level"].astype(str) == "Medium").sum()) if "Dropout_Risk_Level" in drop_filt.columns else 0
perf_high  = int((perf_filt["risk_level"].astype(str) == "High").sum()) if "risk_level" in perf_filt.columns else 0
perf_med   = int((perf_filt["risk_level"].astype(str) == "Medium").sum()) if "risk_level" in perf_filt.columns else 0

if show_academic:
    st.markdown("##### Academic Model — UCI Dataset")
    ak1, ak2, ak3 = st.columns(3)
    ak1.metric("Total Students", f"{len(perf_filt):,}")
    ak2.metric("High Risk", f"{perf_high:,}", delta=f"{perf_high/max(len(perf_filt),1):.0%}", delta_color="inverse")
    ak3.metric("Medium Risk", f"{perf_med:,}")

if show_dropout:
    st.markdown("##### Dropout Model — OULA Dataset")
    dk1, dk2, dk3 = st.columns(3)
    dk1.metric("Total Students", f"{len(drop_filt):,}")
    dk2.metric("High Risk", f"{drop_high:,}", delta=f"{drop_high/max(len(drop_filt),1):.0%}", delta_color="inverse")
    dk3.metric("Medium Risk", f"{drop_med:,}")

st.divider()

# ─── Row 1: Donut + Bar by Module (dropout-only) ──────────────────────────────
if show_dropout:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### Dropout Risk Distribution")
            if "Dropout_Risk_Level" in drop_filt.columns and not drop_filt.empty:
                counts = drop_filt["Dropout_Risk_Level"].astype(str).value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    hole=0.55,
                    color=counts.index,
                    color_discrete_map={"High": "#dc2626", "Medium": "#ca8a04", "Low": "#16a34a"},
                )
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                                  showlegend=True, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dropout data unavailable")

    with col2:
        with st.container(border=True):
            st.markdown("#### Dropout Risk by Course Module")
            if "code_module" in drop_filt.columns and "Dropout_Risk_Level" in drop_filt.columns and not drop_filt.empty:
                mod_risk = (
                    drop_filt.groupby(["code_module", "Dropout_Risk_Level"])
                    .size().reset_index(name="count")
                )
                fig2 = px.bar(
                    mod_risk, x="code_module", y="count", color="Dropout_Risk_Level",
                    barmode="stack",
                    color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
                )
                fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                                   xaxis_title="Module", yaxis_title="Students",
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Module data unavailable")

# ─── Row 1b: Academic Risk Distribution + G1 vs G2 Scatter (academic-only) ────
if show_academic:
    ac1, ac2 = st.columns(2)

    with ac1:
        with st.container(border=True):
            st.markdown("#### Academic Risk Distribution")
            if "risk_level" in perf_filt.columns and not perf_filt.empty:
                a_counts = perf_filt["risk_level"].astype(str).value_counts()
                fig_ap = px.pie(
                    values=a_counts.values,
                    names=a_counts.index,
                    hole=0.55,
                    color=a_counts.index,
                    color_discrete_map={"High": "#dc2626", "Medium": "#ca8a04", "Low": "#16a34a"},
                )
                fig_ap.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                                     showlegend=True, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_ap, use_container_width=True)
            else:
                st.info("Academic data unavailable")

    with ac2:
        with st.container(border=True):
            st.markdown("#### Grade Trajectory (G1 → G2) by Risk Level")
            needed = {"G1", "G2", "risk_level"}
            if needed.issubset(perf_filt.columns) and not perf_filt.empty:
                fig_ag = px.scatter(
                    perf_filt, x="G1", y="G2", color="risk_level",
                    color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
                    opacity=0.7,
                    hover_data=[c for c in ["student_id", "failures", "absences"] if c in perf_filt.columns],
                    labels={"G1": "First Period Grade (G1)", "G2": "Second Period Grade (G2)"},
                )
                fig_ag.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
                fig_ag.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                     legend_title_text="Risk")
                st.plotly_chart(fig_ag, use_container_width=True)
            else:
                st.info("G1/G2 grade data unavailable")

# ─── Row 2: SHAP Importance + Probability Histogram ────────────────────────────
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.markdown("#### Top Risk Factors (Frequency)")
        factors_series: list[str] = []
        for col_name in ["Primary_Risk_Factors"]:
            for df_src in [drop_filt, perf_filt]:
                if col_name in df_src.columns:
                    for val in df_src[col_name].dropna().astype(str):
                        for f in val.split("|"):
                            f = f.strip()
                            if f and f.upper() != "N/A":
                                factors_series.append(f)

        if factors_series:
            factor_counts = pd.Series(factors_series).value_counts().head(10)
            fig3 = px.bar(
                x=factor_counts.values, y=factor_counts.index,
                orientation="h", color=factor_counts.values,
                color_continuous_scale="Reds",
                labels={"x": "Count", "y": "Factor"},
            )
            fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                               showlegend=False, coloraxis_showscale=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No risk factor data available")

with col4:
    with st.container(border=True):
        st.markdown("#### Risk Probability Distribution")
        if "Risk_Probability_Value" in drop_filt.columns and not drop_filt.empty:
            fig4 = px.histogram(
                drop_filt, x="Risk_Probability_Value", nbins=40,
                color_discrete_sequence=["#6366f1"],
                labels={"Risk_Probability_Value": "Dropout Probability"},
            )
            fig4.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                               yaxis_title="Count",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        elif "risk_score" in perf_filt.columns and not perf_filt.empty:
            fig4 = px.histogram(
                perf_filt, x="risk_score", nbins=30,
                color_discrete_sequence=["#6366f1"],
                labels={"risk_score": "Academic Fail Probability"},
            )
            fig4.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Probability data unavailable")

# ─── Top 20 High-Risk Students Tables ─────────────────────────────────────────
def color_risk(val):
    colors = {"High": "background-color:#fee2e2", "Medium": "background-color:#fef9c3", "Low": "background-color:#dcfce7"}
    return colors.get(str(val), "")

if show_dropout:
    st.divider()
    st.markdown("#### Top 20 Highest-Risk Students (Dropout)")
    if not drop_filt.empty and "Risk_Probability_Value" in drop_filt.columns:
        top20 = drop_filt.nlargest(20, "Risk_Probability_Value")
        display_cols = [c for c in ["Student_ID", "code_module", "Dropout_Risk_Level",
                                     "Risk_Probability_Value", "Primary_Risk_Factors",
                                     "Intervention_Status"] if c in top20.columns]
        top20_display = top20[display_cols].copy()
        if "Risk_Probability_Value" in top20_display.columns:
            top20_display["Risk_Probability_Value"] = top20_display["Risk_Probability_Value"].apply(lambda x: f"{x:.1%}")

        if "Dropout_Risk_Level" in top20_display.columns:
            st.dataframe(
                top20_display.style.applymap(color_risk, subset=["Dropout_Risk_Level"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.dataframe(top20_display, hide_index=True, use_container_width=True)
    else:
        st.info("No dropout data available for table")

if show_academic:
    st.divider()
    st.markdown("#### Top 20 Highest-Risk Students (Academic)")
    if not perf_filt.empty and "risk_score" in perf_filt.columns:
        top20_perf = perf_filt.nlargest(20, "risk_score")
        perf_cols = [c for c in ["student_id", "school", "sex", "age", "risk_level",
                                  "risk_score", "G1", "G2", "failures", "absences",
                                  "Primary_Risk_Factors", "predicted_outcome"]
                     if c in top20_perf.columns]
        top20_perf_display = top20_perf[perf_cols].copy()
        if "risk_score" in top20_perf_display.columns:
            top20_perf_display["risk_score"] = top20_perf_display["risk_score"].apply(lambda x: f"{x:.1%}")

        if "risk_level" in top20_perf_display.columns:
            st.dataframe(
                top20_perf_display.style.applymap(color_risk, subset=["risk_level"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.dataframe(top20_perf_display, hide_index=True, use_container_width=True)
    else:
        st.info("No academic data available for table")
