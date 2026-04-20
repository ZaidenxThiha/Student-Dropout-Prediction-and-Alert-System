"""Dropout Alerts page — OULA engagement-based early warning."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_dropout_data, load_actionable_report, deduplicate_dropout, load_config

st.title("Dropout Engagement Risk Alerts")
st.caption("Early warning flags from the OULA Virtual Learning Environment — first 40 days of student engagement (independent dataset from Academic model)")

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading dropout data..."):
    drop_raw = load_dropout_data()
    actionable = load_actionable_report()
    config = load_config()

if drop_raw.empty:
    st.error("Dropout data not found. Run `python src/predict.py --task dropout`.")
    st.stop()

drop_df = deduplicate_dropout(drop_raw)
threshold = config["dropout"].get("threshold", 0.36)
metrics = config["dropout"].get("metrics", {})

# ─── Alert Banner ─────────────────────────────────────────────────────────────
high_risk_count = int((drop_df["Dropout_Risk_Level"].astype(str) == "High").sum())
st.error(f"**{high_risk_count} students** flagged as HIGH RISK from first 40 days of engagement data")

# ─── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Monitored", f"{len(drop_df):,}")
k2.metric("High Risk", f"{high_risk_count:,}", delta=f"{high_risk_count/max(len(drop_df),1):.0%}", delta_color="inverse")
k3.metric("Model Recall", f"{metrics.get('recall', 0.827):.1%}")
k4.metric("Model Precision", f"{metrics.get('precision', 0.727):.1%}")

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    risk_filter = st.multiselect("Risk Level", ["High", "Medium", "Low"], default=["High", "Medium"])

    if "code_module" in drop_df.columns:
        all_modules = sorted(drop_df["code_module"].dropna().unique().tolist())
        module_filter = st.multiselect("Course Module", all_modules, default=all_modules)
    else:
        module_filter = []

    student_q = st.text_input("Search Student ID")

# Apply filters
filtered = drop_df.copy()
if "Dropout_Risk_Level" in filtered.columns:
    filtered = filtered[filtered["Dropout_Risk_Level"].astype(str).isin(risk_filter)]
if module_filter and "code_module" in filtered.columns:
    filtered = filtered[filtered["code_module"].isin(module_filter)]
if student_q.strip():
    filtered = filtered[filtered["Student_ID"].astype(str).str.contains(student_q.strip(), case=False, na=False)]

filtered = filtered.sort_values("Risk_Probability_Value", ascending=False)

# ─── Charts ───────────────────────────────────────────────────────────────────
st.divider()
c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown("#### Risk Distribution by Module")
        if "code_module" in filtered.columns and "Dropout_Risk_Level" in filtered.columns:
            mod_counts = (
                filtered.groupby(["code_module", "Dropout_Risk_Level"])
                .size().reset_index(name="count")
            )
            fig1 = px.bar(
                mod_counts, x="code_module", y="count", color="Dropout_Risk_Level",
                barmode="stack",
                color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
            )
            fig1.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                               xaxis_title="Module", yaxis_title="Students",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Module data unavailable")

with c2:
    with st.container(border=True):
        st.markdown("#### Engagement — Clicks vs Active Days")
        if "total_clicks" in filtered.columns and "active_days" in filtered.columns:
            fig2 = px.scatter(
                filtered, x="total_clicks", y="active_days",
                color="Dropout_Risk_Level" if "Dropout_Risk_Level" in filtered.columns else None,
                color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"},
                opacity=0.5,
                labels={"total_clicks": "Total VLE Clicks", "active_days": "Active Days"},
            )
            fig2.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Engagement data unavailable")

RISK_COLORS = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}
RISK_ORDER = ["Low", "Medium", "High"]
medium_thresh = config["dropout"]["risk_levels"].get("medium", 0.36)
high_thresh = config["dropout"]["risk_levels"].get("high", 0.51)

c3, c4 = st.columns(2)

with c3:
    with st.container(border=True):
        st.markdown("#### Dropout Probability Distribution")
        if "Risk_Probability_Value" in filtered.columns and not filtered.empty:
            fig3 = px.histogram(
                filtered, x="Risk_Probability_Value", nbins=30,
                color="Dropout_Risk_Level" if "Dropout_Risk_Level" in filtered.columns else None,
                color_discrete_map=RISK_COLORS,
                category_orders={"Dropout_Risk_Level": RISK_ORDER},
                labels={"Risk_Probability_Value": "Dropout Probability"},
            )
            fig3.add_vline(x=medium_thresh, line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"Medium ≥ {medium_thresh:.2f}", annotation_position="top")
            fig3.add_vline(x=high_thresh, line_dash="dash", line_color="#dc2626",
                           annotation_text=f"High ≥ {high_thresh:.2f}", annotation_position="top")
            fig3.update_layout(height=340, margin=dict(t=64, b=12, l=10, r=10),
                               bargap=0.05, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Risk")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Risk probability unavailable")

with c4:
    with st.container(border=True):
        st.markdown("#### Mean Dropout Risk — Avg Score × VLE Clicks")
        needed = {"avg_score", "total_clicks", "Risk_Probability_Value"}
        if needed.issubset(filtered.columns) and not filtered.empty:
            hm = filtered.copy()
            hm["score_bin"] = pd.cut(
                pd.to_numeric(hm["avg_score"], errors="coerce"),
                bins=[0, 40, 55, 70, 85, 100],
                labels=["0–40", "40–55", "55–70", "70–85", "85–100"],
            )
            hm["clicks_bin"] = pd.qcut(
                pd.to_numeric(hm["total_clicks"], errors="coerce").rank(method="first"),
                q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"],
            )
            pivot = (
                hm.groupby(["score_bin", "clicks_bin"], observed=False)["Risk_Probability_Value"]
                .mean().reset_index()
                .pivot(index="score_bin", columns="clicks_bin", values="Risk_Probability_Value")
            )
            fig4 = px.imshow(
                pivot, text_auto=".0%", aspect="auto",
                color_continuous_scale="Reds", zmin=0, zmax=1,
                labels=dict(x="VLE Clicks Quintile", y="Avg Score Band", color="Mean Risk"),
            )
            fig4.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("avg_score / total_clicks / risk unavailable")

c5, c6 = st.columns(2)

with c5:
    with st.container(border=True):
        st.markdown("#### Active Days by Risk Level")
        if "active_days" in filtered.columns and "Dropout_Risk_Level" in filtered.columns and not filtered.empty:
            fig5 = px.box(
                filtered, x="Dropout_Risk_Level", y="active_days",
                color="Dropout_Risk_Level",
                category_orders={"Dropout_Risk_Level": RISK_ORDER},
                color_discrete_map=RISK_COLORS, points="outliers",
                labels={"Dropout_Risk_Level": "Risk Level", "active_days": "Active Days (first 40)"},
            )
            fig5.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                               showlegend=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("active_days unavailable")

with c6:
    with st.container(border=True):
        st.markdown("#### Risk Composition by Module (Sunburst)")
        if {"code_module", "Dropout_Risk_Level"}.issubset(filtered.columns) and not filtered.empty:
            sun_df = (
                filtered.groupby(["code_module", "Dropout_Risk_Level"])
                .size().reset_index(name="count")
            )
            fig6 = px.sunburst(
                sun_df, path=["code_module", "Dropout_Risk_Level"], values="count",
                color="Dropout_Risk_Level", color_discrete_map=RISK_COLORS,
            )
            fig6.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Module breakdown unavailable")

with st.expander(f"Alert Table — {len(filtered):,} students", expanded=False):
    table_cols = [c for c in [
        "Student_ID", "code_module", "Risk_Probability_Value", "Dropout_Risk_Level",
        "total_clicks", "active_days", "avg_score", "Primary_Risk_Factors", "Intervention_Status"
    ] if c in filtered.columns]

    display = filtered[table_cols].copy()
    if "Risk_Probability_Value" in display.columns:
        display["Risk_Probability_Value"] = display["Risk_Probability_Value"].apply(lambda x: f"{x:.1%}")
    if "total_clicks" in display.columns:
        display["total_clicks"] = display["total_clicks"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    if "avg_score" in display.columns:
        display["avg_score"] = display["avg_score"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    display.columns = [c.replace("_", " ").title() for c in display.columns]

    def color_risk_row(val):
        c = {"High": "background-color:#fee2e2", "Medium": "background-color:#fef9c3", "Low": "background-color:#dcfce7"}
        return c.get(str(val), "")

    risk_col_display = "Dropout Risk Level"
    if risk_col_display in display.columns:
        st.dataframe(
            display.style.applymap(color_risk_row, subset=[risk_col_display]),
            hide_index=True, use_container_width=True, height=340,
        )
    else:
        st.dataframe(display, hide_index=True, use_container_width=True, height=340)

    st.download_button(
        "Download Intervention Report",
        data=filtered.to_csv(index=False).encode(),
        file_name="dropout_alerts.csv",
        mime="text/csv",
    )

if not actionable.empty:
    with st.container(border=True):
        st.markdown("#### Weekly Action List")
        st.caption("High-risk students requiring immediate action")
        st.dataframe(actionable.head(20), hide_index=True, use_container_width=True, height=260)
