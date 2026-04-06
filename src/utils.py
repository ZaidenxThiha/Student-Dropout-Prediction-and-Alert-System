"""Shared UI helpers used across Streamlit pages."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

RISK_COLORS = {
    "High": "#ef4444",
    "Medium": "#f59e0b",
    "Low": "#10b981",
}


def risk_color(level: str) -> str:
    """Return hex color string for a risk level."""
    return RISK_COLORS.get(level, "#6b7280")


def create_gauge(value: float, title: str = "Risk Score") -> go.Figure:
    """
    Plotly gauge chart for risk probability display.
    value: float in [0, 1]
    """
    if value >= 0.65:
        bar_color = "#ef4444"
    elif value >= 0.4:
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 40], "color": "#d1fae5"},
                {"range": [40, 65], "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=30, r=30),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_donut(labels: list, values: list, colors: list | None = None) -> go.Figure:
    """Plotly donut chart for risk distribution."""
    if colors is None:
        colors = [RISK_COLORS.get(l, "#6b7280") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside",
    ))
    fig.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def style_risk_dataframe(df: pd.DataFrame) -> object:
    """
    Apply background color styling to the risk_level column for st.dataframe.
    Returns a pandas Styler object.
    """
    def _color_cell(val: str) -> str:
        color = risk_color(str(val))
        return f"background-color: {color}20; color: {color}; font-weight: bold"

    if "risk_level" in df.columns:
        return df.style.applymap(_color_cell, subset=["risk_level"])
    if "Dropout_Risk_Level" in df.columns:
        return df.style.applymap(_color_cell, subset=["Dropout_Risk_Level"])
    return df.style
