"""SHAP-based model explainability utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ─── Feature display labels ───────────────────────────────────────────────────
FEATURE_LABELS = {
    "G1": "First period grade",
    "G2": "Mid-term grade",
    "failures": "Number of past course failures",
    "absences": "Number of school absences",
    "studytime": "Weekly study time",
    "goout": "Time spent going out",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "health": "Health status",
    "freetime": "Free time after school",
    "age": "Student age",
    "Dalc": "Weekday alcohol use",
    "Walc": "Weekend alcohol use",
    "total_clicks": "Online learning activity",
    "active_days": "Days active on learning platform",
    "relative_engagement": "Engagement relative to classmates",
    "avg_score": "Average assessment score",
    "avg_lateness": "Average days late on submissions",
    "num_of_prev_attempts": "Number of previous course attempts",
    "studied_credits": "Total credits enrolled",
}

# ─── Parent-facing risk text templates ───────────────────────────────────────
FEATURE_RISK_TEXTS = {
    "G1": (
        "Your child's first period grade is {val}/20, which is {pct}% below the class average of {avg}/20. "
        "Early grades are a strong indicator of final results."
    ),
    "G2": (
        "Your child's mid-term grade is {val}/20, which is {pct}% below the class average of {avg}/20. "
        "Immediate support is recommended before final assessments."
    ),
    "failures": (
        "Your child has experienced {val} past course failure(s). "
        "This is one of the strongest risk factors for continued academic difficulty."
    ),
    "absences": (
        "Your child has missed {val} school days, compared to a class average of {avg} absences. "
        "High absences are closely linked to lower academic performance."
    ),
    "studytime": (
        "Your child's weekly study time is low (level {val} out of 4). "
        "Insufficient study time significantly increases the risk of poor results."
    ),
    "goout": (
        "Your child spends a high amount of time on social outings (level {val} out of 5). "
        "This may be reducing the time available for study."
    ),
    "Dalc": (
        "Your child's weekday alcohol use is high (level {val} out of 5). "
        "This can affect concentration, attendance, and academic progress."
    ),
    "Walc": (
        "Your child's weekend alcohol use is high (level {val} out of 5). "
        "This may be affecting readiness for school and overall study habits."
    ),
    "total_clicks": (
        "Your child has made only {val} interactions on the learning platform "
        "(class average: {avg}). Low online engagement is a key dropout risk indicator."
    ),
    "active_days": (
        "Your child was active on only {val} out of the first 40 days of the course. "
        "Irregular platform use is strongly linked to dropout risk."
    ),
    "avg_score": (
        "Your child's average assessment score is {val}%, compared to the class average of {avg}%. "
        "Assessment performance needs urgent attention."
    ),
    "avg_lateness": (
        "Your child submits assignments an average of {val} days late. "
        "Consistent late submissions indicate difficulty keeping up with course demands."
    ),
    "num_of_prev_attempts": (
        "Your child has attempted this course {val} time(s) before without completing it. "
        "Additional targeted support is strongly recommended."
    ),
    "studied_credits": (
        "Your child is enrolled in {val} credits, which may represent a heavy workload "
        "and could be contributing to the risk of falling behind."
    ),
}

# ─── Parent-facing protective text templates ──────────────────────────────────
FEATURE_SAFE_TEXTS = {
    "G1": "Your child's first period grade is {val}/20 — a solid foundation for the rest of the course.",
    "G2": "Your child's mid-term grade is {val}/20, which is on track.",
    "failures": "Your child has no past course failures, which is a positive sign.",
    "absences": "Your child's attendance is good, with only {val} absences.",
    "studytime": "Your child is dedicating adequate time to studying (level {val} out of 4).",
    "Dalc": "Your child's weekday alcohol use is currently at a lower level, which is a positive sign.",
    "Walc": "Your child's weekend alcohol use is currently at a lower level, which is a positive sign.",
    "total_clicks": "Your child is actively using the learning platform ({val} interactions).",
    "active_days": "Your child has been logging in regularly ({val} active days out of 40).",
    "avg_score": "Your child's average assessment score of {val}% is above the class average of {avg}%.",
    "avg_lateness": "Your child is submitting assignments on time, which is a positive habit.",
    "num_of_prev_attempts": "This is your child's first attempt at this course.",
}

# ─── Parent-facing intervention recommendations ───────────────────────────────
PARENT_INTERVENTIONS = {
    "G1": (
        "Please contact your child's teacher to discuss their first period results and explore "
        "options such as extra tutoring or additional practice materials."
    ),
    "G2": (
        "Your child's mid-term grade requires urgent attention. We strongly recommend arranging "
        "tutoring or academic support sessions before the final assessment period."
    ),
    "failures": (
        "Given your child's history of course difficulties, we recommend scheduling a meeting "
        "with the academic advisor to create a personalised improvement plan."
    ),
    "absences": (
        "Regular attendance is essential for academic success. Please ensure your child attends "
        "school consistently. If there are barriers to attendance, please contact us so we can assist."
    ),
    "studytime": (
        "Encourage your child to set aside dedicated, distraction-free study time each day. "
        "Even 1-2 focused hours daily can make a significant difference to outcomes."
    ),
    "goout": (
        "Help your child find a healthy balance between social activities and schoolwork. "
        "Consider creating a weekly schedule that sets clear study hours."
    ),
    "Dalc": (
        "Please speak with your child about healthy routines during the school week, "
        "as weekday alcohol use can interfere with learning and concentration."
    ),
    "Walc": (
        "Please discuss weekend routines with your child and help them maintain habits "
        "that support school attendance, focus, and study time."
    ),
    "total_clicks": (
        "Please encourage your child to log into the online learning platform every day and "
        "complete the assigned activities. Consistent online engagement is strongly linked to success."
    ),
    "active_days": (
        "Your child is not regularly accessing the online learning materials. Please check in with "
        "them daily and encourage them to log on, even for short periods."
    ),
    "avg_score": (
        "Your child is struggling with graded assessments. Please contact us to discuss tutoring "
        "options or additional support resources available through the institution."
    ),
    "avg_lateness": (
        "Help your child manage their time and meet submission deadlines. If there are specific "
        "reasons for late submissions (e.g. health, work, family), please inform us so we can offer flexibility."
    ),
    "num_of_prev_attempts": (
        "As this is not your child's first attempt at this course, we recommend a meeting with "
        "the academic advisor to develop a targeted, structured support plan."
    ),
    "studied_credits": (
        "Your child may be taking on too heavy a course load. Please speak with an academic advisor "
        "about whether adjusting the number of enrolled credits would be beneficial."
    ),
    "relative_engagement": (
        "Your child's overall engagement with the course is below that of their peers. "
        "Please discuss the course with your child and identify any barriers to participation."
    ),
}


class ModelExplainer:
    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        feature_names: list[str],
        class_index: Optional[int] = 1,
        allowed_features: Optional[set[str]] = None,
    ):
        self.feature_names = feature_names
        self.class_index = class_index
        self.allowed_features = allowed_features
        self.explainer = None
        self.shap_values = None

        if not SHAP_AVAILABLE:
            return

        try:
            self.explainer = shap.TreeExplainer(model)
            X_sample = X_train[feature_names].head(200) if len(X_train) > 200 else X_train[feature_names]
            # Ensure dense float array — sparse input causes downstream scalar errors
            if hasattr(X_sample, "toarray"):
                X_sample = pd.DataFrame(X_sample.toarray(), columns=feature_names)
            X_sample = X_sample.astype(float).reset_index(drop=True)
            raw = self.explainer.shap_values(X_sample)
            self.shap_values = self._select_class_shap(raw)
            self.X_sample = X_sample
        except Exception:
            self.explainer = None

    def _select_class_shap(self, raw) -> np.ndarray:
        """Select the SHAP values for the class aligned with the displayed risk."""
        if isinstance(raw, list):
            if not raw:
                return np.array([])
            idx = self.class_index if self.class_index is not None else len(raw) - 1
            idx = max(0, min(idx, len(raw) - 1))
            return np.array(raw[idx])
        arr = np.array(raw)
        if arr.ndim >= 3 and self.class_index is not None:
            idx = max(0, min(self.class_index, arr.shape[-1] - 1))
            return np.take(arr, idx, axis=-1)
        return arr

    def _get_single_shap(self, student_row: pd.Series) -> Optional[np.ndarray]:
        if self.explainer is None:
            return None
        try:
            row_df = pd.DataFrame([student_row[self.feature_names].values],
                                  columns=self.feature_names)
            raw = self.explainer.shap_values(row_df)
            arr = self._select_class_shap(raw)
            # Flatten to 1-D: (1, n_features) → (n_features,)
            return arr.reshape(-1) if arr.ndim > 1 else arr
        except Exception:
            return None

    def get_student_explanation(self, student_row: pd.Series, population_stats: dict) -> dict:
        """
        Compute SHAP explanation for one student.
        Returns parent-friendly text in all fields.
        """
        shap_vals = self._get_single_shap(student_row)
        if shap_vals is None:
            return {
                "risk_factors": [],
                "protective_factors": [],
                "summary": (
                    "A detailed risk breakdown is not available at this time. "
                    "Please contact the academic advisor for a personal assessment."
                ),
                "interventions": [
                    "Please contact the school to arrange a meeting with the academic advisor.",
                    "Review recent grades and attendance records with your child.",
                ],
                "parent_message": "",
            }

        risk_factors = []
        protective_factors = []
        risk_threshold = 0.02
        protective_threshold = -0.01

        for i, feat in enumerate(self.feature_names):
            if i >= len(shap_vals):
                continue
            if self.allowed_features is not None and feat not in self.allowed_features:
                continue
            # Safely extract scalar from shap value (may be 0-d or 1-d array)
            try:
                sv = float(np.squeeze(shap_vals[i]))
            except Exception:
                continue
            val = student_row.get(feat, None)
            stats = population_stats.get(feat, {})
            mean = stats.get("mean", 0)

            # Format value for display — guard against array-valued cells
            try:
                scalar_val = float(np.squeeze(np.asarray(val))) if val is not None else float("nan")
                numeric_val = scalar_val
                display_val = round(numeric_val, 1)
                if mean != 0:
                    pct_diff = int(abs((numeric_val - mean) / mean * 100))
                else:
                    pct_diff = 0
            except (TypeError, ValueError):
                display_val = str(val) if val is not None else "N/A"
                pct_diff = 0

            display_avg = round(mean, 1)
            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

            # Generate parent-friendly explanation text
            if sv >= risk_threshold:
                tmpl = FEATURE_RISK_TEXTS.get(feat)
                if tmpl:
                    explanation = tmpl.format(val=display_val, avg=display_avg, pct=pct_diff)
                else:
                    explanation = (
                        f"{label} is {display_val}, which is contributing to the risk assessment "
                        f"(class average: {display_avg})."
                    )
            else:
                tmpl = FEATURE_SAFE_TEXTS.get(feat)
                if tmpl:
                    explanation = tmpl.format(val=display_val, avg=display_avg, pct=pct_diff)
                else:
                    explanation = (
                        f"{label} is {display_val}, which is a positive signal for this student."
                    )

            entry = {
                "feature": feat,
                "label": label,
                "value": display_val,
                "avg": display_avg,
                "shap_value": sv,
                "explanation": explanation,
            }

            if sv >= risk_threshold:
                risk_factors.append(entry)
            elif sv <= protective_threshold:
                protective_factors.append(entry)

        risk_factors.sort(key=lambda x: x["shap_value"], reverse=True)
        protective_factors.sort(key=lambda x: x["shap_value"])

        # Build parent-targeted interventions from top risk factors
        interventions = []
        for rf in risk_factors[:4]:
            action = PARENT_INTERVENTIONS.get(rf["feature"])
            if action and action not in interventions:
                interventions.append(action)

        # Plain-language summary
        if risk_factors:
            top = risk_factors[0]
            n_risks = len(risk_factors)
            summary = (
                f"The system has identified {n_risks} concern(s) for this student. "
                f"The most significant is: {top['explanation']}"
            )
        else:
            summary = (
                "No major individual risk signals were detected. "
                "The student's profile appears stable, though general monitoring is advised."
            )

        # Full parent message (formatted for sending)
        parent_message = _build_parent_message(risk_factors, protective_factors, interventions, summary)

        return {
            "risk_factors": risk_factors[:5],
            "protective_factors": protective_factors[:3],
            "summary": summary,
            "interventions": interventions,
            "parent_message": parent_message,
        }

    def plot_waterfall(self, student_row: pd.Series) -> Optional[object]:
        """Plotly bar chart of top SHAP contributions for a single student."""
        if not PLOTLY_AVAILABLE or self.explainer is None:
            return None
        shap_vals = self._get_single_shap(student_row)
        if shap_vals is None:
            return None

        pairs = sorted(
            zip(self.feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        names, vals = zip(*pairs)
        colors = ["#d9534f" if v > 0 else "#5cb85c" for v in vals]
        labels = [FEATURE_LABELS.get(n, n.replace("_", " ").title()) for n in names]

        fig = go.Figure(go.Bar(
            x=list(vals),
            y=labels,
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(
            title="Feature Impact on Risk Score",
            xaxis_title="Impact (positive = increases risk)",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def plot_beeswarm(self, X: pd.DataFrame) -> Optional[object]:
        """Plotly dot-plot summary (beeswarm-style) of global SHAP values."""
        if not PLOTLY_AVAILABLE or self.shap_values is None:
            return None

        n = min(len(X), len(self.shap_values))
        shap_arr = np.array(self.shap_values[:n])
        feat_arr = X[self.feature_names].values[:n]

        mean_abs = np.abs(shap_arr).mean(axis=0)
        order = np.argsort(mean_abs)[-10:]

        fig = go.Figure()
        for idx in order:
            feat = self.feature_names[idx]
            sv = shap_arr[:, idx]
            fv = feat_arr[:, idx]
            try:
                fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            except Exception:
                fv_norm = np.zeros_like(fv)

            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
            fig.add_trace(go.Scatter(
                x=sv,
                y=[label] * len(sv),
                mode="markers",
                marker=dict(
                    color=fv_norm,
                    colorscale="RdBu_r",
                    size=4,
                    opacity=0.6,
                ),
                name=feat,
                showlegend=False,
            ))

        fig.update_layout(
            title="SHAP Beeswarm — Global Feature Impact",
            xaxis_title="SHAP Value",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def get_global_importance(self) -> pd.DataFrame:
        """Return mean absolute SHAP values per feature."""
        if self.shap_values is None:
            return pd.DataFrame({"feature": self.feature_names, "importance": [0.0] * len(self.feature_names)})
        mean_abs = np.abs(np.array(self.shap_values)).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "importance": mean_abs})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ─── Parent message builder ───────────────────────────────────────────────────

def _build_parent_message(risk_factors, protective_factors, interventions, summary) -> str:
    """Build a formatted parent notification message."""
    lines = []
    lines.append("Dear Parent / Guardian,")
    lines.append("")
    lines.append(
        "We are reaching out because our student monitoring system has identified "
        "that your child may need additional academic support. Please find the details below."
    )
    lines.append("")

    if risk_factors:
        lines.append("AREAS OF CONCERN")
        lines.append("-" * 40)
        for i, rf in enumerate(risk_factors[:4], 1):
            lines.append(f"{i}. {rf['explanation']}")
        lines.append("")

    if protective_factors:
        lines.append("POSITIVE INDICATORS")
        lines.append("-" * 40)
        for pf in protective_factors[:2]:
            lines.append(f"- {pf['explanation']}")
        lines.append("")

    if interventions:
        lines.append("RECOMMENDED ACTIONS FOR PARENTS")
        lines.append("-" * 40)
        for i, action in enumerate(interventions, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

    lines.append(
        "We encourage you to discuss these points with your child and to reach out to "
        "the school if you have any questions or concerns. Early action makes a significant difference."
    )
    lines.append("")
    lines.append("Sincerely,")
    lines.append("The Student Success Team")

    return "\n".join(lines)
