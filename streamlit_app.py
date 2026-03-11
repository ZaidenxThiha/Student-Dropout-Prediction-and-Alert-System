import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pred_path = base_dir / "data" / "processed" / "student_predictions.csv"

    st.title("Student Risk Dashboard")

    if not pred_path.exists():
        st.error(
            f"Predictions file not found at {pred_path}. "
            "Run the training/evaluation pipeline to generate predictions first."
        )
        return

    df = load_predictions(pred_path)

    # Summary metrics
    st.metric("Total Students", len(df))
    high_risk = df[df["risk_level"] == "High"]
    st.metric("High Risk Students (Fail risk)", len(high_risk))

    # Risk distribution
    st.subheader("Risk Level Distribution")
    st.bar_chart(df["risk_level"].value_counts())

    # Student search
    st.subheader("Search by Student ID")
    student_id = st.text_input("Enter Student ID", "")
    if student_id.strip():
        try:
            sid = int(student_id)
            student = df[df["student_id"] == sid]
            if student.empty:
                st.warning("No student found with that ID.")
            else:
                st.dataframe(student)
        except ValueError:
            st.warning("Please enter a numeric Student ID.")

    # Risk-level filter
    st.subheader("Students at Risk (filter by level)")
    selected_level = st.selectbox("Choose risk level", ["High", "Medium", "Low"])
    filtered = df[df["risk_level"] == selected_level].copy()
    filtered["predicted_outcome"] = filtered["prediction"].map({0: "Fail", 1: "Pass"})
    if filtered.empty:
        st.info(f"No {selected_level} risk students found.")
    else:
        st.dataframe(filtered)

    # Full table
    st.subheader("All Predictions")
    display_df = df.copy()
    display_df["predicted_outcome"] = display_df["prediction"].map({0: "Fail", 1: "Pass"})
    display_df = display_df.rename(columns={"risk_score": "fail_risk_score"})
    st.dataframe(display_df)


if __name__ == "__main__":
    main()
