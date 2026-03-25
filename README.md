# Student Risk Monitoring System

This project contains two independent student risk workflows and a single Streamlit dashboard that presents their outputs together:

- Academic failure model using the UCI Student Performance dataset
- Dropout risk model using the OULA dataset

These are not a single combined prediction pipeline. The datasets, model inputs, and student records are separate. The dashboard provides a unified monitoring view only.

## What the Project Does

The system supports three dashboard views:

- `Academic Risk`: pass/fail risk scores, risk levels, and academic risk factors
- `Dropout Risk`: dropout probabilities, intervention status, and engagement risk factors
- `Parent Alerts`: one combined message table created by concatenating academic alerts and dropout alerts without merging student datasets

## Repository Layout

```text
Student Dropout Prediction and Alert System/
├── data/
│   ├── raw/
│   │   ├── uci/
│   │   │   ├── student-mat.csv
│   │   │   └── student-por.csv
│   │   └── oula/
│   │       ├── assessments.csv
│   │       ├── courses.csv
│   │       ├── studentAssessment.csv
│   │       ├── studentInfo.csv
│   │       ├── studentRegistration.csv
│   │       ├── studentVle.csv
│   │       └── vle.csv
│   └── processed/
│       ├── performance/
│       │   ├── student_all_cleaned.csv
│       │   ├── student_mat_cleaned.csv
│       │   ├── student_por_cleaned.csv
│       │   ├── student_predictions.csv
│       │   └── student_predictions_holdout.csv
│       └── dropout/
│           ├── dropout_preprocessed.csv
│           ├── engineered_features.csv
│           ├── X_test.csv
│           ├── y_test.csv
│           ├── student_ids.csv
│           ├── Student_risk_report.csv
│           └── actionable_weekly_risk_report.csv
├── models/
│   ├── performance/
│   │   └── pass_classifier_rf.joblib
│   └── dropout/
│       ├── oula_ews_model.pkl
│       └── model_features.pkl
├── notebooks/
│   ├── performance/
│   │   ├── preprocess.ipynb
│   │   ├── train.ipynb
│   │   ├── evaluate.ipynb
│   │   └── report.ipynb
│   └── dropout/
│       ├── preprocess.ipynb
│       ├── train.ipynb
│       ├── evaluate.ipynb
│       └── report.ipynb
├── src/
│   └── predict.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Models

### 1. Academic Failure Model

Dataset:

- `data/raw/uci/student-mat.csv`
- `data/raw/uci/student-por.csv`

Primary artifact:

- `models/performance/pass_classifier_rf.joblib`

Generated output used by the dashboard:

- `data/processed/performance/student_predictions.csv`

Important output fields:

- `student_id`
- `risk_score`
- `risk_level`
- `Primary_Risk_Factors`

### 2. Dropout Risk Model

Dataset:

- `data/raw/oula/studentInfo.csv`
- `data/raw/oula/studentVle.csv`
- `data/raw/oula/studentAssessment.csv`
- `data/raw/oula/assessments.csv`

Primary artifacts:

- `models/dropout/oula_ews_model.pkl`
- `models/dropout/model_features.pkl`

Generated outputs used by the dashboard:

- `data/processed/dropout/Student_risk_report.csv`
- `data/processed/dropout/actionable_weekly_risk_report.csv`

Important output fields:

- `Student_ID`
- `Risk_Probability_Value`
- `Risk_Probability`
- `Primary_Risk_Factors`
- `Intervention_Status`

## Dashboard

Run the dashboard with:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The app uses:

- `page_title = "Student Risk Monitoring System"`
- wide layout
- three tabs:
  - `Academic Risk`
  - `Dropout Risk`
  - `Parent Alerts`

### Parent Alerts Design

The `Parent Alerts` tab is a unified message view, not a merged student dataset.

It works by:

1. Loading the academic prediction output
2. Loading the dropout prediction output
3. Converting both outputs into the same alert structure
4. Concatenating them into one dataframe

Shared alert columns:

- `Student_ID`
- `Alert_Type`
- `Risk_Value`
- `Risk_Label`
- `Risk_Factors`
- `Parent_Message`

Important:

- The academic and dropout datasets are not merged by student
- A row in the parent alerts table comes from one model only
- The same `Student_ID` value across sources should not be treated as the same person unless verified outside this app

## Setup

Create and activate a virtual environment:

```bash
brew install python
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Project

If you are starting from a clean machine on macOS:

```bash
brew install python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/predict.py --task all
streamlit run streamlit_app.py
```

This will:

- install Python
- create a virtual environment
- install project dependencies
- generate the latest performance and dropout report CSV files
- launch the Streamlit dashboard

Core dependencies are listed in `requirements.txt`, including:

- `pandas`
- `scikit-learn`
- `xgboost`
- `shap`
- `streamlit`
- `joblib`

## Notebook Workflow

Launch notebooks with:

```bash
source .venv/bin/activate
jupyter notebook
```

### Performance notebooks

- `notebooks/performance/preprocess.ipynb`
- `notebooks/performance/train.ipynb`
- `notebooks/performance/evaluate.ipynb`
- `notebooks/performance/report.ipynb`

### Dropout notebooks

- `notebooks/dropout/preprocess.ipynb`
- `notebooks/dropout/train.ipynb`
- `notebooks/dropout/evaluate.ipynb`
- `notebooks/dropout/report.ipynb`

The dropout notebooks now use repo-safe paths based on `Path.cwd()` and resolve correctly when run from `notebooks/dropout`.

## Prediction Export Script

The main export utility is `src/predict.py`.

Run all exports:

```bash
source .venv/bin/activate
python src/predict.py --task all
```

Run only performance export:

```bash
python src/predict.py --task performance
```

Run only dropout export:

```bash
python src/predict.py --task dropout
```

Generated files:

- Performance:
  - `data/processed/performance/student_predictions.csv`
- Dropout:
  - `data/processed/dropout/Student_risk_report.csv`
  - `data/processed/dropout/actionable_weekly_risk_report.csv`

## Risk Logic Summary

### Academic risk

- Prediction target is pass/fail
- `risk_score = 1 - P(pass)`
- Risk buckets:
  - `Low` if score `< 0.3`
  - `Medium` if score `< 0.6`
  - `High` otherwise

### Dropout risk

- Dropout probability is generated from the saved XGBoost model
- Default classification threshold in `src/predict.py` is `0.5`
- Dashboard risk bands for display:
  - `Low` if probability `<= 0.3`
  - `Medium` if probability `<= 0.6`
  - `High` if probability `> 0.6`

## Expected Files for the App

At minimum, the Streamlit app expects:

- `data/processed/performance/student_predictions.csv`
- `data/processed/dropout/Student_risk_report.csv`

Optional but supported:

- `data/processed/dropout/actionable_weekly_risk_report.csv`

If the actionable dropout report is missing, the dashboard still runs and shows an empty-state message for that section.

## Known Notes

- Notebook output cells may still contain old Windows-style paths until those notebooks are rerun and saved.
- The files under `notebooks/dropout/..\\models\\*` are accidental artifacts created by earlier notebook path issues and are not part of the intended project structure.
- The dashboard reads exported CSV outputs. It does not train models directly.

## Team Scope

This repository covers:

- academic risk prediction
- dropout risk prediction
- notebook-based preprocessing and evaluation
- export of CSV reports for dashboard use
- Streamlit-based monitoring and parent-facing alert presentation
