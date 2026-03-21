# Student Outcomes Prediction and Prevent Dropout

This project combines two student-risk workflows in one repo:
- `performance`: predicts pass/fail risk from the UCI student performance dataset
- `dropout`: predicts dropout risk and intervention priority from the OULA dataset

Both outputs are surfaced in a single Streamlit dashboard.

## Project Structure

```text
testITprojc/
├── data/
│   ├── raw/
│   │   ├── uci/                  # UCI source files for performance model
│   │   └── oula/                 # OULA source files for dropout model
│   └── processed/
│       ├── performance/          # cleaned UCI data + performance predictions
│       └── dropout/              # dropout reports, test splits, processed OULA data
├── models/
│   ├── performance/              # pass/fail model artifacts
│   └── dropout/                  # dropout model artifacts
├── notebooks/
│   ├── performance/              # preprocess, train, evaluate, report
│   └── dropout/                  # preprocess, train, evaluate, report
├── src/
│   └── predict.py                # performance prediction utility
├── streamlit_app.py              # unified dashboard
└── requirements.txt
```

## Datasets

### UCI Performance
- Raw files live in `data/raw/uci/`
- Main processed outputs live in `data/processed/performance/`
- Main model artifact: `models/performance/pass_classifier_rf.joblib`

### OULA Dropout
- Raw files live in `data/raw/oula/`
- Main processed outputs live in `data/processed/dropout/`
- Main model artifact: `models/dropout/rf_dropout_model.joblib`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

### Performance notebooks
- `notebooks/performance/preprocess.ipynb`
- `notebooks/performance/train.ipynb`
- `notebooks/performance/evaluate.ipynb` — model evaluation and performance metrics
- `notebooks/performance/report.ipynb` — risk-student report with primary risk factors

### Dropout notebooks
- `notebooks/dropout/preprocess.ipynb`
- `notebooks/dropout/train.ipynb`
- `notebooks/dropout/evaluate.ipynb`
- `notebooks/dropout/report.ipynb`

Launch notebooks with:

```bash

source .venv/bin/activate
jupyter notebook
```

## Prediction Workflow

### Performance model
Generate pass/fail predictions from the processed UCI dataset:

```bash
source .venv/bin/activate
python src/predict.py
```

This writes:
- `data/processed/performance/student_predictions.csv`

Expected inputs:
- `data/processed/performance/student_all_cleaned.csv`
- `models/performance/pass_classifier_rf.joblib`

### Dropout model
The Streamlit dashboard reads existing dropout outputs from:
- `data/processed/dropout/Student_risk_report.csv`
- `data/processed/dropout/actionable_weekly_risk_report.csv`

## Streamlit Dashboard

Run the combined dashboard:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The dashboard has two tabs:
- `Pass/Fail Model`
- `Dropout Prevention Model`

Required files:
- `data/processed/performance/student_predictions.csv`
- `data/processed/dropout/Student_risk_report.csv`
- `data/processed/dropout/actionable_weekly_risk_report.csv`

Team Member Contributions

Theworkinthis project has been divided according to the responsibilities of each team member.

Thiha Aung

Thiha Aung is responsible for the following tasks:
- Development of the student performance prediction model.
- Data preprocessing related to performance prediction.
- Model implementation and testing for academic performance classification.
- Contribution to report writing and project documentation.
- 
Thin Lei Sandi

Thin Lei Sandi is responsible for the following tasks:
- Development of the student dropout prediction model.
- Data preprocessing related to dropout prediction.
- Model implementation and testing for dropout risk classification.
- Contribution to report writing and project documentation.
Shared Responsibilities
- Project planning and discussion.
- System concept and overall design.
- Streamlit dashboard integration.
- Midterm report preparation.
= Review of implementation progress

## Notes

- Performance target: `pass` where `1` means `G3 >= 10`
- Performance risk score = `1 - P(pass)`
- Performance risk buckets:
  - `Low` for score `< 0.3`
  - `Medium` for score `< 0.6`
  - `High` for score `>= 0.6`
- Dropout reports already include probability and intervention labels used by the dashboard
