# Student Dropout Prediction and Alert System

An early warning dashboard for educators that combines two independent student risk models into a single monitoring interface. The system identifies students at risk of academic failure or dropout, generates SHAP-based explanations for each prediction, and produces parent-facing notifications.

## What the System Does

Two separate machine learning models run on two separate datasets:

| Model | Dataset | Prediction |
|-------|---------|------------|
| Academic Failure | UCI Student Performance (Math + Portuguese) | Pass / Fail probability |
| Dropout Risk | OULA Virtual Learning Environment | Dropout probability from first 40 days of engagement |

The dashboard presents both model outputs in a unified interface. The student records are not merged — a student ID from the UCI dataset is not the same person as a student ID from the OULA dataset.

## Repository Layout

```
Student Dropout Prediction and Alert System/
├── app.py                          # Streamlit entry point (landing page)
├── pages/
│   ├── 1_Overview.py               # System-wide KPIs and risk distributions
│   ├── 2_Performance.py            # Academic pass/fail model
│   ├── 3_Dropout_Alerts.py         # Dropout risk alerts and interventions
│   ├── 4_Student_Profile.py        # Per-student SHAP explanation and parent notification
│   ├── 5_Model_Insights.py         # Confusion matrices, ROC curves, threshold tuning
│   └── 6_Analytics.py             # Correlations, course analysis, What-If simulator
├── src/
│   ├── data_loader.py              # Data loading and caching utilities
│   ├── predictor.py                # Model loading and prediction functions
│   ├── explainability.py           # SHAP explanation and parent message generation
│   ├── utils.py                    # Shared UI helpers (gauges, colours, styling)
│   └── predict.py                  # CLI script to regenerate CSV exports
├── config/
│   └── model_config.json           # Thresholds, risk level boundaries, model paths
├── models/
│   ├── performance/
│   │   └── pass_classifier_rf.joblib
│   └── dropout/
│       ├── dropout_xgb_optimized.joblib
│       ├── oula_ews_model.pkl
│       └── model_features.pkl
├── data/
│   ├── raw/                        # Source datasets (not committed)
│   │   ├── uci/
│   │   └── oula/
│   └── processed/
│       ├── performance/
│       │   ├── student_predictions.csv
│       │   └── student_predictions_holdout.csv
│       └── dropout/
│           ├── Student_risk_report.csv
│           ├── actionable_weekly_risk_report.csv
│           └── dropout_preprocessed.csv
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
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Prediction Reports

The dashboard reads from pre-generated CSV files. Run this script to produce them:

```bash
python src/predict.py --task all
```

Individual tasks:

```bash
python src/predict.py --task performance
python src/predict.py --task dropout
```

Generated files:

- `data/processed/performance/student_predictions.csv`
- `data/processed/dropout/Student_risk_report.csv`
- `data/processed/dropout/actionable_weekly_risk_report.csv`

## Run the Dashboard

```bash
streamlit run app.py
```

The app opens on a landing page. Use the sidebar to navigate between pages:

| Page | Description |
|------|-------------|
| Overview | System-wide KPIs, risk distributions, and top-risk student list |
| Performance | Academic pass/fail predictions with filtering and manual prediction form |
| Dropout Alerts | Dropout risk flags, engagement charts, and intervention list |
| Student Profile | Per-student risk gauge, SHAP explanation, and parent notification letter |
| Model Insights | Confusion matrices, ROC/PR curves, and interactive threshold tuning |
| Analytics | Risk factor analysis, course-level stats, correlation heatmap, What-If simulator |

## Models

### Academic Failure Model

- Algorithm: Random Forest (sklearn Pipeline with ColumnTransformer preprocessing)
- Training data: UCI Student Performance dataset (math + portuguese combined)
- Target: pass / fail (binary)
- Risk score: `1 - P(pass)`
- Saved artifact: `models/performance/pass_classifier_rf.joblib`
- Key features: G1, G2, absences, failures, studytime, higher education intent

Risk thresholds (configurable in `config/model_config.json`):

| Level | Condition |
|-------|-----------|
| High | risk score >= 0.65 |
| Medium | risk score >= 0.50 |
| Low | risk score < 0.50 |

### Dropout Risk Model

- Algorithm: XGBoost classifier (optimized)
- Training data: OULA Virtual Learning Environment dataset
- Prediction window: first 40 days of course engagement
- Target: dropout / non-dropout (binary)
- Saved artifact: `models/dropout/dropout_xgb_optimized.joblib`
- Key features: total VLE clicks, active days, relative engagement, average assessment score, submission lateness, prior attempts, studied credits

Risk thresholds (configurable in `config/model_config.json`):

| Level | Condition |
|-------|-----------|
| High | probability >= 0.51 |
| Medium | probability >= 0.36 |
| Low | probability < 0.36 |

Reported metrics: Precision 0.727 | Recall 0.827 | F1 0.774 | AUC 0.843

## Student Profile and Parent Notifications

The Student Profile page (`pages/4_Student_Profile.py`) provides:

- Risk gauge showing probability score
- SHAP-based explanation of which features are driving the risk
- Parent-readable summary of concerns and protective factors
- Numbered list of recommended actions for parents
- Editable parent notification letter, downloadable as a `.txt` file

For the academic model, the sklearn Pipeline is unwrapped before SHAP computation: the preprocessor transforms the data to dense arrays, and TreeExplainer runs on the extracted Random Forest estimator directly.

## Configuration

`config/model_config.json` controls all thresholds and model paths. Edit this file to adjust risk boundaries without touching code:

```json
{
    "performance": {
        "model_path": "models/performance/pass_classifier_rf.joblib",
        "threshold": 0.5,
        "risk_levels": { "high": 0.65, "medium": 0.5 }
    },
    "dropout": {
        "model_path": "models/dropout/dropout_xgb_optimized.joblib",
        "model_path_fallback": "models/dropout/oula_ews_model.pkl",
        "threshold": 0.36,
        "risk_levels": { "high": 0.51, "medium": 0.36 },
        "metrics": { "precision": 0.727, "recall": 0.827, "f1": 0.774, "auc": 0.843 }
    }
}
```

## Notebook Workflow

Run notebooks in order for each model:

```bash
source .venv/bin/activate
jupyter notebook
```

Performance notebooks (`notebooks/performance/`):

1. `preprocess.ipynb` — clean and merge UCI datasets
2. `train.ipynb` — train and save Random Forest Pipeline
3. `evaluate.ipynb` — generate metrics and feature importance
4. `report.ipynb` — export summary report

Dropout notebooks (`notebooks/dropout/`):

1. `preprocess.ipynb` — engineer features from OULA VLE data
2. `train.ipynb` — train and optimise XGBoost classifier
3. `evaluate.ipynb` — generate confusion matrix, ROC, PR curves
4. `report.ipynb` — export summary report

## Files Required at Runtime

The dashboard requires these files at minimum:

- `data/processed/performance/student_predictions.csv`
- `data/processed/dropout/Student_risk_report.csv`
- `models/performance/pass_classifier_rf.joblib`
- `models/dropout/dropout_xgb_optimized.joblib`
- `models/dropout/model_features.pkl`
- `config/model_config.json`

If `actionable_weekly_risk_report.csv` or `dropout_preprocessed.csv` are missing, the relevant dashboard sections show an empty-state message and the rest of the app continues to function.

## Dependencies

```
numpy
pandas
scikit-learn
xgboost
shap
streamlit
plotly
joblib
pyarrow
imbalanced-learn
```

Full pinned versions are in `requirements.txt`.
