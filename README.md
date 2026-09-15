# Financial DNA

A client behavioural profiling and risk-screening system for fintech lending, built on a real 23,754-account loan portfolio. Financial DNA segments clients into behavioural profiles, predicts arrears risk and a recommended loan limit at origination, estimates each client's ongoing value, and screens accounts for anomalous financial behaviour — with every prediction explained through SHAP and served through a REST API and a Streamlit dashboard.

## Problem

Loan officers currently make lending decisions — how much to lend, to whom, and how closely to monitor a given account — largely on gut feel and static risk categories, with no unified view of a client's behavioural history and no systematic, explainable way to flag unusual accounts. This project builds that unified view: a single "Financial DNA" profile per client, backed by tuned machine learning models rather than manual rules, with every model output traceable back to the specific features that drove it.

## Components

| # | Component | Model | What it does |
|---|-----------|-------|---------------|
| 1 | Financial DNA Profile | K-Means (k=4) on RFM features | Segments clients into behavioural profiles: Reliable Payer, Established Repeat Borrower, At-Risk/Currently Delinquent, Chronic Defaulter |
| 2 | Risk / Arrears Prediction | LightGBM Classifier (Optuna-tuned) | Predicts probability a loan goes into arrears, using only origination-time-knowable features |
| 3 | Customer Lifetime Value | LightGBM Regressor (RandomizedSearchCV-tuned) | Predicts a client's realized monthly value (interest + fees per month of loan age) — replaces the originally-planned known-pattern fraud classifier, which would have required confirmed fraud labels this dataset doesn't have |
| 4 | Loan Limit Estimation | LightGBM Regressor (Optuna-tuned) | Recommends a loan amount from client and loan-term characteristics |
| 5 | Fraud Detection (unknown patterns) | Isolation Forest | Flags accounts with anomalous financial/repayment characteristics, without needing labeled fraud data |

Each of the four tree-based models (2–5) is explained with SHAP; K-Means (1) is interpreted by its behavioural profile instead, since Shapley values don't apply to a non-tree clustering model.

## Model performance

| Model | Metric | Score |
|---|---|---|
| Segmentation | Silhouette score (k=4) | 0.419 |
| Risk / Arrears | Test ROC-AUC / PR-AUC | 0.875 / 0.589 |
| Risk / Arrears | Operating threshold (F2-optimal) | 0.409 |
| Customer Lifetime Value | Test R² / MAE | 0.79 / ~20,551 |
| Loan Limit | Test R² / MAE | 0.53 / ~1,369,958 |
| Fraud (Isolation Forest) | Anomaly rate (contamination) | 5.0% (1,188 / 23,754 accounts) |

The Risk model's operating threshold isn't the default 0.5 — it's chosen by scanning the precision-recall curve for the F2-optimal cutoff (recall-weighted), since missing a real default is costlier than one extra manual review. The Isolation Forest's anomaly rate is directly set by its `contamination=0.05` parameter, which sklearn uses to derive the model's internal decision boundary at training time.

## Explainability (SHAP)

Every tree-based model is explained with `shap.TreeExplainer`, at two levels:

- **Global** — summary plots (bar + beeswarm) showing which features matter most across the whole test set, for each of the four tree models.
- **Individual** — waterfall plots tracing one specific client's prediction from the model's average output to their actual score, feature by feature, with all contributions summing exactly to the final prediction.

In production, the REST API computes the same SHAP values per request and returns each prediction's top 5 drivers, so the explanation isn't notebook-only — it's part of the live API contract and surfaces directly on the loan officer's dashboard.

## Deployment architecture

```
financial_dna.ipynb  ──▶  artifacts/*.joblib, metadata.json
                                   │
                                   ▼
                              app.py (FastAPI)
                          /predict/segment
                          /predict/risk
                          /predict/loan-limit
                          /predict/clv
                          /predict/fraud
                          /predict/client-profile
                                   │
                                   ▼
                      streamlit_app.py (Streamlit UI)
```

Trained models and every object needed to preprocess a new request the same way training data was preprocessed (feature-column lists, categorical dtypes, target-encoding maps, the tuned risk threshold) are saved to `artifacts/` via `joblib`. The FastAPI backend loads all of this once at startup, builds a SHAP explainer per model, and exposes one endpoint per prediction plus a combined `/predict/client-profile` endpoint for the dashboard. The Streamlit UI is a thin client: one form, one API call per lookup, rendered as five metric tiles (segment, risk, loan limit, CLV, fraud flag) with SHAP driver charts underneath each.

## Project structure

```
financial_dna/
├── financial_dna.ipynb       # full notebook: EDA, cleaning, all 5 components, SHAP, artifact saving
├── app.py                    # FastAPI backend
├── streamlit_app.py          # Streamlit dashboard
├── artifacts/
│   ├── segment_scaler.joblib
│   ├── segment_kmeans.joblib
│   ├── segment_names.joblib
│   ├── risk_model.joblib
│   ├── risk_encoders.joblib
│   ├── clv_model.joblib
│   ├── limit_model.joblib
│   ├── fraud_model.joblib
│   ├── fraud_feature_cols.joblib
│   └── metadata.json
└── README.md
```

## Setup

```bash
pip install fastapi uvicorn shap joblib pandas numpy lightgbm scikit-learn streamlit requests optuna
```

or

```bash
pip install requirements.txt
```


## Running it

**1. Train and save the models** — run `financial_dna.ipynb` top to bottom (Restart & Run All). This populates `artifacts/`.

**2. Start the API** (from the project root, so the relative `artifacts/` path resolves):

```bash
uvicorn app:app --reload --port 8000
```

Verify at `http://127.0.0.1:8000/health` (should return `{"status":"ok"}`) or browse the interactive docs at `http://127.0.0.1:8000/docs`.

**3. Start the dashboard**, in a second terminal, with the API still running:

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Fill in a client's details and click **Get Financial DNA profile** to see their segment, risk score, recommended limit, CLV estimate, and fraud screen, each with its SHAP-driven explanation.

## API reference

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | Service status |
| `/predict/segment` | POST | Behavioural segment name |
| `/predict/risk` | POST | Risk score, high-risk flag, threshold used, top SHAP drivers |
| `/predict/loan-limit` | POST | Recommended loan limit, top SHAP drivers |
| `/predict/clv` | POST | Predicted realized value rate, top SHAP drivers |
| `/predict/fraud` | POST | Anomaly flag, anomaly score, top SHAP drivers |
| `/predict/client-profile` | POST | All five combined, for the dashboard |

`/predict/client-profile` takes a request body with four nested keys — `risk_req`, `segment_req`, `clv_req`, `fraud_req` — since FastAPI requires each Pydantic body parameter nested under its own name when an endpoint takes more than one.

## Data

`Loan_approval.xlsx` — 23,754 accounts, 49 original columns, cleaned to 44 (dropped near-fully-missing identifier columns, rebuilt `Total Balance` from its component balances, engineered `Has Made Last Payment`, dropped collinear/zero-variance columns). No usable client identifier exists in the raw data (`Account Holder ID`/`Account Holder Name` are 100% missing), which is why the Financial DNA profile is built from behavioural (RFM) features rather than any client lookup key.

## Known limitations

Loan Limit and Customer Lifetime Value are continuous regression outputs with no categorical banding — unlike Risk (tuned probability threshold) and Fraud (contamination-based decision boundary), there's no "high CLV" or "high limit" cutoff defined anywhere in the code. Segmentation has no SHAP explanation, since K-Means isn't a tree-based model. The Loan Limit model's R² (0.53) is noticeably lower than Risk's or CLV's, reflecting that recommending a specific loan amount from origination-time features alone is a genuinely harder prediction problem than classifying arrears risk or estimating realized value.

