"""
Financial DNA REST API -- serves all five components with SHAP-backed
explanations on every prediction, so a flagged client, a loan-limit
figure, a CLV estimate, or a fraud flag can always be traced back to
the features that drove it.

Endpoints:
  GET  /health
  POST /predict/segment         -> Financial DNA behavioural segment (Component 1)
  POST /predict/risk            -> risk score + SHAP drivers          (Component 2)
  POST /predict/loan-limit      -> recommended limit + SHAP drivers   (Component 4)
  POST /predict/clv             -> customer lifetime value + SHAP     (Component 3)
  POST /predict/fraud           -> anomaly flag + SHAP drivers        (Component 5)
  POST /predict/client-profile  -> all of the above combined, for the loan-officer screen
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACT_DIR = 'artifacts'

app = FastAPI(title="Financial DNA API", version="0.2.0")

# ---- load every artifact once, at startup, not per-request ----
risk_model = joblib.load(f'{ARTIFACT_DIR}/risk_model.joblib')
risk_encoders = joblib.load(f'{ARTIFACT_DIR}/risk_encoders.joblib')
limit_model = joblib.load(f'{ARTIFACT_DIR}/limit_model.joblib')
clv_model = joblib.load(f'{ARTIFACT_DIR}/clv_model.joblib')
fraud_model = joblib.load(f'{ARTIFACT_DIR}/fraud_model.joblib')
fraud_feature_cols = joblib.load(f'{ARTIFACT_DIR}/fraud_feature_cols.joblib')
segment_scaler = joblib.load(f'{ARTIFACT_DIR}/segment_scaler.joblib')
segment_kmeans = joblib.load(f'{ARTIFACT_DIR}/segment_kmeans.joblib')
segment_names = joblib.load(f'{ARTIFACT_DIR}/segment_names.joblib')
with open(f'{ARTIFACT_DIR}/metadata.json') as f:
    metadata = json.load(f)

CAT_FEATURES = metadata['cat_features']
RISK_FEATURE_COLS = metadata['risk_feature_cols']
LIMIT_NUM_FEATURES = metadata['limit_num_features']
CLV_FEATURE_COLS = metadata['clv_feature_cols']
CLV_CAT_FEATURES = metadata['clv_cat_features']
RFM_COLS = metadata['rfm_cols']
ANOMALY_COLS = metadata['anomaly_cols']
RISK_THRESHOLD = metadata['risk_threshold']

# ---- SHAP explainers -- cheap to build from a saved tree model, so we
# build them once at startup rather than persisting explainer objects ----
risk_explainer = shap.TreeExplainer(risk_model)
limit_explainer = shap.TreeExplainer(limit_model)
clv_explainer = shap.TreeExplainer(clv_model)
fraud_explainer = shap.TreeExplainer(fraud_model)


def top_shap_drivers(shap_row, feature_names, feature_values, n=5):
    """Turns one row of SHAP values into a small, UI-friendly list of the
    n features that pushed the prediction most, in either direction."""
    order = np.argsort(-np.abs(shap_row))[:n]
    return [
        {
            "feature": feature_names[i],
            "value": feature_values[i] if not isinstance(feature_values[i], (np.generic,)) else feature_values[i].item(),
            "shap_value": round(float(shap_row[i]), 4),
            "direction": "increases" if shap_row[i] > 0 else "decreases",
        }
        for i in order
    ]


class LoanFeatures(BaseModel):
    branch: str
    product: str
    gender: str = Field(..., description="Male / Female")
    age_band: str
    loan_usage: str
    loan_usage_business: str
    nature_of_business: str
    was_refinanced: str = Field(..., description="'Yes' or 'No'")
    environmental_social_risk: str
    interest_rate: float
    number_of_installments: int
    completed_loan_cycles: int
    deposits_balance: float
    arrears_tolerance_period: int
    loan_age_days: int


class RiskRequest(LoanFeatures):
    loan_amount: float


class LoanLimitRequest(LoanFeatures):
    pass


class SegmentRequest(BaseModel):
    repayment_ratio: float
    days_in_arrears: float
    days_since_last_payment: float
    loan_frequency: int
    loan_amount: float
    total_balance: float


class ClvRequest(BaseModel):
    age_band: str
    gender: str
    completed_loan_cycles: int
    deposits_balance: float
    nature_of_business: str
    loan_usage: str
    loan_usage_business: str
    branch: str
    product: str
    loan_amount: float
    interest_rate: float
    number_of_installments: int
    loan_age_months: float


class FraudRequest(BaseModel):
    loan_amount: float
    interest_rate: float
    days_in_arrears: float
    total_balance: float
    total_paid: float
    total_due: float
    principal_balance: float
    interest_balance: float
    penalty_balance: float


def _to_risk_row(req: RiskRequest) -> pd.DataFrame:
    row = {
        'Branch': req.branch, 'Product': req.product,
        'Gender (Client)': req.gender, 'Age Band (Client)': req.age_band,
        'Loan Usage': req.loan_usage, 'Loan usage Business': req.loan_usage_business,
        'Nature of business': req.nature_of_business, 'Was Refinanced': req.was_refinanced,
        'Environmental & Social Risk': req.environmental_social_risk,
        'Loan Amount': req.loan_amount, 'Interest Rate': req.interest_rate,
        'Number of Installments': req.number_of_installments,
        'Completed Loan Cycles (Client)': req.completed_loan_cycles,
        'Deposits Balance (Client)': req.deposits_balance,
        'Arrears Tolerance Period': req.arrears_tolerance_period,
        'Loan Age Days': req.loan_age_days,
    }
    X = pd.DataFrame([row])
    X['Branch Arrears Rate'] = X['Branch'].map(risk_encoders['branch_map']).fillna(
        risk_encoders['global_arrears_rate'])
    X['Product Arrears Rate'] = X['Product'].map(risk_encoders['product_map']).fillna(
        risk_encoders['global_arrears_rate'])
    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')
    return X[RISK_FEATURE_COLS]


def _to_limit_row(req: LoanLimitRequest) -> pd.DataFrame:
    row = {
        'Branch': req.branch, 'Product': req.product,
        'Gender (Client)': req.gender, 'Age Band (Client)': req.age_band,
        'Loan Usage': req.loan_usage, 'Loan usage Business': req.loan_usage_business,
        'Nature of business': req.nature_of_business, 'Was Refinanced': req.was_refinanced,
        'Environmental & Social Risk': req.environmental_social_risk,
        'Interest Rate': req.interest_rate, 'Number of Installments': req.number_of_installments,
        'Completed Loan Cycles (Client)': req.completed_loan_cycles,
        'Deposits Balance (Client)': req.deposits_balance,
        'Arrears Tolerance Period': req.arrears_tolerance_period,
        'Loan Age Days': req.loan_age_days,
    }
    X = pd.DataFrame([row])
    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')
    return X[CAT_FEATURES + LIMIT_NUM_FEATURES]


def _to_clv_row(req: ClvRequest) -> pd.DataFrame:
    row = {
        'Age Band (Client)': req.age_band, 'Gender (Client)': req.gender,
        'Completed Loan Cycles (Client)': req.completed_loan_cycles,
        'Deposits Balance (Client)': req.deposits_balance,
        'Nature of business': req.nature_of_business, 'Loan Usage': req.loan_usage,
        'Loan usage Business': req.loan_usage_business, 'Branch': req.branch,
        'Product': req.product, 'Loan Amount': req.loan_amount,
        'Interest Rate': req.interest_rate, 'Number of Installments': req.number_of_installments,
        'Loan_Age_Months': max(req.loan_age_months, 1.0),
    }
    X = pd.DataFrame([row])
    for c in CLV_CAT_FEATURES:
        X[c] = X[c].astype('category')
    return X[CLV_FEATURE_COLS]


def _to_fraud_row(req: FraudRequest) -> pd.DataFrame:
    row = {
        'Loan Amount': req.loan_amount, 'Interest Rate': req.interest_rate,
        'Days In Arrears': req.days_in_arrears, 'Total Balance': req.total_balance,
        'Total Paid': req.total_paid, 'Total Due': req.total_due,
        'Principal Balance': req.principal_balance, 'Interest Balance': req.interest_balance,
        'Penalty Balance': req.penalty_balance,
    }
    return pd.DataFrame([row])[ANOMALY_COLS]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/risk")
def predict_risk(req: RiskRequest):
    X = _to_risk_row(req)
    proba = float(risk_model.predict_proba(X)[0, 1])
    shap_vals = risk_explainer.shap_values(X)
    row = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    return {
        "risk_score": round(proba, 4),
        "is_high_risk": proba >= RISK_THRESHOLD,
        "threshold_used": round(RISK_THRESHOLD, 4),
        "top_drivers": top_shap_drivers(row, list(X.columns), X.iloc[0].tolist()),
    }


@app.post("/predict/loan-limit")
def predict_loan_limit(req: LoanLimitRequest):
    X = _to_limit_row(req)
    pred = float(limit_model.predict(X)[0])
    shap_vals = limit_explainer.shap_values(X)[0]
    return {
        "recommended_loan_limit": round(pred, 2),
        "top_drivers": top_shap_drivers(shap_vals, list(X.columns), X.iloc[0].tolist()),
    }


@app.post("/predict/clv")
def predict_clv(req: ClvRequest):
    X = _to_clv_row(req)
    pred = float(clv_model.predict(X)[0])
    shap_vals = clv_explainer.shap_values(X)[0]
    return {
        "predicted_realized_value_rate": round(pred, 2),
        "top_drivers": top_shap_drivers(shap_vals, list(X.columns), X.iloc[0].tolist()),
    }


@app.post("/predict/fraud")
def predict_fraud(req: FraudRequest):
    X = _to_fraud_row(req)
    flag = int(fraud_model.predict(X)[0])
    score = float(fraud_model.decision_function(X)[0])
    shap_vals = fraud_explainer.shap_values(X)[0]
    return {
        "is_anomalous": flag == -1,
        "anomaly_score": round(score, 4),
        "top_drivers": top_shap_drivers(shap_vals, list(X.columns), X.iloc[0].tolist()),
    }


@app.post("/predict/segment")
def predict_segment(req: SegmentRequest):
    row = pd.DataFrame([[
        req.repayment_ratio, req.days_in_arrears, req.days_since_last_payment,
        req.loan_frequency, req.loan_amount, req.total_balance,
    ]], columns=RFM_COLS)
    X_scaled = segment_scaler.transform(row)
    cluster_id = int(segment_kmeans.predict(X_scaled)[0])
    return {
        "segment_id": cluster_id,
        "segment_name": segment_names.get(cluster_id, segment_names.get(str(cluster_id))),
    }


@app.post("/predict/client-profile")
def predict_client_profile(risk_req: RiskRequest, segment_req: SegmentRequest,
                            clv_req: ClvRequest, fraud_req: FraudRequest):
    """Bundles every component for the loan-officer screen, so the UI
    doesn't need five separate round trips. Note the request body must
    nest each part under its own key: {"risk_req": {...}, "segment_req": {...},
    "clv_req": {...}, "fraud_req": {...}} — FastAPI requires this when a
    single endpoint takes multiple Pydantic body parameters."""
    risk_out = predict_risk(risk_req)
    limit_out = predict_loan_limit(LoanLimitRequest(**risk_req.dict(exclude={'loan_amount'})))
    segment_out = predict_segment(segment_req)
    clv_out = predict_clv(clv_req)
    fraud_out = predict_fraud(fraud_req)
    return {"risk": risk_out, "loan_limit": limit_out, "segment": segment_out,
            "clv": clv_out, "fraud": fraud_out}