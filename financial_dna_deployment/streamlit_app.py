"""
Financial DNA -- lightweight loan-officer UI (Streamlit).
Run alongside the API:
    uvicorn app:app --reload --port 8000
    streamlit run streamlit_app.py
"""
import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial DNA -- Client Profile", layout="wide")
st.title("Financial DNA — Client Profile")
st.caption("Behavioural segment, risk, recommended loan limit, lifetime value, and fraud screening — one client at a time.")

BRANCHES = ["Branch-001", "Branch-022", "Branch-023", "Branch-024", "Branch-028", "Branch-031", "Branch-041"]
PRODUCTS = ["Product-A", "Product-B", "Product-C"]
AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]

with st.form("client_form"):
    st.subheader("Client & loan details")
    c1, c2, c3 = st.columns(3)
    with c1:
        branch = st.selectbox("Branch", BRANCHES)
        product = st.selectbox("Product", PRODUCTS)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age_band = st.selectbox("Age band", AGE_BANDS)
        was_refinanced = st.selectbox("Was refinanced", ["No", "Yes"])
    with c2:
        loan_usage = st.text_input("Loan usage", "Business")
        loan_usage_business = st.text_input("Loan usage (business)", "Trade")
        nature_of_business = st.text_input("Nature of business", "Retail")
        env_social_risk = st.selectbox("Environmental & social risk", ["Not Assessed", "Low", "Medium", "High"])
        completed_cycles = st.number_input("Completed loan cycles", min_value=0, value=2)
    with c3:
        loan_amount = st.number_input("Loan amount", min_value=0.0, value=500000.0, step=10000.0)
        interest_rate = st.number_input("Interest rate (%)", min_value=0.0, value=12.5, step=0.5)
        installments = st.number_input("Number of installments", min_value=1, value=12)
        deposits_balance = st.number_input("Deposits balance", min_value=0.0, value=50000.0, step=5000.0)
        arrears_tolerance = st.number_input("Arrears tolerance period (days)", min_value=0, value=30)

    st.markdown("---")
    st.subheader("Loan performance to date (for segmentation, CLV, and fraud screening)")
    c4, c5, c6 = st.columns(3)
    with c4:
        loan_age_days = st.number_input("Loan age (days)", min_value=0, value=200)
        loan_age_months = st.number_input("Loan age (months)", min_value=1.0, value=6.0, step=1.0)
        days_in_arrears = st.number_input("Days in arrears", min_value=0.0, value=0.0)
    with c5:
        total_balance = st.number_input("Total balance", min_value=0.0, value=200000.0, step=10000.0)
        total_paid = st.number_input("Total paid", min_value=0.0, value=300000.0, step=10000.0)
        total_due = st.number_input("Total due", min_value=0.0, value=200000.0, step=10000.0)
    with c6:
        principal_balance = st.number_input("Principal balance", min_value=0.0, value=150000.0, step=10000.0)
        interest_balance = st.number_input("Interest balance", min_value=0.0, value=30000.0, step=5000.0)
        penalty_balance = st.number_input("Penalty balance", min_value=0.0, value=0.0, step=5000.0)
        days_since_last_payment = st.number_input("Days since last payment", min_value=0.0, value=15.0)

    submitted = st.form_submit_button("Get Financial DNA profile", use_container_width=True)


def top_drivers_chart(drivers, title):
    df = pd.DataFrame(drivers)
    if df.empty:
        return
    df = df.set_index("feature")[["shap_value"]].rename(columns={"shap_value": title})
    st.bar_chart(df)


if submitted:
    shared = dict(
        branch=branch, product=product, gender=gender, age_band=age_band,
        loan_usage=loan_usage, loan_usage_business=loan_usage_business,
        nature_of_business=nature_of_business, was_refinanced=was_refinanced,
        environmental_social_risk=env_social_risk, interest_rate=interest_rate,
        number_of_installments=int(installments), completed_loan_cycles=int(completed_cycles),
        deposits_balance=deposits_balance, arrears_tolerance_period=int(arrears_tolerance),
        loan_age_days=int(loan_age_days),
    )
    payload = {
        "risk_req": {**shared, "loan_amount": loan_amount},
        "segment_req": {
            "repayment_ratio": 0.0 if (total_paid + total_due) == 0 else round(total_paid / (total_paid + total_due), 4),
            "days_in_arrears": days_in_arrears,
            "days_since_last_payment": days_since_last_payment,
            "loan_frequency": int(completed_cycles),
            "loan_amount": loan_amount,
            "total_balance": total_balance,
        },
        "clv_req": {
            "age_band": age_band, "gender": gender, "completed_loan_cycles": int(completed_cycles),
            "deposits_balance": deposits_balance, "nature_of_business": nature_of_business,
            "loan_usage": loan_usage, "loan_usage_business": loan_usage_business,
            "branch": branch, "product": product, "loan_amount": loan_amount,
            "interest_rate": interest_rate, "number_of_installments": int(installments),
            "loan_age_months": loan_age_months,
        },
        "fraud_req": {
            "loan_amount": loan_amount, "interest_rate": interest_rate,
            "days_in_arrears": days_in_arrears, "total_balance": total_balance,
            "total_paid": total_paid, "total_due": total_due,
            "principal_balance": principal_balance, "interest_balance": interest_balance,
            "penalty_balance": penalty_balance,
        },
    }

    try:
        resp = requests.post(f"{API_URL}/predict/client-profile", json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the Financial DNA API at {API_URL}. Is `uvicorn app:app --port 8000` running? ({e})")
        st.stop()

    st.markdown("---")
    st.subheader("Financial DNA profile")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Segment", result["segment"]["segment_name"])
    m2.metric("Risk score", f"{result['risk']['risk_score']:.1%}",
               delta="HIGH RISK" if result["risk"]["is_high_risk"] else "within tolerance",
               delta_color="inverse" if result["risk"]["is_high_risk"] else "normal")
    m3.metric("Recommended limit", f"{result['loan_limit']['recommended_loan_limit']:,.0f}")
    m4.metric("Predicted CLV (rate/mo)", f"{result['clv']['predicted_realized_value_rate']:,.0f}")
    m5.metric("Fraud screen", "ANOMALY" if result["fraud"]["is_anomalous"] else "normal",
               delta=f"score {result['fraud']['anomaly_score']:.3f}",
               delta_color="inverse" if result["fraud"]["is_anomalous"] else "off")

    st.markdown("#### Why the model said this (SHAP top drivers)")
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Risk score")
        top_drivers_chart(result["risk"]["top_drivers"], "Risk score")
        st.caption("Recommended loan limit")
        top_drivers_chart(result["loan_limit"]["top_drivers"], "Loan limit")
    with d2:
        st.caption("Customer lifetime value")
        top_drivers_chart(result["clv"]["top_drivers"], "CLV")
        st.caption("Fraud / anomaly score")
        top_drivers_chart(result["fraud"]["top_drivers"], "Fraud score")

    with st.expander("Raw API response"):
        st.json(result)
else:
    st.info("Fill in the client and loan details, then click **Get Financial DNA profile**.")