import streamlit as st
import pandas as pd
import boto3
import json

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="centered")
st.title("💳 Loan Default Predictor")
st.markdown("Enter borrower details to predict whether a loan will be **Fully Paid** or **Charged Off**.")

# Sidebar: AWS credentials
with st.sidebar:
    st.header("🔑 AWS Credentials")
    st.markdown("Paste these from your **AWS Learner Lab** session.")
    access_key    = st.text_input("Access Key ID",     type="password")
    secret_key    = st.text_input("Secret Access Key", type="password")
    session_token = st.text_input("Session Token",     type="password")
    region        = st.text_input("Region",            value="us-east-1")
    endpoint_name = st.text_input("Endpoint Name",     value="loan-default-endpoint")
    st.caption("Credentials expire every few hours. Re-paste from Learner Lab when needed.")

# Input form
st.subheader("Borrower Information")
col1, col2 = st.columns(2)

with col1:
    loan_amnt  = st.number_input("Loan Amount ($)",   min_value=500,   max_value=40000,  value=10000, step=500)
    int_rate   = st.slider("Interest Rate (%)",       min_value=5.0,   max_value=30.0,   value=13.5,  step=0.1)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=55000, step=1000)
    dti        = st.slider("Debt-to-Income Ratio",    min_value=0.0,   max_value=50.0,   value=18.5,  step=0.1)

with col2:
    term           = st.selectbox("Loan Term",      [" 36 months", " 60 months"])
    grade          = st.selectbox("Loan Grade",     ["A", "B", "C", "D", "E", "F", "G"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

st.markdown("---")

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    if not access_key or not secret_key or not session_token:
        st.error("Please enter your AWS credentials in the sidebar first.")
    else:
        input_data = pd.DataFrame([{
            "loan_amnt":      loan_amnt,
            "term":           term,
            "int_rate":       int_rate,
            "annual_inc":     annual_inc,
            "dti":            dti,
            "home_ownership": home_ownership,
            "grade":          grade
        }])

        try:
            boto_session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token,
                region_name=region
            )
            runtime = boto_session.client("sagemaker-runtime")

            payload = input_data.to_csv(index=False)
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="text/csv",
                Body=payload
            )

            result = json.loads(response["Body"].read().decode("utf-8"))
            prediction = result[0] if isinstance(result, list) else result

            if prediction == 1:
                st.error("⚠️ **Charged Off** — This borrower is predicted to **default**.")
            else:
                st.success("✅ **Fully Paid** — This borrower is predicted to **repay** the loan.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
