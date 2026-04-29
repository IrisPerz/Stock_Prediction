import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
import shap
from joblib import dump, load

warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# FIX 1: removed the drop Unnamed column line since your X_train won't have it
file_path = os.path.join(current_dir, 'X_train.csv')
dataset = pd.read_csv(file_path)

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_loan.shap",
    # FIX 2: changed model.tar.gz to finalized_loan_model.tar.gz to match your notebook
    "pipeline"  : "finalized_loan_model.tar.gz",
    "keys"      : ['loan_amnt', 'int_rate', 'annual_inc', 'dti'],
    "inputs"    : [
        {"name": "loan_amnt",  "min": 500.0,   "max": 40000.0,  "default": 10000.0, "step": 500.0},
        {"name": "int_rate",   "min": 5.0,     "max": 30.0,     "default": 13.5,    "step": 0.1},
        {"name": "annual_inc", "min": 10000.0, "max": 500000.0, "default": 55000.0, "step": 1000.0},
        {"name": "dti",        "min": 0.0,     "max": 50.0,     "default": 18.5,    "step": 0.1}
    ]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # FIX 3: removed extra } that was causing a syntax error
        mapping = {0: "✅ Fully Paid", 1: "⚠️ Charged Off (Default)"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    imputer = best_pipeline.named_steps['imputer']
    scaler  = best_pipeline.named_steps['scaler']
    input_df = pd.DataFrame(input_df)
    input_df_transformed = scaler.transform(imputer.transform(input_df))
    feature_names = dataset.columns.tolist()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    # FIX 4: removed [0, :, 1] indexing that caused errors — use [0] directly
    shap.plots.waterfall(shap_values[0], max_display=12)
    st.pyplot(fig)
    top_feature = pd.Series(
        shap_values[0].values,
        index=shap_values[0].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("💳 Loan Default Predictor — LendingClub")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

original = dataset.iloc[0:1].to_dict()
original.update(user_inputs)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
