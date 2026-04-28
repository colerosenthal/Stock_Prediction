"""
Fraud Detection Streamlit App
IEEE-CIS Fraud Detection | Cole Rosenthal | April 2026

Professor suggestion implemented:
- Loads X_train.csv from GitHub Portfolio folder
- User provides values for top 4 features only
- Remaining columns filled from X_train mean values
- JSONSerializer used for endpoint calls
- SHAP waterfall plot displayed
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import tarfile
import tempfile
import posixpath
import json

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

import shap
from joblib import load

warnings.simplefilter("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — IEEE-CIS",
    page_icon="🛡️",
    layout="wide"
)

# ── AWS credentials ──────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS session ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Model info ───────────────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_fraud.shap",
    "pipeline"  : "finalized_fraud_model.tar.gz",
    # Top 4 most important features from LightGBM feature importance
    # Update these keys to match the actual top features from your notebook output
    "keys"      : ["C1", "C2", "C5", "C6"],
    "inputs"    : [
        {"name": "C1",  "min": 0.0, "max": 10.0,  "default": 1.0, "step": 0.1,
         "help": "Transaction count feature C1"},
        {"name": "C2",  "min": 0.0, "max": 10.0,  "default": 1.0, "step": 0.1,
         "help": "Transaction count feature C2"},
        {"name": "C5",  "min": 0.0, "max": 5.0,   "default": 0.0, "step": 0.1,
         "help": "Transaction count feature C5"},
        {"name": "C6",  "min": 0.0, "max": 10.0,  "default": 1.0, "step": 0.1,
         "help": "Transaction count feature C6"},
    ]
}

# ── Load pipeline and artifacts from S3 ─────────────────────────────────────
@st.cache_resource
def load_artifacts(_session, bucket):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"sklearn-pipeline-deployment/{filename}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_files = [f for f in tar.getnames() if f.endswith('.joblib')]

    feature_names = joblib.load('fraud_feature_names.joblib')
    threshold     = joblib.load('fraud_threshold.joblib')
    return feature_names, float(threshold)

feature_names, best_threshold = load_artifacts(session, aws_bucket)

# ── Load X_train from GitHub (Portfolio folder) ──────────────────────────────
# Professor suggestion: save X_train.csv in Portfolio folder on GitHub
# so Streamlit can fill non-displayed columns with training data means
@st.cache_data
def load_x_train():
    """
    Load X_train.csv from the Portfolio folder in the GitHub repo.
    Update the URL below to match your actual GitHub repo path.
    """
    GITHUB_RAW_URL = (
        "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/"
        "YOUR_REPO_NAME/main/Portfolio/X_train.csv"
    )
    try:
        X_train = pd.read_csv(GITHUB_RAW_URL)
        return X_train
    except Exception:
        # Fallback: return zeros if file not yet uploaded
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

X_train_ref = load_x_train()

# Compute column means from X_train for filling non-displayed features
train_means = X_train_ref.mean().to_dict()

# ── Load SHAP explainer ──────────────────────────────────────────────────────
def load_shap_explainer(_session, bucket):
    s3_client  = _session.client('s3')
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])
    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key=posixpath.join('explainer', MODEL_INFO["explainer"])
        )
    with open(local_path, "rb") as f:
        return load(f)

# ── Prediction via SageMaker endpoint ───────────────────────────────────────
def call_endpoint(input_dict):
    """
    Call SageMaker endpoint with a JSON dict.
    JSON keeps column names — matches what input_fn expects.
    """
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        result     = predictor.predict(input_dict)
        pred_label = int(result[0][0])
        pred_proba = float(result[0][1])
        return pred_label, pred_proba, 200
    except Exception as e:
        return None, None, str(e)

# ── Load pipeline for SHAP preprocessing ────────────────────────────────────
def load_pipeline(_session, bucket):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    if not os.path.exists(filename):
        s3_client.download_file(
            Filename=filename,
            Bucket=bucket,
            Key=f"sklearn-pipeline-deployment/{filename}"
        )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
    return joblib.load('fraud_model_lgbm.joblib')

# ── SHAP waterfall display ───────────────────────────────────────────────────
def display_shap(input_df, session, bucket):
    try:
        # Recreate explainer directly from the loaded model
        model = load_pipeline(session, bucket)
        explainer = shap.TreeExplainer(model)
        input_array = np.array(input_df.values.astype(float))
        shap_values = explainer.shap_values(input_array)

        st.subheader("🔍 Decision Transparency (SHAP Waterfall Plot)")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, input_df, show=False, plot_type='bar')
        plt.tight_layout()
        st.pyplot(fig)

        top_idx = int(np.abs(shap_values[0]).argmax())
        top_feature = (feature_names[top_idx]
                       if top_idx < len(feature_names) else str(top_idx))
        st.info(
            f"**Key Driver:** The feature most responsible for this prediction "
            f"was **{top_feature}**."
        )
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
        
# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ IEEE-CIS Fraud Detection")
st.markdown("**Real-time fraud scoring powered by LightGBM on AWS SageMaker**")
st.markdown("---")

# Stats bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",      "LightGBM (Tuned)")
c2.metric("Features",   str(len(feature_names)))
c3.metric("Threshold",  f"{best_threshold:.3f}")
c4.metric("Deployment", "AWS SageMaker")

st.markdown("---")
st.subheader("Transaction Input")
st.markdown(
    "Enter values for the **top 4 most predictive features**. "
    "All remaining features are filled automatically from training data averages."
)

with st.form("prediction_form"):
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                label=inp["name"],
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
                help=inp.get("help", "")
            )
    submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

if submitted:
    # Build full feature dict — start with training means, override with user inputs
    input_dict = {feat: float(train_means.get(feat, 0.0)) for feat in feature_names}
    for feat, val in user_inputs.items():
        if feat in input_dict:
            input_dict[feat] = float(val)

    # Also build DataFrame for SHAP
    input_df = pd.DataFrame([input_dict])[feature_names]

    with st.spinner("Contacting SageMaker endpoint..."):
        pred_label, pred_proba, status = call_endpoint(input_dict)

    if status == 200:
        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            if pred_label == 1:
                st.error("⚠️ **FRAUD DETECTED**")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")
        with r2:
            st.metric("Fraud Probability", f"{pred_proba:.4f}")
        with r3:
            if pred_proba >= best_threshold:
                risk = "HIGH 🔴"
            elif pred_proba >= 0.20:
                risk = "MEDIUM 🟡"
            else:
                risk = "LOW 🟢"
            st.metric("Risk Level", risk)

        st.markdown("**Fraud Risk Score**")
        st.progress(min(float(pred_proba), 1.0))

        # SHAP explanation
        display_shap(input_df, session, aws_bucket)

    else:
        st.error(f"Endpoint error: {status}")

st.markdown("---")
st.caption(
    "IEEE-CIS Fraud Detection | Cole Rosenthal | Machine Learning for Finance | April 2026"
)
