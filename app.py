import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------------
# Load model & feature list
# ----------------------------------
model = joblib.load("traffic_severity_model.pkl")
feature_cols = joblib.load("features_columns.pkl")

THRESHOLD = 0.15

st.set_page_config(page_title="Traffic Severity Prediction", layout="wide")

st.title("🚦 Traffic Collision Severity Prediction")
st.write(
    "This app predicts whether a traffic collision will be **Severe** or **Non-Severe** "
    "based on high-risk conditions."
)

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("🔧 Input Features")

user_input = {}

for col in feature_cols:
    user_input[col] = st.sidebar.slider(
        col,
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

input_df = pd.DataFrame([user_input])

# ----------------------------------
# Normal Prediction
# ----------------------------------
if st.button("🔍 Predict Severity"):
    proba = model.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("📊 Prediction Result")
    st.write(f"**Severe Accident Probability:** `{proba:.3f}`")

    if pred == 1:
        st.error("🚨 Predicted Outcome: **SEVERE ACCIDENT**")
    else:
        st.success("✅ Predicted Outcome: **NON-SEVERE ACCIDENT**")

    st.caption(f"Decision Threshold = {THRESHOLD}")

# ----------------------------------
# 🔥 Severe Scenario Test Button
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Demo Mode")

if st.sidebar.button("🔥 Test Severe Scenario"):
    test_input = pd.DataFrame([{col: 1.0 for col in feature_cols}])

    proba = model.predict_proba(test_input)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("🔥 High-Risk Scenario Test")
    st.write("All features set to **maximum risk (1.0)**")

    st.write(f"**Severe Accident Probability:** `{proba:.3f}`")

    if pred == 1:
        st.error("🚨 Predicted Outcome: **SEVERE ACCIDENT**")
    else:
        st.warning("⚠️ Model still predicts Non-Severe (very conservative model)")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "📌 This model prioritizes recall of severe accidents and uses threshold tuning "
    "to align with real-world road safety objectives."
)
