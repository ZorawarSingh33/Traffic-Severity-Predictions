import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# App Configuration
# ----------------------------------
st.set_page_config(
    page_title="Traffic Severity Prediction",
    layout="wide"
)

# ----------------------------------
# Custom Color Theme Styling
# ----------------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #E8E2DB;
    color: #1A3263;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #547792;
    color: white;
}

/* Headings */
h1, h2, h3 {
    color: #1A3263;
}

/* Buttons */
.stButton > button {
    background-color: #1A3263;
    color: white;
    border-radius: 6px;
    padding: 0.55em 1.2em;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #547792;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-color: #FAB95B;
}

/* Alert boxes */
div.stAlert-success {
    background-color: rgba(84, 119, 146, 0.15);
    color: #1A3263;
}

div.stAlert-error {
    background-color: rgba(250, 185, 91, 0.25);
    color: #1A3263;
}

div.stAlert-warning {
    background-color: rgba(250, 185, 91, 0.15);
    color: #1A3263;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load Model & Features
# ----------------------------------
model = joblib.load("traffic_severity_model.pkl")
feature_cols = joblib.load("features_columns.pkl")

THRESHOLD = 0.15

# ----------------------------------
# App Header
# ----------------------------------
st.title("🚦 Traffic Collision Severity Prediction")

st.write(
    "This application estimates the **likelihood of a severe traffic collision** "
    "based on key risk-related factors. The model is intentionally calibrated "
    "to prioritize **road safety and early risk detection**, making it suitable "
    "for real-world decision support."
)

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("🔧 Risk Factor Inputs")

st.sidebar.write(
    "Adjust the sliders to simulate different road, vehicle, and environmental conditions. "
    "Higher values indicate higher relative risk."
)

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
# Prediction Output
# ----------------------------------
if st.button("🔍 Assess Collision Severity"):
    proba = model.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("⚠️ Risk Assessment Result")

    st.progress(float(proba))

    if pred == 1:
        st.error(
            f"**High Risk Detected** — Estimated probability of a severe collision: "
            f"**{proba:.2%}**"
        )
    else:
        st.success(
            f"**Lower Risk Detected** — Estimated probability of a severe collision: "
            f"**{proba:.2%}**"
        )

    st.caption(
        f"Classification threshold set at {THRESHOLD}, "
        "favoring conservative safety-focused predictions."
    )

# ----------------------------------
# High-Risk Scenario Demo
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Scenario Demonstration")

st.sidebar.write(
    "This option simulates a worst-case scenario by setting all risk factors "
    "to their maximum values."
)

if st.sidebar.button("Run High-Risk Simulation"):
    test_input = pd.DataFrame([{col: 1.0 for col in feature_cols}])
    proba = model.predict_proba(test_input)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("🔥 High-Risk Scenario Outcome")
    st.progress(float(proba))

    if pred == 1:
        st.error(f"Severe collision probability under extreme conditions: **{proba:.2%}**")
    else:
        st.warning(
            f"The model remains conservative even under extreme inputs "
            f"(probability: **{proba:.2%}**)."
        )

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "📌 This predictive system demonstrates applied machine learning for road safety, "
    "combining probability-based risk scoring with threshold optimization to support "
    "data-driven decision-making."
)

