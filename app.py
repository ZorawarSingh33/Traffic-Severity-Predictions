import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
/* App background + text */
.stApp {
    background-color: #E8E2DB;
    color: #1A3263;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #4A5A6A;
}

/* Sidebar text */
[data-testid="stSidebar"] .block-container {
    color: #E3E7EB !important;
}

/* Slider labels + min/max values */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider span {
    color: #C9D1D9 !important;
    font-weight: 500;
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
    background-color: #162447;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-color: #6FA3BF;
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
# Header
# ----------------------------------
st.title("🚦 Traffic Collision Severity Prediction")

st.write(
    "This application estimates the **probability of a severe traffic collision** "
    "based on selected risk-related factors. The model is calibrated to prioritize "
    "**early detection of severe outcomes** for road safety decision support."
)

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("🔧 Risk Factor Inputs")
st.sidebar.write("Higher values represent higher relative risk.")

user_input = {
    col: st.sidebar.slider(
        col,
        0.0,
        1.0,
        0.0,
        0.05
    )
    for col in feature_cols
}

input_df = pd.DataFrame([user_input])

# ----------------------------------
# Prediction + Graph
# ----------------------------------
if st.button("🔍 Assess Collision Severity"):
    proba = model.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("⚠️ Risk Assessment Result")

    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.barh(
        ["Non-Severe", "Severe"],
        [1 - proba, proba],
        color=["#6FA3BF", "#D9A87E"]  # subtle colors
    )
    ax.axvline(
        THRESHOLD,
        color="#2F3B4C",
        linestyle="--",
        linewidth=2,
        label="Decision Threshold"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.legend(loc="lower right")
    ax.set_title("Predicted Collision Severity Risk", fontsize=12)
    st.pyplot(fig)

    if pred == 1:
        st.error(
            f"**High Risk Detected** — Estimated probability of severe collision: "
            f"**{proba:.2%}**"
        )
    else:
        st.success(
            f"**Lower Risk Detected** — Estimated probability of severe collision: "
            f"**{proba:.2%}**"
        )

    st.caption(
        f"Decision threshold set at {THRESHOLD}, "
        "favoring conservative, safety-focused predictions."
    )

# ----------------------------------
# High-Risk Scenario Demo
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Scenario Demonstration")

if st.sidebar.button("Run High-Risk Simulation"):
    test_input = pd.DataFrame([{col: 1.0 for col in feature_cols}])
    proba = model.predict_proba(test_input)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("🔥 High-Risk Scenario Outcome")

    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.barh(
        ["Non-Severe", "Severe"],
        [1 - proba, proba],
        color=["#6FA3BF", "#D9A87E"]
    )
    ax.axvline(THRESHOLD, color="#2F3B4C", linestyle="--", linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_title("Extreme Risk Simulation")
    st.pyplot(fig)

    if pred == 1:
        st.error(f"Severe collision probability: **{proba:.2%}**")
    else:
        st.warning(
            f"Model remains conservative even at extreme inputs "
            f"(probability: **{proba:.2%}**)."
        )

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption(
    "📌 Demonstrates applied machine learning for road safety using "
    "probability-based risk scoring and threshold optimization."
)

