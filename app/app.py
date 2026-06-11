import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")

model = joblib.load(model_path)


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

h1 {
    color: #38bdf8;
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)


st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter important transaction details to detect fraudulent activity."
)


col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

    v14 = st.slider("V14", -20.0, 10.0, 0.0)
    v10 = st.slider("V10", -20.0, 10.0, 0.0)
    v12 = st.slider("V12", -20.0, 10.0, 0.0)

with col2:

    v17 = st.slider("V17", -20.0, 10.0, 0.0)
    v4  = st.slider("V4",  -10.0, 15.0, 0.0)
    v3  = st.slider("V3",  -30.0, 10.0, 0.0)
    v9  = st.slider("V9",  -20.0, 20.0, 0.0)


input_data = pd.DataFrame({
    'V1':  [0], 'V2':  [0], 'V3':  [v3],  'V4':  [v4],
    'V5':  [0], 'V6':  [0], 'V7':  [0],   'V8':  [0],
    'V9':  [v9], 'V10': [v10], 'V11': [0], 'V12': [v12],
    'V13': [0], 'V14': [v14], 'V15': [0],  'V16': [0],
    'V17': [v17], 'V18': [0], 'V19': [0],  'V20': [0],
    'V21': [0], 'V22': [0], 'V23': [0],   'V24': [0],
    'V25': [0], 'V26': [0], 'V27': [0],   'V28': [0],
    'Amount_scaled': [amount],
    'Time_scaled':   [0]
})

# Ensure column order matches training
input_data = input_data[[
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount_scaled','Time_scaled'
]]

if st.button("🔍 Predict Transaction"):

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    st.metric("Fraud Probability", f"{probability * 100:.2f}%")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")