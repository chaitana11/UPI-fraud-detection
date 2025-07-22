# fraud_app.py or app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# App UI settings
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🔍 UPI Fraud Detection App")

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        # Step 1: Prepare features
        if "isFraud" in df.columns:
            X = df.drop("isFraud", axis=1)
        else:
            X = df.copy()

        # Step 2: One-hot encode categorical features
        X_encoded = pd.get_dummies(X)

        # Step 3: Ensure columns match the model
        model_columns = model.get_booster().feature_names
        for col in model_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[model_columns]

        # Step 4: Scale
        X_scaled = scaler.transform(X_encoded)

        # Step 5: Predict
        predictions = model.predict(np.asarray(X_scaled, dtype=np.float32))

        df["Prediction"] = ["🟥 Fraud" if p == 1 else "🟩 Genuine" for p in predictions]

        st.success("✅ Prediction Complete")
        st.subheader("🔎 Results")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error: {e}")
