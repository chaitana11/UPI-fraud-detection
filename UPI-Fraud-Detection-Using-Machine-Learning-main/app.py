import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from datetime import datetime as dt
import base64
import pickle
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt



# Load model
pickle_file_path = "UPI Fraud Detection Final.pkl"
loaded_model = pickle.load(open(pickle_file_path, 'rb'))



# Setup
st.set_page_config(page_title="UPI Fraud Detector", layout="centered")

# Title
st.title("🔐 UPI Transaction Fraud Detector")
st.markdown("Detect fraudulent UPI transactions using machine learning. Analyze single entries or upload CSV files for bulk prediction. Advanced dashboard features included.")

# Dropdown options
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
      'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
      'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment',
      'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

# Tabs
tab = st.sidebar.radio("Navigation", ["💼 Single Check", "📂 CSV Upload", "📈 Dashboard", "🧠 Explainability", "⚙️ Admin Panel"])

def preprocess_inputs(amt, year, month, ttype, gateway, state, mcat):
    tt_oh = [1 if x == ttype else 0 for x in tt]
    pg_oh = [1 if x == gateway else 0 for x in pg]
    ts_oh = [1 if x == state else 0 for x in ts]
    mc_oh = [1 if x == mcat else 0 for x in mc]
    return [amt, year, month] + tt_oh + pg_oh + ts_oh + mc_oh

if tab == "💼 Single Check":
    st.subheader("🔍 Inspect a Single Transaction")
    with st.form("single_transaction_form"):
        tran_date = st.date_input("Transaction Date", datetime.date.today())
        month = tran_date.month
        year = tran_date.year

        col1, col2 = st.columns(2)
        with col1:
            tran_type = st.selectbox("Transaction Type", tt)
            tran_state = st.selectbox("Transaction State", ts)
        with col2:
            pmt_gateway = st.selectbox("Payment Gateway", pg)
            merch_cat = st.selectbox("Merchant Category", mc)

        amt = st.number_input("Transaction Amount", min_value=0.0, step=0.1)

        single_check = st.form_submit_button("Check Transaction")

    if single_check:
        with st.spinner("Checking transaction..."):
            features = preprocess_inputs(amt, year, month, tran_type, pmt_gateway, tran_state, merch_cat)
            result = loaded_model.predict([features])[0]
            st.success("Transaction Checked!")
            st.markdown("✅ **Not Fraudulent**" if result == 0 else "⚠️ **Fraudulent Transaction Detected!**")

elif tab == "⚙️ Admin Panel":
    st.subheader("⚙️ Admin Panel – Retrain Model")
    st.info("Upload labeled data (must include 'fraud' column) to retrain the model.")

    new_data = st.file_uploader("📁 Upload Labeled CSV", type="csv")

    if new_data:
        df_new = pd.read_csv(new_data)
        st.write("✅ Data Preview:")
        st.dataframe(df_new.head())

        if 'fraud' not in df_new.columns:
            st.error("❌ 'fraud' column is required in uploaded data.")
        else:
            if st.button("🚀 Retrain Model"):
                try:
                    # Extract features
                    df_new[['Month', 'Year']] = df_new['Date'].str.split('-', expand=True)[[1, 2]].astype(int)
                    df_new.drop(columns=['Date'], inplace=True)

                    df_new = df_new.reindex(columns=['Amount', 'Year', 'Month', 'Transaction_Type',
                                                     'Payment_Gateway', 'Transaction_State',
                                                     'Merchant_Category', 'fraud'])

                    # Preprocess all rows
                    X_data = []
                    for _, row in df_new.iterrows():
                        features = preprocess_inputs(row['Amount'], row['Year'], row['Month'],
                                                     row['Transaction_Type'], row['Payment_Gateway'],
                                                     row['Transaction_State'], row['Merchant_Category'])
                        X_data.append(features)

                    X = np.array(X_data)
                    y = df_new['fraud'].astype(int)

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Retrain model
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    model.fit(X_train, y_train)

                    # Evaluate
                    acc = model.score(X_test, y_test)
                    st.success(f"✅ Model retrained successfully with accuracy: **{acc:.2%}**")

                    # Save model
                    retrained_path = "retrained_model.pkl"
                    with open(retrained_path, 'wb') as f:
                        pickle.dump(model, f)
                    st.markdown(f"💾 New model saved to `{retrained_path}`")

                except Exception as e:
                    st.error(f"⚠️ Error during retraining: {e}")


elif tab == "📈 Dashboard":
    st.subheader("📈 Fraud Detection Summary")
    df = pd.read_csv("sample.csv")
    df['fraud'] = ["No", "Yes", "No", "No", "Yes"]
    fraud_counts = df['fraud'].value_counts().reset_index()
    fraud_counts.columns = ['Label', 'Count']

    chart = alt.Chart(fraud_counts).mark_bar().encode(
        x='Label',
        y='Count',
        color='Label'
    ).properties(title="Fraud vs Non-Fraud Count")

    st.altair_chart(chart, use_container_width=True)

elif tab == "🧠 Explainability":
    st.subheader("🧠 Model Explainability (SHAP)")
    try:
        input_data = preprocess_inputs(1500, 2024, 5, "Purchase", "PhonePe", "Karnataka", "Purchases")
        feature_names = (
            ["Amount", "Year", "Month"] +
            [f"TT_{x}" for x in tt] +
            [f"PG_{x}" for x in pg] +
            [f"TS_{x}" for x in ts] +
            [f"MC_{x}" for x in mc]
        )
        X_sample = pd.DataFrame([input_data], columns=feature_names)

        explainer = shap.Explainer(loaded_model)
        shap_values = explainer(X_sample)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be generated: {e}")

elif tab == "⚙️ Admin Panel":
    st.subheader("⚙️ Admin Panel – Retrain Model")
    st.info("This section allows admin to upload labeled data and retrain the model. (Simulation only)")
    new_data = st.file_uploader("Upload Labeled CSV (with fraud column)", type="csv")
    if new_data:
        df_new = pd.read_csv(new_data)
        st.write("Received data:")
        st.dataframe(df_new.head())
        if st.button("Simulate Retraining"):
            st.success("✅ Model retraining simulated successfully!")

