import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest_fraud_model.pkl")

st.title("Credit Card Fraud Detection")
st.write("Upload a CSV file to detect potentially fraudulent transactions.")

uploaded_file = st.file_uploader(
    "Upload transaction data",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Transaction Data")
    st.dataframe(data.head())

    # Make predictions
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= 0.34).astype(int)

    data["Fraud Probability"] = probabilities
    data["Prediction"] = predictions

    st.subheader("Predictions")
    st.dataframe(data)

