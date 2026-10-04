import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/random_forest_fraud_model.pkl")

st.title("Credit Card Fraud Detection")

st.write("Upload transaction data to predict fraudulent transactions.")

st.header("Transaction Data")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded data:")
    st.dataframe(data)

    if st.button("Predict"):
        X = data.drop(columns=["Class"], errors="ignore")

        predictions = model.predict(X)

        data["Prediction"] = predictions

        data["Prediction"] = data["Prediction"].map({
    0: "Normal",
    1: "Fraud"
})

        fraud_count = (predictions == 1).sum()
        total_transactions = len(predictions)
        fraud_percentage = (fraud_count / total_transactions) * 100

        st.metric("Fraudulent Transactions", fraud_count)
        st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
        st.write("Prediction Results:")
        st.dataframe(data)