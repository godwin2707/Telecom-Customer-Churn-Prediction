import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import mysql.connector

# Loading the saved pickle file
with open("rf.pkl", "rb") as f:
    model = pickle.load(f)


with open("column_encoder.pkl", "rb") as f:
    ct = pickle.load(f)


with open("label_encoder.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Columns expected by the model
required_columns = list(ct.feature_names_in_)

# seeting the default values for the UI
default_values = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 0,
    'PhoneService': 'No',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 0.0,
    'TotalCharges': 0.0
}

# streamlit UI
st.title("Telecom Customer Churn Prediction")
st.write("Fill in the customer details to predict churn:")

# Use Yes/No for Senior Citizen in UI, instead of 1/0 
senior_input = st.selectbox("Senior Citizen", ['No', 'Yes'])

input_data = {
    'gender': st.selectbox("Gender", ['Male', 'Female']),
    'SeniorCitizen': 1 if senior_input == 'Yes' else 0,
    'Partner': st.selectbox("Partner", ['Yes', 'No']),
    'Dependents': st.selectbox("Dependents", ['Yes', 'No']),
    'tenure': st.number_input("Tenure (months)", min_value=0, max_value=100, value=0),
    'PhoneService': st.selectbox("Phone Service", ['Yes', 'No']),
    'MultipleLines': st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service']),
    'InternetService': st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No']),
    'OnlineSecurity': st.selectbox("Online Security", ['Yes', 'No', 'No internet service']),
    'OnlineBackup': st.selectbox("Online Backup", ['Yes', 'No', 'No internet service']),
    'DeviceProtection': st.selectbox("Device Protection", ['Yes', 'No', 'No internet service']),
    'TechSupport': st.selectbox("Tech Support", ['Yes', 'No', 'No internet service']),
    'StreamingTV': st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service']),
    'StreamingMovies': st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service']),
    'Contract': st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year']),
    'PaperlessBilling': st.selectbox("Paperless Billing", ['Yes', 'No']),
    'PaymentMethod': st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ]),
    'MonthlyCharges': st.number_input("Monthly Charges", min_value=0.0, value=0.0),
    'TotalCharges': st.number_input("Total Charges", min_value=0.0, value=0.0)
}

if st.button("Predict Churn"):
    df_input = pd.DataFrame([input_data])

    # Fill missing expected columns
    for col in required_columns:
        if col not in df_input.columns:
            df_input[col] = default_values[col]

    df_input = df_input[required_columns]
    df_input.fillna(value=default_values, inplace=True)

    # Applying column transformer
    X_transformed = ct.transform(df_input)

    # Predict using the model
    prediction = model.predict(X_transformed)[0]
    probability = model.predict_proba(X_transformed)[0][1]

    # Displaying results
    st.subheader("🔍 Prediction Result")
    st.success("Customer is likely to churn." if prediction == 1 else "Customer is not likely to churn.")
    st.write(f"Probability of churn: *{probability:.2%}*")

    # SQl Connection:
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="godwin07",  
            database="telco_churn"
        )
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS churn_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                gender VARCHAR(10),
                SeniorCitizen INT,
                Partner VARCHAR(5),
                Dependents VARCHAR(5),
                tenure INT,
                PhoneService VARCHAR(5),
                MultipleLines VARCHAR(20),
                InternetService VARCHAR(20),
                OnlineSecurity VARCHAR(20),
                OnlineBackup VARCHAR(20),
                DeviceProtection VARCHAR(20),
                TechSupport VARCHAR(20),
                StreamingTV VARCHAR(20),
                StreamingMovies VARCHAR(20),
                Contract VARCHAR(20),
                PaperlessBilling VARCHAR(5),
                PaymentMethod VARCHAR(30),
                MonthlyCharges FLOAT,
                TotalCharges FLOAT,
                Prediction INT,
                Probability FLOAT
            )
        """)

        # Insert data
        insert_query = """
            INSERT INTO churn_predictions (
                gender, SeniorCitizen, Partner, Dependents, tenure,
                PhoneService, MultipleLines, InternetService, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                MonthlyCharges, TotalCharges, Prediction, Probability
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = list(input_data.values()) + [int(prediction), float(probability)]
        cursor.execute(insert_query, values)
        conn.commit()
        cursor.close()
        conn.close()
        st.success("Prediction saved to database successfully!")

    except mysql.connector.Error as e:
        st.error(f"MySQL error: {e}")
