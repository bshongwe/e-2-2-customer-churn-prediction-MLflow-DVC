#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
from mlops_project.pipeline.prediction.prediction_pipeline import PredictionPipeline
import os

# Set page title and icon
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:")

# Title and description
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn.")

# Ensure model is loaded
@st.cache_resource
def load_model():
    """Load the prediction pipeline with cached model."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
    return PredictionPipeline(model_path=model_path)

# Load model
try:
    pipeline = load_model()
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input form for user data
with st.form(key='churn_prediction_form'):
    st.header("Customer Information")
    
    # Input fields with default values
    surname = st.text_input("Surname", value="Doe")
    credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=600)
    geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'], index=0)
    gender = st.selectbox("Gender", options=['Female', 'Male'], index=0)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=0.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_credit_card = st.checkbox("Has Credit Card", value=True)
    is_active_member = st.checkbox("Is Active Member", value=True)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Churn')

# Prediction logic
if submit_button:
    try:
        # Prepare data for prediction
        data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_credit_card else 0,
            'IsActiveMember': 1 if is_active_member else 0,
            'EstimatedSalary': estimated_salary
        }
        df = pd.DataFrame([data])

        # Predict
        prediction = pipeline.predict(df)
        probability = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline.model, 'predict_proba') else None

        # Display results
        st.header("Prediction Results")
        if prediction[0] == 0:
            st.success("Prediction: No Churn (Customer is likely to stay)")
        else:
            st.error("Prediction: Churn (Customer is likely to leave)")

        # Show probability if available
        if probability is not None:
            st.write(f"Churn Probability: {probability[0]:.2%}")

        # Show input data for reference
        st.subheader("Input Data")
        st.write(df)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Add footer
st.write("---")
st.write("Powered by Streamlit and your ML model")