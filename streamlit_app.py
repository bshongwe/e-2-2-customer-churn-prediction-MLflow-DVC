#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlops_project')))
import streamlit as st
import pandas as pd
import numpy as np
from mlops_project.pipeline.prediction import PredictionPipeline
# from mlops_pipeline.prediction.prediction_pipeline import PredictionPipeline  # Import from mlops_project, not src
import logging
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle command-line arguments for data type or use environment variable, matching app.py
if len(sys.argv) > 1 and sys.argv[1] == '--data-type':
    os.environ['DATA_TYPE'] = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('DATA_TYPE', 'dummy')
else:
    os.environ.setdefault('DATA_TYPE', 'dummy')  # Default to dummy if not set

logger.info(f"Starting Streamlit app with DATA_TYPE: {os.environ.get('DATA_TYPE', 'dummy')}")

# Set page title and icon
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:")

# Title and description
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn.")

# Ensure model is loaded
@st.cache_resource
def load_model():
    """Load the prediction pipeline with cached model."""
    try:
        # Resolve model path relative to project root
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / 'models' / 'model.pkl'
        logger.info(f"Loading model from: {model_path} with DATA_TYPE: {os.environ.get('DATA_TYPE', 'dummy')}")
        return PredictionPipeline(model_path=str(model_path))
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model
try:
    pipeline = load_model()
    st.write("Model loaded successfully.")
except Exception as e:
    st.error("Failed to load model. Please ensure model.pkl is in the models directory.")
    logger.error(f"Model loading failed: {str(e)}")
    st.stop()

# Input form for user data
with st.form(key='churn_prediction_form'):
    st.header("Customer Information")
    
    # Input fields with default values, matching app.py
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

# Prediction logic, matching app.py's VALIDATIONS and encoding
VALIDATIONS = {
    'creditScore': lambda x: 0 <= int(x) <= 850,
    'age': lambda x: 0 < int(x) <= 120,
    'tenure': lambda x: 0 <= int(x) <= 10,
}

if submit_button:
    try:
        # Prepare data for prediction, matching app.py
        data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': int(has_credit_card),
            'IsActiveMember': int(is_active_member),
            'EstimatedSalary': estimated_salary
        }
        df = pd.DataFrame([data])

        # Validate data, as in app.py
        for key, validation_func in VALIDATIONS.items():
            if key in data and not validation_func(data[key]):
                st.error(f"Invalid {key}. Please check your input.")
                st.stop()

        # Predict
        prediction = pipeline.predict(df)
        probability = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline.model, 'predict_proba') else None

        # Display results, matching app.py's output
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
        logger.error(f"Error during prediction: {str(e)}")
        st.error("An error occurred during prediction. Please try again or check the logs.")
        logger.error(traceback.format_exc())

# Add footer (ensure it's always displayed)
st.markdown("---")
st.write("Powered by Streamlit and your ML model")