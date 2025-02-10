#!/usr/bin/env python3

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class PredictionPipeline(BaseEstimator):
    def __init__(self, model_path='../../models/model.pkl'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None

        # Define preprocessing steps for numeric and categorical data
        numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        categorical_features = ['Geography', 'Gender']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def load_model(self):
        """Load the model from a file."""
        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def fit(self, X, y=None):
        """Fit the preprocessor on the data. Note: This assumes model training happens elsewhere."""
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        """Transform the data using the preprocessor."""
        return self.preprocessor.transform(X)

    def predict(self, X):
        """Predict using the loaded model."""
        if self.model is None:
            self.load_model()

        # Ensure the input is a pandas DataFrame with expected columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        expected_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        if not all(col in X.columns for col in expected_columns):
            raise ValueError(f"DataFrame must contain columns: {expected_columns}")

        # Preprocess the data before prediction
        X_transformed = self.transform(X)

        # Use the model to predict
        return self.model.predict(X_transformed)

    def predict_proba(self, X):
        """Predict probabilities for binary classification."""
        if self.model is None:
            self.load_model()
        
        X_transformed = self.transform(X)
        return self.model.predict_proba(X_transformed) if hasattr(self.model, 'predict_proba') else None