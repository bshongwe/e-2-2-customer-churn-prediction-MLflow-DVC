#!/usr/bin/env python3

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class PredictionPipeline(BaseEstimator):
    def __init__(self, model_path=None):
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'src/mlops_project/models/model.pkl')
        self.model_path = model_path
        self.model = None
        # List of numeric features:
        # CreditScore: Credit score of the customer
        # Age: Age of the customer
        # Tenure: Number of years the customer has been with the bank
        # categorical_features: 
        # 'Geography' - the country of the customer
        # 'Gender' - the gender of the customer
        categorical_features = ['Geography', 'Gender']
        # NumOfProducts: Number of products the customer has with the bank
        # EstimatedSalary: Estimated salary of the customer
        numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

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
    def transform(self, X):
        """
        Transform the data using the preprocessor.

        Parameters:
        X (pd.DataFrame): DataFrame containing the features to be transformed. 
                          Expected columns are ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                          'Balance', 'NumOfProducts', 'EstimatedSalary'].

        Returns:
        np.ndarray: Transformed feature array.
        """
        return self.preprocessor.transform(X)
    def predict(self, X):
        """
        Predict using the loaded model.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to predict, must contain the following columns:
            ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

        Returns
        -------
        numpy.ndarray
            The predicted values.
        """
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
        """
        Predict probabilities for binary classification.

        Parameters:
        X (pd.DataFrame): DataFrame containing the features to be transformed. 
                          Expected columns are ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                          'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'].

        Returns:
        np.ndarray: Predicted probabilities.
        """
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

        # Use the model to predict probabilities
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_transformed)
        else:
            raise AttributeError("The loaded model does not support probability prediction (predict_proba).")


    def predict_proba(self, X):
        """Predict probabilities for binary classification."""
        if self.model is None:
            self.load_model()
        
        X_transformed = self.transform(X)
        return self.model.predict_proba(X_transformed) if hasattr(self.model, 'predict_proba') else None