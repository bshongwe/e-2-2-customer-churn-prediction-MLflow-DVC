#!/usr/bin/env python3

import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class PredictionPipeline(BaseEstimator):
    def __init__(self, model_path=None):
        """
        Initialize the PredictionPipeline with optional model path.

        Parameters:
        -----------
        model_path : str, optional
            Path to the saved model file. If None, defaults to 'models/model.pkl' relative to the project root.
        """
        if model_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            model_path = os.path.join(base_dir, 'models', 'model.pkl')
        self.model_path = model_path
        self.model = None

        # Define features
        self.numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        self.categorical_features = ['Geography', 'Gender']

        # Define preprocessing steps
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
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def load_model(self):
        """Load the model from a file."""
        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def fit(self, X, y=None):
        """
        Fit the preprocessor on the data. Note: This assumes model training happens elsewhere.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data for fitting the preprocessor.
        y : None, optional
            Ignored, for compatibility with scikit-learn.

        Returns:
        --------
        self : PredictionPipeline
            Returns self.
        """
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        """
        Transform the data using the preprocessor.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the features to be transformed. Expected columns are:
            ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
             'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'].

        Returns:
        --------
        np.ndarray
            Transformed feature array.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        expected_columns = self.numeric_features + self.categorical_features + ['HasCrCard', 'IsActiveMember']
        if not all(col in X.columns for col in expected_columns):
            raise ValueError(f"DataFrame must contain columns: {expected_columns}")
        
        return self.preprocessor.transform(X)

    def predict(self, X):
        """
        Predict using the loaded model.

        Parameters:
        -----------
        X : pd.DataFrame
            The input data to predict, must contain the following columns:
            ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
             'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

        Returns:
        --------
        np.ndarray
            The predicted values.
        """
        if self.model is None:
            self.load_model()

        X_transformed = self.transform(X)
        return self.model.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict probabilities for binary classification.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing the features to be transformed. Expected columns are:
            ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
             'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'].

        Returns:
        --------
        np.ndarray
            Predicted probabilities.
        """
        if self.model is None:
            self.load_model()

        X_transformed = self.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_transformed)
        else:
            raise AttributeError("The loaded model does not support probability prediction (predict_proba).")