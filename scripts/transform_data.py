#!/usr/bin/env python3

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from the specified path."""
    try:
        data_file = list(data_path.glob("*.csv"))[0]
        df = pd.read_csv(data_file)
        logger.info(f"Data loaded from {data_file}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def transform_data(df: pd.DataFrame, output_path: Path) -> None:
    """Transform the data and save to output path."""
    try:
        # Define features for modeling (exclude identifiers and target)
        numeric_features = ['CreditScore', 'Age',
                            'Tenure', 'Balance',
                            'NumOfProducts', 'EstimatedSalary']
        categorical_features = ['Geography', 'Gender']
        target_column = 'Exited'

        # Ensure all required columns are present
        required_columns = numeric_features + categorical_features + [target_column, 'HasCrCard', 'IsActiveMember']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns for transformation: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit and transform features, keeping target separate
        X = df[numeric_features + categorical_features + ['HasCrCard', 'IsActiveMember']]
        y = df[target_column] if target_column in df.columns else None

        # Transform features
        X_transformed = preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_feature_names) + ['HasCrCard', 'IsActiveMember']

        # Create DataFrame with transformed features
        df_transformed = pd.DataFrame(X_transformed, columns=all_feature_names)

        # Add target back if it exists
        if y is not None:
            df_transformed[target_column] = y.values

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        output_file = output_path / 'transformed_data.csv'
        df_transformed.to_csv(output_file, index=False)
        logger.info(f"Transformed data saved to {output_file}")

    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        raise

if __name__ == "__main__":
    # Assume input is from artifacts/data_ingestion/
    # and output to artifacts/data_transformation/
    input_path = Path("artifacts/data_ingestion")
    output_path = Path("artifacts/data_transformation")

    if not input_path.exists():
        logger.error(f"Input data directory not found: {input_path}")
        sys.exit(1)

    try:
        df = load_data(input_path)
        transform_data(df, output_path)
        logger.info("Data transformation completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        sys.exit(1)