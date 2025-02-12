#!/usr/bin/env python3

import os
import sys
import pandas as pd
import yaml
from ensure import ensure_annotations
from box import ConfigBox
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """Read YAML file and return ConfigBox."""
    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {str(e)}")
        raise

@ensure_annotations
def validate_data(data_path: Path, schema_path: Path) -> bool:
    """Validate the dataset against the schema."""
    try:
        # Read schema
        schema = read_yaml(schema_path)
        logger.info(f"Schema loaded from {schema_path}")

        # Load data
        data_file = list(data_path.glob("*.csv"))[0]  # Assuming one CSV file in the directory
        df = pd.read_csv(data_file)
        logger.info(f"Data loaded from {data_file}, shape: {df.shape}")

        # Validate column names
        expected_columns = schema.columns
        if list(df.columns) != list(expected_columns):
            missing_cols = set(expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_columns)
            logger.error(f"Column mismatch: Missing {missing_cols}, Extra {extra_cols}")
            return False

        # Validate data types
        for col, dtype in schema.dtypes.items():
            if df[col].dtype != dtype:
                logger.error(f"Column {col} has type {df[col].dtype}, expected {dtype}")
                return False

        # Validate ranges and constraints (example for numeric columns)
        numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        for col in numeric_columns:
            if col in df.columns:
                if df[col].min() < schema.ranges.get(col, {}).get('min', float('-inf')) or \
                   df[col].max() > schema.ranges.get(col, {}).get('max', float('inf')):
                    logger.error(f"Column {col} out of range: min={df[col].min()}, max={df[col].max()}")
                    return False

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            # Optionally, fail if too many missing values
            if (missing_values / len(df) > 0.1).any():  # Example threshold: 10% missing
                logger.error("Too many missing values, validation failed")
                return False

        logger.info("Data validation successful")
        return True

    except Exception as e:
        logger.error(f"Error during data validation: {str(e)}")
        return False

if __name__ == "__main__":
    # Assume data is in artifacts/data_ingestion/ as per CI/CD
    data_path = Path("artifacts/data_ingestion")
    schema_path = Path("schema.yaml")

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        sys.exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        sys.exit(1)

    if validate_data(data_path, schema_path):
        logger.info("Data validation completed successfully")
        sys.exit(0)
    else:
        logger.error("Data validation failed")
        sys.exit(1)