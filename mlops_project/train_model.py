#!/usr/bin/env python3

# Standard Library
import os
import sys
import argparse

# Third-Party Libraries
import numpy as np
import pandas as pd
import joblib

# Scikit-Learn Modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_dummy_data():
    """Generate and return dummy data for model training."""
    print("Generating dummy data...")
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    print("Dummy data generated: X shape:", X.shape, "y shape:", y.shape)
    return X, y

def find_csv_file(directory, pattern='.csv'):
    """Find the first CSV file in the given directory matching the pattern."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern):
                return os.path.join(root, file)
    return None

def load_real_data(data_path, target_column):
    """Load real data from a CSV file or directory."""
    try:
        print(f"Checking if {data_path} is a directory or file...")
        if os.path.isdir(data_path):
            print(f"Searching for CSV file in directory: {data_path}")
            csv_file = find_csv_file(data_path)
            if not csv_file:
                raise FileNotFoundError(f"No CSV file found in directory: {data_path}. Contents: {os.listdir(data_path)}")
            data_path = csv_file
            print(f"Found CSV file: {csv_file}")
        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"File or directory not found at: {data_path}")
            print(f"Loading file directly: {data_path}")

        print(f"Loading real data from {data_path}...")
        data = pd.read_csv(data_path)
        print("Data loaded. Shape:", data.shape)
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        raise

def train_model(X, y, model_path):
    """Train a Random Forest Classifier and save it."""
    try:
        print("Starting model training...")
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split: Train shape:", X_train.shape, "Test shape:", X_test.shape)

        # Train the model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        print("Model training completed.")

        # Evaluate model on test data
        accuracy = rf.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")

        # Validate model path
        if not model_path:
            raise ValueError("Model path cannot be empty.")
        
        # Ensure the models directory exists
        model_dir = os.path.dirname(model_path) or '.'
        os.makedirs(model_dir, exist_ok=True)
        print(f"Ensuring directory exists for model path: {model_path}")

        # Save the model
        joblib.dump(rf, model_path)
        print(f"Model saved at: {model_path}")

        # Verify the saved model
        model_size = os.path.getsize(model_path)
        print(f"Model file size: {model_size} bytes")
        if model_size < 1024:  # Arbitrary threshold, adjust as needed
            print("Error: Model file size is unusually small, it might be corrupted.", file=sys.stderr)
            raise ValueError("Model file is empty or corrupted.")

        # Verify model can be loaded
        try:
            loaded_model = joblib.load(model_path)
            print("Model verification: Successfully loaded the saved model.")
        except Exception as e:
            print(f"Error verifying saved model: {e}", file=sys.stderr)
            raise
    except Exception as e:
        print(f"Error training or saving model: {e}", file=sys.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest model with either dummy or real data, dynamically locating CSV files.")
    parser.add_argument('--use_dummy', action='store_true', help="Use dummy data for training")
    parser.add_argument('--data_path', type=str, help="Path to the CSV file or directory containing real data")
    parser.add_argument('--target_column', type=str, default='Churn', help="Name of the target column in the real dataset")
    parser.add_argument('--model_path', type=str, default='models/model.pkl', help="Path to save the trained model")

    args = parser.parse_args()

    # Check environment variable for data type if no args specified
    if not args.use_dummy and not args.data_path:
        data_type = os.environ.get('DATA_TYPE', 'dummy').lower()
        if data_type == 'dummy':
            args.use_dummy = True
        elif data_type == 'real':
            args.data_path = 'artifacts/data_transformation'  # Default real data path, adjust as needed
            args.target_column = 'Churn'
        else:
            print("Invalid DATA_TYPE environment variable. Use 'dummy' or 'real'.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    # Check if at least one data source is specified
    if not args.use_dummy and not args.data_path:
        print("Please specify at least one data source: --use_dummy or --data_path, or set DATA_TYPE environment variable.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Ensure only one data source is used
    if args.use_dummy and args.data_path:
        print("Error: Cannot use both --use_dummy and --data_path. Choose one data source.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.use_dummy:
        X, y = generate_dummy_data()
        train_model(X, y, args.model_path)
    elif args.data_path:
        if not args.target_column:
            print("Error: --target_column is required when using --data_path.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        X, y = load_real_data(args.data_path, args.target_column)
        train_model(X, y, args.model_path)

if __name__ == "__main__":
    main()
