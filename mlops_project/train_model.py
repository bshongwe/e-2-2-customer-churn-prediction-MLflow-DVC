#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib
import argparse
import os
import sys

def generate_dummy_data():
    """Generate and return dummy data for model training."""
    print("Generating dummy data...")
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    print("Dummy data generated: X shape:", X.shape, "y shape:", y.shape)
    return X, y

def load_real_data(data_path, target_column):
    """Load real data from a CSV file."""
    try:
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

def train_model(X, y, model_path='models/model.pkl'):
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

        # Ensure the models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Ensuring directory exists for model path: {model_path}")

        # Save the model
        joblib.dump(rf, model_path)
        print(f"Model saved at: {model_path}")

        # Verify the saved model
        model_size = os.path.getsize(model_path)
        print(f"Model file size: {model_size} bytes")
        if model_size < 1024:  # Arbitrary threshold, adjust as needed
            print("Warning: Model file size is unusually small, it might be corrupted.", file=sys.stderr)
        
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
    parser = argparse.ArgumentParser(description="Train a Random Forest model with either dummy or real data.")
    parser.add_argument('--use_dummy', action='store_true', help="Use dummy data for training")
    parser.add_argument('--data_path', type=str, help="Path to the CSV file containing real data")
    parser.add_argument('--target_column', type=str, default='Churn', help="Name of the target column in the real dataset")
    parser.add_argument('--model_path', type=str, default='models/model.pkl', help="Path to save the trained model")

    args = parser.parse_args()

    if args.use_dummy:
        X, y = generate_dummy_data()
        train_model(X, y, args.model_path)
    elif args.data_path:
        X, y = load_real_data(args.data_path, args.target_column)
        train_model(X, y, args.model_path)
    else:
        print("Please specify either --use_dummy or provide --data_path for real data.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()