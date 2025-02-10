#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib
import argparse

def generate_dummy_data():
    """Generate and return dummy data for model training."""
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    return X, y

def load_real_data(data_path, target_column):
    """Load real data from a CSV file."""
    data = pd.read_csv(data_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

def train_model(X, y, model_path='models/model.pkl'):
    """Train a Random Forest Classifier and save it."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate model on test data
    accuracy = rf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the model
    joblib.dump(rf, model_path)
    print(f"Model saved at: {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest model with either dummy or real data.")
    parser.add_argument('--use_dummy', action='store_true', help="Use dummy data for training")
    parser.add_argument('--data_path', type=str, help="Path to the CSV file containing real data")
    parser.add_argument('--target_column', type=str, default='target', help="Name of the target column in the real dataset")
    parser.add_argument('--model_path', type=str, default='models/model.pkl', help="Path to save the trained model")

    args = parser.parse_args()

    if args.use_dummy:
        X, y = generate_dummy_data()
        train_model(X, y, args.model_path)
    elif args.data_path:
        X, y = load_real_data(args.data_path, args.target_column)
        train_model(X, y, args.model_path)
    else:
        print("Please specify either --use_dummy or provide --data_path for real data.")
        parser.print_help()

if __name__ == "__main__":
    main()