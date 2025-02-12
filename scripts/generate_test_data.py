#!/usr/bin/env python3

import pandas as pd
from sklearn.datasets import make_classification
import os

# Ensure the data directory exists
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Generate dummy data
# Adjust n_features to match your model's expected input (e.g., 10 features for simplicity)
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

# Create DataFrames
# Assuming your model expects specific feature names, you can rename columns accordingly
feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])  # Adjust to match number of features
y_df = pd.DataFrame({'Churn': y})

# Save to CSV files
X_df.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
y_df.to_csv(os.path.join(data_dir, 'test_labels.csv'), index=False)

print(f"Test data saved to {os.path.join(data_dir, 'test_data.csv')}")
print(f"Test labels saved to {os.path.join(data_dir, 'test_labels.csv')}")