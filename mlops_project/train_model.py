#!/usr/bin/env python3

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Generate some dummy data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Save the model to a .pkl file in the 'models' directory
model_path = 'models/model.pkl'
joblib.dump(rf, model_path)

print(f"Model saved at: {model_path}")