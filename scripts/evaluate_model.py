import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def find_csv_files(directory, pattern):
    """
    Find CSV files in the given directory with the specified pattern.
    
    :param directory: Directory to search in
    :param pattern: Filename pattern to match
    :return: List of matching file paths
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern):
                matching_files.append(os.path.join(root, file))
    return matching_files

def load_test_data(directory='.'):
    # Look for test data and labels
    test_data_files = find_csv_files(directory, 'test_data.csv')
    test_label_files = find_csv_files(directory, 'test_labels.csv')
    
    if not test_data_files or not test_label_files:
        raise FileNotFoundError("Could not find required test data or label files.")
    
    # Assuming we only want the first match found
    X_test = pd.read_csv(test_data_files[0])
    y_test = pd.read_csv(test_label_files[0])['Churn'].values  # Assuming 'Churn' is your target variable
    
    return X_test, y_test

def load_model(directory='.'):
    # Look for model file
    model_files = find_csv_files(directory, '.pkl')
    
    if not model_files:
        raise FileNotFoundError("Model file not found.")
    
    # Load the first model file found
    return joblib.load(model_files[0])

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    model = load_model()
    evaluate_model(model, X_test, y_test)
