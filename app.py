#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
import os
import subprocess
import logging
import nbformat
from nbconvert import PythonExporter
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlops_project')))
from mlops_project.pipeline.prediction import PredictionPipeline

# Handle command-line arguments for data type or use environment variable
if len(sys.argv) > 1 and sys.argv[1] == '--data-type':
    os.environ['DATA_TYPE'] = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('DATA_TYPE', 'dummy')
else:
    os.environ.setdefault('DATA_TYPE', 'dummy')  # Default to dummy if not set

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Starting app with DATA_TYPE: {os.environ.get('DATA_TYPE', 'dummy')}")

def convert_and_run_notebook(notebook_path):
    """Convert and run a Jupyter notebook."""
    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        exporter = PythonExporter()
        source, meta = exporter.from_notebook_node(nb)
        
        temp_py_file = 'temp_script_from_notebook.py'
        with open(temp_py_file, 'w') as f:
            f.write(source)
        
        logger.info(f"Running {notebook_path} with DATA_TYPE: {os.environ.get('DATA_TYPE', 'dummy')}")
        result = subprocess.run(["python", temp_py_file], check=True, capture_output=True, text=True, env=os.environ)
        logger.info(f"Notebook execution output: {result.stdout}")
        
        if os.path.exists(temp_py_file):
            os.remove(temp_py_file)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {notebook_path}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running notebook: {str(e)}")
        return False

def run_notebooks_in_directory(directory):
    """Run all notebooks in the specified directory."""
    success = True
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(directory, filename)
            if not convert_and_run_notebook(filepath):
                success = False
    return success

# Validation for prediction inputs
VALIDATIONS = {
    'creditScore': lambda x: 0 <= int(x) <= 850,
    'age': lambda x: 0 < int(x) <= 120,
    'tenure': lambda x: 0 <= int(x) <= 10,
}

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    """Route to train the pipeline by running notebooks."""
    research_directory = os.path.join(os.path.dirname(__file__), "research")
    if not os.path.exists(research_directory) or not os.path.isdir(research_directory):
        logger.error(f"Directory {research_directory} does not exist or is not a directory")
        return jsonify({"error": "Training failed: research directory not found."}), 500

    try:
        with app.app_context():
            logger.info(f"Starting training in directory: {research_directory} with DATA_TYPE: {os.environ.get('DATA_TYPE', 'dummy')}")
            success = run_notebooks_in_directory(research_directory)
            if success:
                logger.info("Training completed successfully.")
                return jsonify({"message": "Training Successful!"})
            else:
                logger.error("Training failed: one or more notebooks failed to execute.")
                return jsonify({"error": "Training failed: one or more notebooks failed to execute."}), 500
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during training."}), 500

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Collecting data from form
            data = {
                'Surname': request.form['Surname'],
                'creditScore': request.form['creditScore'],
                'geography': request.form['geography'],
                'gender': request.form['gender'],
                'age': request.form['age'],
                'tenure': request.form['tenure'],
                'balance': request.form['balance'],
                'numberOfProducts': request.form['numberOfProducts'],
                'creditCard': request.form['creditCard'],
                'activeMember': request.form['activeMember'],
                'estimatedSalary': request.form['estimatedSalary']
            }

            # Validate data
            for key, validation_func in VALIDATIONS.items():
                if not validation_func(data[key]):
                    return jsonify({"error": f"Invalid {key}. Please check your input."}), 400

            # Convert data types
            data['creditScore'] = int(data['creditScore'])
            data['age'] = int(data['age'])
            data['tenure'] = int(data['tenure'])
            data['balance'] = float(data['balance'])
            data['numberOfProducts'] = int(data['numberOfProducts'])
            data['creditCard'] = bool(int(data['creditCard']))
            data['activeMember'] = bool(int(data['activeMember']))
            data['estimatedSalary'] = float(data['estimatedSalary'])

            # Map categorical data to numerical or encoded values
            geography_encoding = {'France': 0, 'Spain': 1, 'Germany': 2}
            data['geography'] = geography_encoding.get(data['geography'], -1)
            gender_encoding = {'Female': 0, 'Male': 1}
            data['gender'] = gender_encoding.get(data['gender'], -1)

            # Prepare data for prediction
            field_names = [
                'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
            ]
            matrix = np.array([
                data['creditScore'],
                data['geography'],
                data['gender'],
                data['age'],
                data['tenure'],
                data['balance'],
                data['numberOfProducts'],
                int(data['creditCard']),
                int(data['activeMember']),
                data['estimatedSalary']
            ]).reshape(1, -1)
            df = pd.DataFrame(matrix, columns=field_names)

            # Predict
            try:
                prediction_pipeline = PredictionPipeline()
                prediction = prediction_pipeline.predict(df)
            except Exception as e:
                logger.error(f"Error loading or predicting with model: {str(e)}")
                return jsonify({"error": "Failed to load model for prediction. Please ensure training is complete."}), 500

            result = 'No' if prediction[0] == 0 else 'Yes'
            return render_template('result.html', prediction=result)
        except Exception as e:
            # Log the error
            logger.error(f"Error occurred: {str(e)}")
            return jsonify({"error": "An error occurred while processing your request."}), 500

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=bool(os.environ.get('DEBUG', 'False')))