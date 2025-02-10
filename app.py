from flask import Flask, render_template, request, jsonify
import sys
import os
import subprocess
import logging
import nbformat
from nbconvert import PythonExporter
import numpy as np
import pandas as pd
from mlops_project.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_and_run_notebook(notebook_path):
    # Read notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert to Python script
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    
    # Write Python script to temporary file
    temp_py_file = 'temp_script_from_notebook.py'
    with open(temp_py_file, 'w') as f:
        f.write(source)
    
    try:
        # Execute converted script
        logging.info(f"Running {notebook_path}")
        subprocess.run(["python", temp_py_file], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {notebook_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_py_file):
            os.remove(temp_py_file)

def run_notebooks_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(directory, filename)
            convert_and_run_notebook(filepath)

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
    research_directory = os.path.join(os.path.dirname(__file__), "research")
    if os.path.exists(research_directory) and os.path.isdir(research_directory):
        run_notebooks_in_directory(research_directory)
        return jsonify({"message": "Training Successful!"})
    else:
        logging.error(f"Directory {research_directory} does not exist or is not a directory")
        return jsonify({"error": "Training failed: research directory not found."}), 500

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
            data['creditCard'] = bool(int(data['creditCard']))  # Convert to boolean for clarity
            data['activeMember'] = bool(int(data['activeMember']))  # Convert to boolean for clarity
            data['estimatedSalary'] = float(data['estimatedSalary'])

            # Map categorical data to numerical or encoded values
            geography_encoding = {'France': 0, 'Spain': 1, 'Germany': 2}  # Example mapping, adjust as needed
            data['geography'] = geography_encoding.get(data['geography'], -1)  # -1 for unknown values
            gender_encoding = {'Female': 0, 'Male': 1}  # Example mapping, adjust as needed
            data['gender'] = gender_encoding.get(data['gender'], -1)  # -1 for unknown values

            # Prepare data for prediction
            field_names = [
                'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
            ]
            matrix = np.array([
                data['creditScore'], data['geography'], data['gender'], data['age'], data['tenure'],
                data['balance'], data['numberOfProducts'], int(data['creditCard']), int(data['activeMember']), data['estimatedSalary']
            ]).reshape(1, -1)
            df = pd.DataFrame(matrix, columns=field_names)

            # Predict
            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(df)

            result = 'No' if prediction[0] == 0 else 'Yes'
            return render_template('result.html', prediction=result)
        except Exception as e:
            # Log the error
            logging.error(f"Error occurred: {str(e)}")
            return jsonify({"error": "An error occurred while processing your request."}), 500

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=bool(os.environ.get('DEBUG', 'False')))