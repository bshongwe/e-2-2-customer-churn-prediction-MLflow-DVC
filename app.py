#!/usr/bin/env python3

import os
import subprocess
import logging
import nbformat
from nbconvert import PythonExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_and_run_notebook(notebook_path):
    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert to Python script
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    
    # Write the Python script to a temporary file
    temp_py_file = 'temp_script_from_notebook.py'
    with open(temp_py_file, 'w') as f:
        f.write(source)
    
    try:
        # Execute the converted script
        logging.info(f"Running {notebook_path}")
        subprocess.run(["python", temp_py_file], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {notebook_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_py_file):
            os.remove(temp_py_file)

def run_notebooks_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(directory, filename)
            convert_and_run_notebook(filepath)

if __name__ == "__main__":
    research_directory = os.path.join(os.path.dirname(__file__), "research")
    if os.path.exists(research_directory) and os.path.isdir(research_directory):
        run_notebooks_in_directory(research_directory)
    else:
        logging.error(f"Directory {research_directory} does not exist or is not a directory")