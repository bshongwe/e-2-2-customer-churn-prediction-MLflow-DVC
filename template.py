import os
import logging
from pathlib import Path

project_name = "mlops_project"

list_of_files = [
    ".github/workflows/ci-cd.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "streamlit_app.py",  # Added for Streamlit app
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "scripts/evaluate_model.py",  # Added as part of pipeline
    "scripts/validate_data.py",  # Added as part of pipeline
    "scripts/transform_data.py",  # Added as part of pipeline
    "scripts/generate_test_data.py",  # Added as part of pipeline
    "data/",  # Added directory for data files
    "models/",  # Added directory for model files
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Handle directories (ends with /)
    if filename == "" and filepath.exists() and filepath.is_dir():
        logging.info(f"Directory {filepath} already exists")
        continue

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
