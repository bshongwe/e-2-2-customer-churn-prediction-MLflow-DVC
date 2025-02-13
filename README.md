![Visits Badge](https://badges.pufler.dev/visits/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub Repo Stars](https://img.shields.io/github/stars/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub Forks](https://img.shields.io/github/forks/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub Issues](https://img.shields.io/github/issues/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub License](https://img.shields.io/github/license/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)

# 🚀 End-to-End Customer Churn Prediction using MLflow & DVC  

An end-to-end machine learning project for predicting **customer churn**, leveraging **MLflow** for experiment tracking, **DVC** for data versioning, and implementing a robust **CI/CD pipeline**. This project follows a modular approach, covering **data ingestion, validation, transformation, model training, evaluation**, and **deployment**.

---

## 📁 Project Structure  

The project is organized as follows:

- **`artifacts/`** - Root directory for all artifacts  
  - **`data_ingestion/`** - Stores downloaded and unzipped data  
  - **`data_validation/`** - Contains data validation results  
  - **`data_transformation/`** - Holds transformed data  
  - **`model_trainer/`** - Includes trained models  
  - **`model_evaluation/`** - Stores evaluation metrics  

- **`data/`** - Source data directory  
- **`models/`** - Directory for saved models  
- **`research/`** - Jupyter notebooks for exploratory data analysis and experiments  
- **`scripts/`** - Python scripts for data processing and model operations  
- **`mlops_project/`** - Modularized code structure  
  - **`pipeline/`** - Contains prediction and other ML pipeline components  
  - **`utils/`** - Utility functions  
  - **`config/`** - Configuration files  
  - **`entity/`** - Entity definitions for configuration  

- **`app.py`** - Flask application for model prediction and training  
- **`streamlit_app.py`** - Streamlit application for interactive predictions  
- **`requirements.txt`** - Project dependencies  
- **`Dockerfile`** - Docker configuration for containerization  
- **`.github/workflows/`** - GitHub Actions workflows for CI/CD

---
<br></br>
## 🚀 CI/CD Pipeline Overview 🚀🎯

This project leverages **GitHub Actions** for an automated, end-to-end **ML workflow**, ensuring efficient and reliable execution of each stage:

| 🔧 **Job**               | 📌 **Description** |
|-------------------------|--------------------|
| 📥 **Data Ingestion**   | Downloads the dataset from a secure location and extracts it for further processing. |
| ✅ **Data Validation**  | Validates data against a schema to ensure integrity and correctness. |
| 🔄 **Data Transformation** | Applies necessary transformations (encoding, scaling) to prepare data for training. |
| 🤖 **Model Training**   | Trains a Random Forest model using both dummy and real data. |
| 📊 **Model Evaluation** | Evaluates the trained model's performance on a test set. |
| 🚀 **Deploy**           | Builds and pushes a Docker image to GitHub Container Registry, then deploys via SSH. |

---

<br></br>
## 🚀 Getting Started  

Follow these steps to set up and run the project:

<br></br>
### 🔧 Installation  

```bash
# Clone the repository  
git clone https://github.com/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC.git
```

<br></br>
### ▶️📥 Navigate to project directory  

```bash
cd e-2-2-customer-churn-prediction-MLflow-DVC
```

<br></br>
### 🚀 Install dependencies 🔧

```bash
pip install -r requirements.txt
```

<br></br>
### ▶️ Running the Pipeline
To execute the pipeline:

```bash
# Run the ML pipeline using DVC
dvc repro
```

## 🤖 Training The Model

### 🤖📜 Train the model with dummy data

```bash
python train_model.py --use_dummy
```

### 🤖📜 Or with real data (ensure data is transformed first)

```bash
python train_model.py --data_path artifacts/data_transformation/transformed_data.csv --target_column Exited
```

### 🌐 Start Flask server
```bash
python app.py
```

## 🚀 Run Streamlit app for interactive prediction

```bash
streamlit run streamlit_app.py
```

<br></br>
## 🧪 Running Tests

Ensure your code works as expected by running:

```bash
pytest
```

<br></br>
## 🐳 Deployment

    Flask App: Deployed using Docker and SSH via GitHub Actions. Automatically triggers training on deployment through a /train endpoint.
    Streamlit App: Deployed to Streamlit Community Cloud via GitHub Actions, allowing for interactive model predictions.


<br></br>
# 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

<br></br>
# 🙌 Acknowledgements
Special thanks to the following tools that made this project possible:  

    🧪 MLflow - Experiment tracking  
    📦 DVC - Data versioning  
    🚀 GitHub Actions - CI/CD automation  
    🐳 Docker - Containerization  
    🖥️ Streamlit - Interactive UI for ML applications


Happy coding! 🚀
