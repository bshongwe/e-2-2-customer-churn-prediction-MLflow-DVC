![Visits Badge](https://badges.pufler.dev/visits/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub Repo Stars](https://img.shields.io/github/stars/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub Forks](https://img.shields.io/github/forks/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub Issues](https://img.shields.io/github/issues/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)
![GitHub License](https://img.shields.io/github/license/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC)

# 🚀 End-to-End Customer Churn Prediction using MLflow & DVC  

An end-to-end machine learning project for predicting **customer churn**, utilizing **MLflow** for experiment tracking and **DVC** for data versioning. This project follows a modular approach, covering **data ingestion, validation, transformation, model training, and evaluation**.

---

## 📁 Project Structure  

The project is organized as follows:  

- **`artifacts_root/`** - Root directory for all artifacts  
- **`data_ingestion/`**  
  - 📂 `artifacts/data_ingestion/` - Directory for data ingestion artifacts  
  - 📄 `source_URL` - URL for downloading data  
  - 📄 `local_data_file` - Local path for storing downloaded data  
  - 📂 `unzip_dir/` - Directory for extracted data  
- **`data_validation/`**  
  - 📂 `artifacts/data_validation/` - Directory for validation artifacts  
  - 📂 `unzip_data_dir/` - Path to extracted data  
  - 📄 `STATUS_FILE` - Stores validation results  
- **`data_transformation/`**  
  - 📂 `artifacts/data_transformation/` - Directory for transformed data  
  - 📄 `data_path` - Path to transformed data  
  - 📄 `preprocessor.pkl` - Serialized preprocessor file  
- **`model_trainer/`**  
  - 📂 `artifacts/model_trainer/` - Directory for training artifacts  
  - 📄 `train_data_path` - Path to training dataset  
  - 📄 `test_data_path` - Path to test dataset  
  - 📄 `model.pkl` - Trained model file  
- **`model_evaluation/`**  
  - 📂 `artifacts/model_evaluation/` - Directory for evaluation artifacts  
  - 📄 `test_data_path` - Path to test dataset  
  - 📄 `model_path` - Path to trained model  
  - 📄 `metrics.json` - File storing model performance metrics  

---

## ⚡ CI/CD Pipeline  

This project includes a **CI/CD pipeline** powered by **GitHub Actions**, automating key ML lifecycle steps:

1. **📥 Data Ingestion** - Downloads and extracts dataset  
2. **✅ Data Validation** - Checks dataset integrity  
3. **🔄 Data Transformation** - Preprocesses and transforms data  
4. **📊 Model Training** - Trains a machine learning model  
5. **📈 Model Evaluation** - Assesses model performance  

---

## 🚀 Getting Started  

Follow these steps to set up and run the project:

### 🔧 Installation  

```bash
# Clone the repository  
git clone https://github.com/bshongwe/e-2-2-customer-churn-prediction-MLflow-DVC.git

# Navigate to project directory  
cd e-2-2-customer-churn-prediction-MLflow-DVC

# Install dependencies  
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline  

To execute the pipeline, use **DVC**:

```bash
# Run the full pipeline  
dvc repro
```

---

## 📜 License  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🙌 Acknowledgements  

Special thanks to the following tools that made this project possible:  

- 🧪 [MLflow](https://mlflow.org/) - Experiment tracking  
- 📦 [DVC](https://dvc.org/) - Data versioning  
```

Happy coding! 🚀