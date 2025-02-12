import os
import sys
import mlflow
import dagshub
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from mlops_project.logger import logging
from mlops_project.exception import MyException
from mlops_project.entity.config_entity import ModelEvaluationConfig
from mlops_project.utils.common import save_json
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dagshub.init(repo_owner='KartikGarg20526', repo_name='End-to-End-Customer-Churn-Prediction-using-MLflow-and-DVC', mlflow=True)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        return accuracy, precision, recall, f1
    
    def log_into_mlflow(self):
        """
        Evaluating the model and logging all the metrics into mlflow

        Returns: None
        """
        try:
            logging.info("Reading test data")
            test_data = pd.read_csv(self.config.test_data_path)

            logging.info("Loading model")
            model = joblib.load(self.config.model_path)

            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[[self.config.target_column]]

            logging.info("Logging into mlflow")
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                
                logging.info("Calculating metrics")
                predictions = model.predict(test_x)

                accuracy, precision, recall, f1 = self.eval_metrics(test_y, predictions)

                scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

                logging.info("Saving metrics in json file")
                save_json(path=Path(self.config.metric_file_name), data=scores)

                logging.info("Logging params into mlflow")
                mlflow.log_params(self.config.all_params)

                logging.info("Logging metrics into mlflow")
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1-score", f1)

                # Model registry does not work with file store
                logging.info("Logging model into mlflow")
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise MyException(e, sys)