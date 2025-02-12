import os
import sys
import joblib
import pandas as pd
from mlops_project.logger import logging
from mlops_project.exception import MyException
from mlops_project.entity.config_entity import ModelTrainerConfig
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        """
        Training and saving the model
        
        Returns: None
        """
        try:
            logging.info("Reading train and test data")
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            logging.info("Splitting the dataset into features and target")
            X_train = train_data.drop([self.config.target_column], axis=1)
            X_test = test_data.drop([self.config.target_column], axis=1)
            y_train = train_data[[self.config.target_column]]
            y_test = test_data[[self.config.target_column]]

            logging.info("Training the model")
            rfc = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                bootstrap=self.config.bootstrap
            )

            rfc.fit(X_train, y_train)

            logging.info("Saving the model")
            joblib.dump(rfc, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            raise MyException(e, sys)