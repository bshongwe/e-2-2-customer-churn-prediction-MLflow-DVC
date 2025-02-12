import os
import sys
import joblib
import pandas as pd
import numpy as np
from mlops_project.logger import logging
from mlops_project.exception import MyException
from mlops_project.entity.config_entity import DataTransformationConfig
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the rows with missing values in the dataset
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            pd.DataFrame: Data without missing values
        """
        logging.info("Dropping the rows with missing values in the dataset")
        return data.dropna()
    
    def _drop_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the duplicate rows in the dataset
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            pd.DataFrame: Data without duplicate rows
        """
        logging.info("Dropping the duplicate rows in the dataset")
        return data.drop_duplicates()

    def _split_features_and_target(self, data: pd.DataFrame) -> tuple:
        """
        Splits the dataset into features and target
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            pd.DataFrame: Features
            pd.DataFrame: Target
        """
        logging.info("Splitting the dataset into features and target")
        X, y = data.drop(columns=[self.config.target_column], axis=1), data[self.config.target_column]
        return X, y
    
    def _drop_irrelevant_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the irrelevant columns in the dataset
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            pd.DataFrame: Data without irrelevant columns
        """
        logging.info("Dropping the irrelevant columns in the dataset")
        return data.drop(['RowNumber','CustomerId', 'Surname'], axis=1)
    
    def _convert_column_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert specified columns to appropriate data types.

        Parameters:
        data (pd.DataFrame): The input data.

        Returns:
        pd.DataFrame: The data with converted column types.
        """
        logging.info("Converting column types")
        columns_to_convert = ['HasCrCard', 'IsActiveMember', 'Age']
        for column in columns_to_convert:
            data[column] = data[column].astype('int')
        return data
    
    def _map_gender_column(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map the 'Gender' column to numeric values.

        Parameters:
        X (pd.DataFrame): The feature data.

        Returns:
        pd.DataFrame: The feature data with 'Gender' column mapped.
        """
        logging.info("Mapping 'Gender' column to numeric values")
        mapping = {'Male': 0, 'Female': 1}
        X['Gender'] = X['Gender'].map(mapping).astype(int)
        return X
    
    def _select_columns_by_type(self, X: pd.DataFrame) -> tuple:
        """
        Select numerical and categorical columns from the feature data.

        Parameters:
        X (pd.DataFrame): The feature data.

        Returns:
        tuple: Lists of numerical and categorical column names.
        """
        logging.info("Selecting numerical and categorical columns")
        num_cols = X.select_dtypes(include=np.number).columns.to_list()
        cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()
        return num_cols, cat_cols
    
    def _create_transformer(self, num_cols: list, cat_cols: list) -> ColumnTransformer:
        """
        Create a column transformer for preprocessing.

        Parameters:
        num_cols (list): List of numerical column names.
        cat_cols (list): List of categorical column names.

        Returns:
        ColumnTransformer: The column transformer.
        """
        logging.info("Creating column transformer")
        num_pipeline = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('one_hot_enc', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        transformer = ColumnTransformer(transformers=[
            ('num_pipeline', num_pipeline, num_cols),
            ('cat_pipeline', cat_pipeline, cat_cols),
            ],remainder='passthrough',
            n_jobs=-1
        )

        return transformer
    
    def _save_transformer(self, transformer: ColumnTransformer) -> None:
        """
        Save the fitted transformer to a file.

        Parameters:
        transformer (ColumnTransformer): The fitted column transformer.
        """
        logging.info("Saving transformer")
        joblib.dump(transformer, os.path.join(self.config.root_dir, self.config.preprocessor_name))

    def _train_test_split(self, X: pd.DataFrame, y:pd.DataFrame) -> tuple:
        """
        Splits the dataset into training and testing sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.DataFrame): Target variable.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logging.info("Splitting the dataset into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return (X_train, X_test, y_train, y_test)
    
    def preprocess_data(self) -> None:
        """
        Preprocess the data by handling missing values, removing duplicates, converting data types, 
        dropping irrelevant columns, splitting data into train and test and applying transformations.

        Parameters:
        data (pd.DataFrame): The raw input data.

        Returns: None
        """
        try:
            data = pd.read_csv(self.config.data_path)    

            data = self._handle_missing_values(data)
            data = self._drop_duplicates(data)
            data = self._drop_irrelevant_columns(data)
            data = self._convert_column_types(data)

            X,y = self._split_features_and_target(data)
            X = self._map_gender_column(X)

            num_cols, cat_cols = self._select_columns_by_type(X)

            X_train, X_test, y_train, y_test = self._train_test_split(X,y)

            transformer = self._create_transformer(num_cols, cat_cols)

            logging.info("Applying transformations")
            X_train_transformed = transformer.fit_transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            feature_names = transformer.get_feature_names_out()

            X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

            logging.info("Saving transformer")
            self._save_transformer(transformer)

            y_train_df = y_train.to_frame().reset_index(drop=True)
            y_test_df = y_test.to_frame().reset_index(drop=True)

            logging.info("Concatenating dataframes")
            train_processed = pd.concat([X_train_transformed_df, y_train_df], axis=1)
            test_processed = pd.concat([X_test_transformed_df, y_test_df], axis=1)

            logging.info("Saving processed data")
            train_processed.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test_processed.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        except Exception as e:
            raise MyException(e, sys)