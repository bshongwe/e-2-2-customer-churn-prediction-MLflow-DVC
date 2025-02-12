import os
import sys
import pandas as pd
from mlops_project.logger import logging
from mlops_project.exception import MyException
from mlops_project.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_columns(self)-> bool:
        """
        Validate all the columns of the data
        
        Returns: bool value of validation
        """
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            data_columns = list(data.columns)
            data_dtypes =  data.dtypes.astype(str).tolist()

            schema = self.config.all_schema
            schema_cols = self.config.all_schema.keys()

            logging.info("Validating the columns of the data")
            for col, dtype in zip(data_columns, data_dtypes):
                if (col not in schema_cols) or (schema.get(col) != dtype):
                    validation_status = False
                    break
                else:
                    validation_status = True
            
            logging.info(f"Validation status: {validation_status}")
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
        
        except Exception as e:
            raise MyException(e, sys)