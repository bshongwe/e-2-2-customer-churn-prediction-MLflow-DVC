import sys
from pathlib import Path
from mlops_project.config.configuration import ConfigurationManager
from mlops_project.components.data_transformation import DataTransformation
from mlops_project.logger import logging
from mlops_project.exception import MyException

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:

            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.preprocess_data()
            else:
                raise Exception("Data schema is not valid")
            
        except Exception as e:
            raise MyException(e, sys)
        

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise MyException(e, sys)