#!/usr/bin/env python3

import sys
from mlops_project.logger import logging
from mlops_project.exception import MyException
from mlops_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlops_project.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlops_project.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlops_project.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from mlops_project.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise MyException(e, sys)


STAGE_NAME = "Data Validation stage"

try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise MyException(e, sys)


STAGE_NAME = "Data Transformation stage"

try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise MyException(e, sys)


STAGE_NAME = "Model Trainer stage"

try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerTrainingPipeline()        
    obj.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise MyException(e, sys)


STAGE_NAME = "Model Evaluation stage"

try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise MyException(e, sys)
