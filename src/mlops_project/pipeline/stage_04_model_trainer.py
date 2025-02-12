import sys
from mlops_project.config.configuration import ConfigurationManager
from mlops_project.components.model_trainer import ModelTrainer
from mlops_project.logger import logging
from mlops_project.exception import MyException

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()
        except Exception as e:
            raise MyException(e, sys)
        

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()        
        obj.main()
        logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise MyException(e, sys)