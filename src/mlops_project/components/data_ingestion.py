import os
import sys
import urllib.request as request
import zipfile
from mlops_project.logger import logging
from mlops_project.entity.config_entity import DataIngestionConfig
from mlops_project.exception import MyException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Download file from source url
        Returns: None
        """
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_data_file
                )
                logging.info(f"{filename} downloaded with following info: \n{headers}")
            else:
                logging.info(f"File already exists: {self.config.local_data_file}")
                
        except Exception as e:
            raise MyException(e, sys)

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory 
        Returns: None
        """  
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            logging.info(f"Extracting zip file: {self.config.local_data_file} into dir: {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)

        except Exception as e:
            raise MyException(e, sys)