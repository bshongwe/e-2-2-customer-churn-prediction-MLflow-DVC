import os
import yaml
import sys
import json
from pathlib import Path
from src.mlops_project.exception import MyException
from src.mlops_project.logger import logging
from box import ConfigBox
from box.exceptions import BoxValueError

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns
    Args:
        path_to_yaml: path to yaml file
    Raises:
        ValueError: if yaml file is empty
        e: empty exception
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            config_box = ConfigBox(content)
            return config_box
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise MyException(e, sys)
    

def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")

def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")
