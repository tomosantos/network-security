from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

import os
import sys
import yaml
import dill
import pickle
import numpy as np
import pandas as pd


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)