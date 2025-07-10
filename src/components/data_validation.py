from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constant.training_pipeline import SCHEMA_FILE_PATH

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.utils import read_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os
import sys


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config)

            logging.info(f'Required Number of Columns: {number_of_columns}')
            logging.info(f'DataFrame Number of Columns: {len(dataframe.columns)}')

            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_numerical_columns = self._schema_config.get('numerical_columns', [])
            df_numerical_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

            logging.info(f'Expected Numerical Columns: {expected_numerical_columns}')
            logging.info(f'DataFrame Numerical Columns: {df_numerical_columns}')

            if set(expected_numerical_columns) == set(df_numerical_columns):
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            ## validate number of columns
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                error_message = f'Train Dataframe does not contain all collumns\n'
                raise NetworkSecurityException(error_message, sys)
            
            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                error_message = f'Test DataFrame does not contain all columns\n'
                raise NetworkSecurityException(error_message, sys)

            ## validate numerical columns
            status = self.validate_numerical_columns(dataframe=train_df)
            if not status:
                error_message = f'Train DataFrame does not contain all numerical columns'
                raise NetworkSecurityException(error_message, sys)

            status = self.validate_numerical_columns(dataframe=test_df)
            if not status:
                error_message = f'Test DataFrame does not contain all numerical columns'
                raise NetworkSecurityException(error_message, sys)
            
            ## check datadrift

        except Exception as e:
            raise NetworkSecurityException(e, sys)