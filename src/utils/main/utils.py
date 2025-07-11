from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

import os
import sys
import yaml
import dill
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

## Yaml related functions
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


## Numpy related functions
def save_numpy_array_data(file_path: str, array: np.array):
    """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
    """

    try:
        dir_path = os.path.dirname(file_path)    
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


## Pickle related funcions
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info('Entered the save_object method of Utils class')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info('Exited the save_object method of Utils class')
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'The file: {file_path} does not exist')
        with open(file_path, 'rb') as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

## Evaluate model
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple classification models using RandomizedSearchCV for hyperparameter tuning.
    X_train, y_train: Training data
    X_test, y_test: Testing data
    models: Dictionary of models to evaluate
    params: Dictionary of hyperparameters for each model
    """
    try:
        report = dict()

        for model_name, model in models.items():
            logging.info(f'Evaluating model: {model_name}')

            param_grid = params[model_name]
            
            rs = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=60, 
                                    scoring='roc_auc', cv=3, verbose=2, random_state=42, n_jobs=-1)
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = roc_auc_score(y_train, y_train_pred)
            test_model_score = roc_auc_score(y_test, y_test_pred)

            # save score in the report
            report[model_name] = test_model_score
            logging.info(f'Model: {model_name}, Best Score: {test_model_score}, Best Params: {rs.best_params_}')

        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)