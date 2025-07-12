import os
import sys

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.constant.training_pipeline import MODEL_FILE_NAME, FINAL_MODEL_DIR

from src.utils.main.utils import save_object, load_object
from src.utils.main.utils import load_numpy_array_data
from src.utils.main.utils import evaluate_models
from src.utils.ml.model.estimator import NetworkModel
from src.utils.ml.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

import dagshub
import mlflow
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

# dagshub.init(repo_owner='tomosantos', repo_name='network-security', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def track_mlflow(self, model_name: str, best_model, classification_metric: get_classification_score, stage: str):
        """
        Logs model name, metrics, and other details to MLflow.
        model_name: Name of the model being logged
        best_model: The trained model object
        classification_metric: Object containing classification metrics (precision, recall, F1-score)
        stage: Indicates whether the metrics are from training or testing (e.g., 'train' or 'test')
        """
        try:
            mlflow.set_registry_uri('https://dagshub.com/tomosantos/network-security.mlflow')
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # log model name
                mlflow.log_param('model_name', model_name)
                mlflow.log_param('stage', stage)

                f1_score = classification_metric.f1_score
                precision_score = classification_metric.precision_score
                recall_score = classification_metric.recall_score

                mlflow.log_metric(f'{stage}_f1_score', f1_score)
                mlflow.log_metric(f'{stage}_precision', precision_score)
                mlflow.log_metric(f'{stage}_recall_score', recall_score)
                mlflow.sklearn.log_model(best_model, 'model')

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(best_model, 'model', registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(best_model, 'model')

                logging.info(f'Logged {model_name} metrics and model to MLflow for {stage} stage')
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                    "Random Forest": RandomForestClassifier(verbose=1),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                    "Logistic Regression": LogisticRegression(verbose=1),
                    "AdaBoost": AdaBoostClassifier()
                }
            
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 128, 256]
                },
                "Gradient Boosting":{
                    'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1, .01, .05, .001],
                    'subsample':[0.6, 0.7, 0.75, 0.85, 0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models, params=params)
            
            ## get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## get best model name form dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            ## track experiments with mlflow
            self.track_mlflow(model_name=best_model_name, best_model=best_model,
                              classification_metric=classification_train_metric, stage='train')

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            self.track_mlflow(model_name=best_model_name, best_model=best_model,
                              classification_metric=classification_test_metric, stage='test')

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=NetworkModel)

            final_model_file_path = os.path.join(FINAL_MODEL_DIR, MODEL_FILE_NAME)

            # model pusher
            save_object(final_model_file_path, best_model)
            
            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            logging.info(f'Model Trainer Artifact: {model_trainer_artifact}')

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)