import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlProject.utils.common import save_json
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import urllib.request


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1
    
    def log_into_mlflow(self):
        """
        Standardized production routine tracking pipeline metrics, parameters, 
        and model binaries across local and automated CI/CD environments.
        """
        # Load processed evaluation sets and binary checkpoints
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        # 1. Environment Routing Setup
        default_tracking_uri = "http://127.0.0.1:5000"
        try:
            # Probe environment to see if the local MLflow server UI instance is listening
            urllib.request.urlopen(default_tracking_uri, timeout=1)
            mlflow.set_tracking_uri(default_tracking_uri)
        except Exception:
            # Fallback to isolated database tracking for head-less cloud systems (GitHub Actions)
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # 2. Pipeline Tracking Run Execution
        with mlflow.start_run():
            predicted_status = model.predict(test_x)
            (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predicted_status)
            
            # Save tracking metrics to local artifacts folder
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log evaluation metrics to the active MLflow store
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # 3. Standardized Model Logging
            # Forcing serialization_format="pickle" avoids security scanner constraints (skops)
            # when tracking third-party algorithms wrapped in scikit-learn classes.
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="model", 
                    registered_model_name="XGBoostCreditRiskModel",
                    serialization_format="pickle"
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="model",
                    serialization_format="pickle"
                )