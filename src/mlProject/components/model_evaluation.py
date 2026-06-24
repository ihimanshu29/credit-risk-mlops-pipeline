import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlProject.utils.common import save_json
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse


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
        """Streams pipeline runs, tracking matrices, and model binaries into MLflow UI"""
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        #  DEFENSIVE TRACKING GATE: 
        # Safely determine if we use the localhost server or a local directory
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        try:
            # Attempt a quick 1-second handshake with the local server
            urllib.request.urlopen(default_tracking_uri, timeout=1)
            mlflow.set_tracking_uri(default_tracking_uri)
        except Exception:
            # Fallback for GitHub Actions / Headless environments
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Start MLflow tracking session
        with mlflow.start_run():
            predicted_status = model.predict(test_x)
            (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predicted_status)
            
            # 1. Save metrics locally to keep original functionality intact
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # 2. Log Metrics to MLflow Dashboard
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # 3. Log Model binary with signatures directly to the MLflow Registry
            # Model registry works perfectly now because the URI scheme is 'http'
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGBoostCreditRiskModel")
            else:
                mlflow.sklearn.log_model(model, "model")