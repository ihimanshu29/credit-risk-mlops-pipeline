import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]

        # Initialize the modern XGBoost Classifier
        model = XGBClassifier(
            n_estimators=self.config.n_estimators, 
            max_depth=self.config.max_depth, 
            learning_rate=self.config.learning_rate,
            random_state=42
        )
        # XGBoost prefers 1D array for labels, hence .values.ravel()
        model.fit(train_x, train_y.values.ravel())

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))