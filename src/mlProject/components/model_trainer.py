import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        # Load fully transformed data (now preserving the target column header)
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # --- EXPLICIT CONFIGURATION-DRIVEN SPLIT ---
        # Isolate features from targets using the exact name from schema.yaml
        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        logger.info(f"Initializing XGBoost using scale_pos_weight={self.config.scale_pos_weight}")

        # Initialize the modern XGBoost Classifier using completely configuration-driven parameters
        model = XGBClassifier(
            n_estimators=self.config.n_estimators, 
            max_depth=self.config.max_depth, 
            learning_rate=self.config.learning_rate,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=42
        )
        
        # Train against the preprocessed feature matrices
        model.fit(train_x, train_y)

        # Serialize model output
        model_output_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(model, model_output_path)
        logger.info(f"Trained model artifact saved successfully at: {model_output_path}")