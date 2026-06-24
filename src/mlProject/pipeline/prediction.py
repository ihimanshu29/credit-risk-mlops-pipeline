import joblib
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load both the model AND the data transformer
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.pkl'))

    def predict(self, data: pd.DataFrame):
        # 1. Transform the incoming data (Imputing, Scaling, One-Hot Encoding)
        transformed_data = self.preprocessor.transform(data)
        
        # 2. Predict using the XGBoost model
        prediction = self.model.predict(transformed_data)
        
        return prediction