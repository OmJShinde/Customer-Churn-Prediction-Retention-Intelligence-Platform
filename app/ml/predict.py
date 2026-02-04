import joblib
import pandas as pd
import logging
from app.core.config import settings
from app.core.logger import logger

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from {settings.MODEL_PATH}: {e}")
            # Fallback or re-raise depending on policy. For now, log.
    
    def predict(self, input_df: pd.DataFrame):
        """
        Returns a dictionary with probability and prediction.
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise ValueError("Model not loaded.")
        
        try:
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0][1]
            
            return {
                "churn_prediction": int(prediction),
                "churn_probability": float(probability)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e

# Global instance
predictor = ChurnPredictor()
