import shap
import pandas as pd
import numpy as np
import joblib
import logging
from app.core.config import settings
from app.core.logger import logger

class ShapExplainer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self._load_resources()

    def _load_resources(self):
        try:
            full_pipeline = joblib.load(settings.MODEL_PATH)
            self.model = full_pipeline.named_steps['classifier']
            self.preprocessor = full_pipeline.named_steps['preprocessor']
            logger.info("SHAP Explainer resources loaded.")
        except Exception as e:
            logger.error(f"Failed to load model for SHAP: {e}")

    def explain_local(self, input_df: pd.DataFrame):
        """
        Returns feature importance for a specific prediction.
        """
        if self.model is None:
            self._load_resources()
        
        try:
            # Transform input using the pipeline's preprocessor
            X_encoded = self.preprocessor.transform(input_df)
            
            # Create explainer if not cached (TreeExplainer is fast enough to init usually, 
            # but for speed we could cache it if we had a background dataset)
            # using TreeExplainer as we use XGBoost
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            
            shap_values = self.explainer.shap_values(X_encoded)
            
            # Get feature names from preprocessor
            feature_names = self._get_feature_names()
            
            # Map values to features (handling the list return from shap_values for classification)
            # XGBoost binary classification returns a single array, sklearn RF returns list.
            # Assuming XGBoost here based on train.py
            
            if isinstance(shap_values, list):
                vals = shap_values[1][0] # Positive class
            else:
                vals = shap_values[0]

            # Create a dict of feature: importance
            # We sort by absolute value
            feature_importance = dict(zip(feature_names, vals))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Return top 5 drivers
            top_drivers = {k: float(v) for k, v in sorted_importance[:5]}
            
            # Generate simple text explanation
            explanation_text = "Key factors: " + ", ".join([f"{k} ({'increases' if v > 0 else 'decreases'} risk)" for k, v in sorted_importance[:3]])
            
            return {
                "feature_importance": top_drivers,
                "explanation_text": explanation_text
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": "Could not generate explanation"}

    def _get_feature_names(self):
        """
        Extracts feature names from the column transformer.
        """
        output_features = []
        try:
            transformers = self.preprocessor.transformers_
            for name, estimator, columns in transformers:
                if name == 'num':
                    output_features.extend(columns)
                elif name == 'cat':
                    # Handle different definitions of get_feature_names_out depending on version
                    if hasattr(estimator, 'get_feature_names_out'):
                        output_features.extend(estimator.get_feature_names_out(columns))
                    else:
                        output_features.extend(estimator.get_feature_names(columns))
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            # Fallback
            output_features = [f"Feature {i}" for i in range(100)] 
        return output_features

shap_service = ShapExplainer()
