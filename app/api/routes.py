from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas import CustomerInput, PredictionOutput, ExplanationOutput, RetentionStrategy
from app.ml.predict import predictor
from app.explainability.shap_explainer import shap_service
from app.genai.retention_engine import retention_engine
import pandas as pd
from app.core.logger import logger

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict_churn(customer: CustomerInput):
    """
    Predicts customer churn probability.
    """
    logger.info("Received prediction request")
    try:
        # Convert Pydantic to DataFrame (1 row)
        input_data = customer.dict()
        df = pd.DataFrame([input_data])
        
        result = predictor.predict(df)
        
        # Add risk factors (placeholder logic if not from SHAP yet, but we'll merge them in flow)
        # Actually the prediction output schema asks for risk factors. 
        # We can run SHAP here or separately. Let's return basics here.
        result['risk_factors'] = [] 
        
        return result
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain", response_model=ExplanationOutput)
async def explain_churn(customer: CustomerInput):
    """
    Explains the prediction using SHAP.
    """
    try:
        input_data = customer.dict()
        df = pd.DataFrame([input_data])
        explanation = shap_service.explain_local(df)
        
        if "error" in explanation:
            raise HTTPException(status_code=500, detail=explanation['error'])
            
        return explanation
    except Exception as e:
        logger.error(f"Explanation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retention", response_model=RetentionStrategy)
async def generate_retention(churn_prob: float, risk_factors: list[str]):
    """
    Generates a retention strategy.
    """
    try:
        strategy = retention_engine.generate_strategy(churn_prob, risk_factors)
        return strategy
    except Exception as e:
        logger.error(f"Retention endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
