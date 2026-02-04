import pandas as pd
import numpy as np
import os
from app.core.logger import logger

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified filepath.
    """
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        # Handle TotalCharges being object type due to empty strings
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise e

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the dataframe: cleans data, encodes categoricals.
    """
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Simple preprocessing output for now (X, y)
    # We will use a transformer pipeline in the ML module, so here just basic cleaning
    
    # Drop rows with missing target if applicable
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df
