import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import logging
from app.core.logger import logger
from app.core.config import settings
from app.data.loader import load_data, preprocess_data

def train_model():
    """
    Trains the churn prediction model and saves it.
    """
    logger.info("Starting model training pipeline...")
    
    # 1. Load Data
    try:
        df = load_data(settings.DATA_PATH)
    except FileNotFoundError:
        logger.error("Data file not found. Please ensure data is available.")
        return

    df = preprocess_data(df)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define Preprocessing Pipeline
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # 4. Define Model (XGBoost)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        ))
    ])
    
    # 5. Train
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC-AUC Prediction: {roc_auc:.4f}")
    
    report = classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{report}")
    
    # 7. Save Model
    logger.info(f"Saving model to {settings.MODEL_PATH}")
    joblib.dump(model, settings.MODEL_PATH)
    
    return model, roc_auc

if __name__ == "__main__":
    train_model()
