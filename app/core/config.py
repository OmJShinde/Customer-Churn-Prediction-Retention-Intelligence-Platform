import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Customer Churn Prediction"
    VERSION: str = "1.0.0"
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/churn_model.pkl")
    DATA_PATH: str = os.getenv("DATA_PATH", "app/data/telco_customer_churn.csv")

settings = Settings()
