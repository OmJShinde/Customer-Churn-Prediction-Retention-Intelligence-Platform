from pydantic import BaseModel, Field, validator
from typing import Optional, List

class CustomerInput(BaseModel):
    # Demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., description="Whether the customer is a senior citizen (1, 0)")
    Partner: str = Field(..., description="Whether the customer has a partner (Yes, No)")
    Dependents: str = Field(..., description="Whether the customer has dependents (Yes, No)")
    
    # Services
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company")
    PhoneService: str = Field(..., description="Whether the customer has a phone service (Yes, No)")
    MultipleLines: str = Field(..., description="Whether the customer has multiple lines (Yes, No, No phone service)")
    InternetService: str = Field(..., description="Customer's internet service provider (DSL, Fiber optic, No)")
    OnlineSecurity: str = Field(..., description="Whether the customer has online security (Yes, No, No internet service)")
    OnlineBackup: str = Field(..., description="Whether the customer has online backup (Yes, No, No internet service)")
    DeviceProtection: str = Field(..., description="Whether the customer has device protection (Yes, No, No internet service)")
    TechSupport: str = Field(..., description="Whether the customer has tech support (Yes, No, No internet service)")
    StreamingTV: str = Field(..., description="Whether the customer has streaming TV (Yes, No, No internet service)")
    StreamingMovies: str = Field(..., description="Whether the customer has streaming movies (Yes, No, No internet service)")
    
    # Account
    Contract: str = Field(..., description="The contract term (Month-to-month, One year, Two year)")
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing (Yes, No)")
    PaymentMethod: str = Field(..., description="The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))")
    MonthlyCharges: float = Field(..., ge=0, description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., ge=0, description="The total amount charged to the customer")

    @validator("TotalCharges", pre=True)
    def handle_empty_total_charges(cls, v):
        if v == " " or v == "":
            return 0.0
        return v

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_factors: List[str]

class ExplanationOutput(BaseModel):
    feature_importance: dict
    explanation_text: str

class RetentionStrategy(BaseModel):
    strategy: str
    action_items: List[str]
    email_draft: Optional[str] = None
