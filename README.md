# Customer Churn Prediction & Retention Intelligence Platform

## 1. Project Overview

The **Customer Churn Prediction & Retention Intelligence Platform** is an end-to-end enterprise solution designed to predict, explain, and mitigate customer churn. This system leverages advanced machine learning to forecast the likelihood of customer attrition, utilizes Explainable AI (XAI) to identify the specific drivers behind each prediction, and employs Generative AI to formulate personalized retention strategies.

This platform addresses the critical business need for proactive customer retention by transforming raw data into actionable intelligence, enabling organizations to intervene before valuable customers leave.

## 2. Business Problem Statement

**Customer Churn** is the phenomenon where customers stop doing business with a company. For subscription-based businesses like telecommunications, high churn rates directly erode revenue, increase customer acquisition costs, and reduce long-term profitability.

Predictive analytics allows businesses to shift from a reactive to a proactive stance. Instead of trying to win back a customer after they have cancelled, organizations can identify at-risk customers in advance and offer targeted incentives to retain them. This project demonstrates how modern ML techniques can accurately identify these customers and provide the context needed for effective intervention.

## 3. Dataset Description

The project is built upon the **IBM Telco Customer Churn Dataset**, a widely recognized industry standard for churn analysis.

*   **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
*   **Records:** 7,043 unique customer profiles
*   **Features:** 21 attributes per customer
*   **Target Variable:** `Churn` (Yes/No) - Indicates if the customer left within the last month.

### Feature Breakdown

#### A. Customer Demographics
*   `gender`: Whether the customer is male or female.
*   `SeniorCitizen`: Flag indicating if the customer is 65 or older (1 = Yes, 0 = No).
*   `Partner`: Whether the customer has a partner.
*   `Dependents`: Whether the customer has dependents.
*   `tenure`: The number of months the customer has stayed with the company.

#### B. Service Subscriptions
*   `PhoneService`: Service usage indicator.
*   `MultipleLines`: Indicates if customer has single or multiple phone lines.
*   `InternetService`: Provider type (DSL, Fiber optic, or No).
*   `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Boolean flags for various add-on services. Note that for customers without internet service, these columns may contain "No internet service".

#### C. Contract & Billing
*   `Contract`: The contract term (Month-to-month, One year, Two year). Month-to-month contracts historically show higher churn rates.
*   `PaperlessBilling`: Whether the customer uses paperless billing.
*   `PaymentMethod`: The method used for payment (Electronic check, Mailed check, Bank transfer, Credit card).

#### D. Financial Information
*   `MonthlyCharges`: The amount charged to the customer monthly.
*   `TotalCharges`: The cumulative amount charged to the customer.

## 4. Project Architecture

The system is designed as a modular, API-first microservice architecture.

```mermaid
graph TD
    User[User / Analyst] --> UI[Streamlit Dashboard]
    UI --> API[FastAPI Backend]
    
    subgraph "Core Intelligence Engine"
        API --> ML[ML Prediction Engine]
        API --> XAI[Explainability Engine (SHAP)]
        API --> GenAI[GenAI Retention Engine]
    end
    
    ML --> Model[XGBoost Model]
    XAI --> Model
    GenAI --> Rules[Retention Rules / LLM]
```

### Core Components
1.  **Data Ingestion & Validation**: Pydantic schemas ensure that all incoming data meets strict type and range requirements before processing.
2.  **ML Engine**: Preprocesses data and generates churn probability scores using a trained XGBoost classifier.
3.  **Explainability Engine**: Uses SHAP (SHapley Additive exPlanations) to calculate the marginal contribution of each feature to the final prediction.
4.  **GenAI Retention Engine**: Synthesizes the risk score and key drivers to generate a readable, actionable retention strategy.
5.  **UI Layer**: A responsive dashboard that presents complex analytics in a user-friendly format.

## 5. Technology Stack

*   **Language**: Python 3.10+
*   **API Framework**: FastAPI (High performance, async support)
*   **Frontend**: Streamlit (Rapid data application development)
*   **Machine Learning**:
    *   `scikit-learn`: Data preprocessing pipelines
    *   `xgboost`: Gradient boosting framework
    *   `joblib`: Model serialization
*   **Explainable AI**: `shap` (Model interpretation)
*   **Data Processing**: `pandas`, `numpy`
*   **Validation**: `pydantic`
*   **Visualization**: `plotly`

## 6. Machine Learning Approach

### Problem Formulation
This is a supervised binary classification problem where the goal is to predict the probability of the positive class (`Churn = Yes`).

### Preprocessing & Feature Engineering
*   **Handling Missing Values**: `TotalCharges` is coerced to numeric, with missing values (for new customers) filled with 0.
*   **Encoding**:
    *   Numerical features (`tenure`, `MonthlyCharges`) are standardized using `StandardScaler`.
    *   Categorical features are encoded using `OneHotEncoder` to handle non-numeric data suitable for XGBoost.
*   **Pipeline**: All preprocessing steps are wrapped in a Scikit-learn FeatureUnion to prevent data leakage during inference.

### Model Selection: XGBoost
XGBoost was selected over Logistic Regression and Random Forest due to its:
*   Superior performance on structured/tabular data.
*   Ability to capture complex non-linear relationships.
*   Built-in handling of missing values and regularization to prevent overfitting.

### Evaluation Metrics
*   **ROC-AUC**: Primary metric to evaluate the model's ability to distinguish between classes.
*   **Precision/Recall**: Monitored to balance the cost of false positives (unnecessary discounts) vs. false negatives (lost customers).

## 7. Explainable AI (XAI)

In regulated industries and enterprise decision-making, "black box" predictions are often unacceptable. Stakeholders need to know *why* a customer is flagged as high risk.

### SHAP Implementation
This project uses **SHAP (SHapley Additive exPlanations)**, a game-theoretic approach to explain the output of any machine learning model.
*   **Global Importance**: Identifying which features (e.g., Contract Type, Tenure) are most predictive across the entire dataset.
*   **Local Importance**: For any specific customer, SHAP values quantify how much each feature increased or decreased their risk score relative to the baseline.

## 8. GenAI Integration

The project includes a generative component to close the loop from "insight" to "action."

*   **Input**: The engine receives the predicted Churn Probability and the Top 3 Risk Factors identified by SHAP.
*   **Logic**: It maps these inputs to a repository of retention tactics (e.g., specific discounts for price-sensitive users, tech support outreach for service failures).
*   **Output**: A cohesive strategy summary and a draft communication (email/script) for the customer success team.
*   **Safety**: Usage of structured prompts and rule-based fallbacks ensures the AI does not hallucinate non-existent company policies.

## 9. User Interface Overview

The UI is built with **Streamlit** to be a professional, enterprise-grade dashboard.

*   **Target Audience**: Customer Success Managers, Retention Specialists, and Business Analysts.
*   **Design Philosophy**:
    *   **Clarity**: High-contrast typography and clear visual hierarchy.
    *   **Timelessness**: Minimalist aesthetic focusing on data over decoration.
    *   **Feedback**: Subtle animations and loading states provide system status visibility.
*   **Key Features**:
    *   Interactive sidebar for customer profile simulation.
    *   Real-time gauge metrics for risk visualization.
    *   Dynamic bar charts for feature importance.
    *   Actionable strategy cards.

## 10. Security & Robustness

*   **Configuration**: All sensitive configuration (API keys, file paths) is managed via environment variables (`.env`).
*   **Validation**: Pydantic models enforce strict data typing at the API boundary, rejecting malformed requests before they reach the model.
*   **Logging**: A rotating file logger captures all system events, errors, and access logs for auditability.
*   **Resilience**: The system is designed to degrade gracefully (e.g., providing ML predictions even if the GenAI service is unreachable).

## 11. Project Structure

```bash
customer-churn-ai/
│
├── app/
│   ├── api/            # API Routes and Pydantic Schemas
│   ├── core/           # Configuration, Logging, and Security
│   ├── data/           # Data Loaders and Preprocessing Logic
│   ├── explainability/ # SHAP Explanation Logic
│   ├── genai/          # Retention Strategy Generation
│   ├── ml/             # Model Training and Inference Service
│   └── ui/             # Streamlit Dashboard Code
│
├── logs/               # Application Logs
├── models/             # Serialized ML Models (.pkl)
├── requirements.txt    # Project Dependencies
├── .env.example        # Environment Variable Template
└── README.md           # Project Documentation
```

## 12. How to Run the Project

### Prerequisites
*   Python 3.10 or higher
*   pip package manager

### Installation

1.  **Clone the repository**
2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Execution

1.  **Start the API Server**:
    In a terminal:
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

2.  **Start the Dashboard**:
    In a NEW terminal:
    ```bash
    streamlit run app/ui/dashboard.py
    ```

## 13. Future Enhancements

*   **MLOps Pipeline**: Integrate with tools like MLflow or DVC for model versioning and experiment tracking.
*   **Real-time Inference**: Connect to a live database or event stream (Kafka) for triggering predictions on customer activity.
*   **Drift Detection**: Implement monitoring to alert when the input data distribution shifts significantly from the training data.
*   **A/B Testing**: Framework to test different retention strategies and measure their actual success rates.

## 14. Conclusion

This project demonstrates a production-ready approach to AI-driven customer retention. By combining robust machine learning, transparent explainability, and generative AI, it provides a tool that not only predicts the future but empowers businesses to change it.
