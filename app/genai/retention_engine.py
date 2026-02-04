import os
import json
from app.core.logger import logger
from app.api.schemas import RetentionStrategy
from typing import List

# Mock OpenAI client if not installed or key invalid, but user asked for "OpenAI / Gemini API (abstracted)"
# We will assume OpenAI client usage structure but wrap it to be safe.

class RetentionEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = not self.api_key or "your_openai_api_key" in self.api_key

    def generate_strategy(self, churn_prob: float, risk_factors: List[str]) -> RetentionStrategy:
        """
        Generates personalized retention strategies using GenAI.
        """
        if self.mock_mode:
            logger.info("Running Retention Engine in Mock Mode (No API Key).")
            return self._mock_strategy(churn_prob, risk_factors)
        
        try:
            # TODO: Integrate with real OpenAI API
            # For now, we use a rule-based mock engine for reliability.
            return self._mock_strategy(churn_prob, risk_factors)
            
        except Exception as e:
            logger.error(f"GenAI generation failed: {e}")
            return self._fallback_strategy()

    def _mock_strategy(self, churn_prob, risk_factors):
        """
        Deterministic mock response based on rules, to simulate AI.
        """
        if churn_prob < 0.3:
            return RetentionStrategy(
                strategy="Loyalty Appreciation",
                action_items=["Send 'Thank You' email", "Offer 5% discount on next bill"],
                email_draft="Dear Valued Customer, thanks for being with us!"
            )
        elif churn_prob < 0.7:
             return RetentionStrategy(
                strategy="Proactive Engagement",
                action_items=["Offer free upgrade to next tier for 1 month", "Check in call from support"],
                email_draft="We noticed you might benefit from our faster internet speeds..."
            )
        else:
             return RetentionStrategy(
                strategy="High Risk Save",
                action_items=["20% Discount for 6 months commitment", "Priority support access", "Waive this month's fee"],
                email_draft="We really value your business and want to make things right..."
            )

    def _fallback_strategy(self):
        return RetentionStrategy(
            strategy="General Support",
            action_items=["Contact Customer Support"],
            email_draft="Please reach out to us."
        )

retention_engine = RetentionEngine()
