from pydantic import BaseModel, Field
from enum import Enum
from uuid import UUID
from datetime import datetime
import pandas as pd

class MerchantCategory(str, Enum):
    GROCERY = "grocery"
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    OTHER = "other"

class TransactionSchema(BaseModel):
    user_id: int = Field(gt=0)
    amount: float = Field(gt=0, le=100_000)
    merchant_category: MerchantCategory
    location_lat: float = Field(ge=-90, le=90)
    location_lon: float = Field(ge=-180, le=180)
    is_international: bool = False
    
    def to_features(self):
        return pd.DataFrame([self.model_dump(mode="json")])

class PredictionResponseSchema(BaseModel):
    transaction_id: UUID
    timestamp: datetime
    is_fraud: bool
    fraud_probability: float
    risk_level: str
