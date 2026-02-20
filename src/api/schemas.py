"""
src/api/schemas.py
Pydantic request/response schemas for the yield prediction API.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class PredictRequest(BaseModel):
    wafer_id: str = Field(default="wafer_001", description="Unique wafer identifier")
    sensor_readings: Dict[str, float] = Field(
        description="Dictionary of sensor_name: reading_value"
    )
    threshold: Optional[float] = Field(
        default=0.15,
        description="Decision threshold (default 0.15, tuned on validation set)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "wafer_id": "wafer_001",
                "sensor_readings": {
                    "sensor_033": 0.42,
                    "sensor_103": -0.15,
                    "sensor_031": 1.23,
                },
                "threshold": 0.15,
            }
        }

class PredictResponse(BaseModel):
    wafer_id: str
    failure_probability: float = Field(description="Probability of yield failure [0,1]")
    prediction: int             = Field(description="Binary prediction: 0=pass, 1=fail")
    prediction_label: str       = Field(description="PASS or FAIL")
    confidence: float           = Field(description="Model confidence in prediction")
    threshold_used: float       = Field(description="Decision threshold applied")

class BatchPredictRequest(BaseModel):
    wafers: List[PredictRequest]
    threshold: Optional[float] = 0.15

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    n_wafers: int
    n_predicted_fail: int
    n_predicted_pass: int
    fail_rate: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    version: str
