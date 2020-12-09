from datetime import datetime
from typing import Optional, List, Tuple

from pydantic import BaseModel, Extra


NumericHistogram = List[Tuple[float, float]]
TimeHistogram = List[Tuple[datetime, float]]


class ModelEndpoint(BaseModel):
    project: Optional[str]
    model: Optional[str]
    function: Optional[str]
    tag: Optional[str]
    model_class: Optional[str]
    labels: Optional[List[str]]

    class Config:
        extra = Extra.allow


class ModelEndpointsList(BaseModel):
    endpoints: List[ModelEndpoint]


class FeatureDetails(BaseModel):
    name: str
    weight: float
    expected_min: float
    expected_avg: float
    expected_max: float
    expected_hist: NumericHistogram
    actual_min: Optional[float]
    actual_avg: Optional[float]
    actual_max: Optional[float]
    actual_hist: Optional[NumericHistogram]


class ModelEndpointState(BaseModel):
    model_endpoint: ModelEndpoint
    first_request: Optional[str]
    last_request: Optional[str]
    average_latency: Optional[float]
    accuracy: Optional[float]
    error_count: Optional[int]
    alert_count: Optional[int]
    drift_status: Optional[str]
    requests_histogram: Optional[TimeHistogram]
    predictions_histogram: Optional[TimeHistogram]
    feature_details: Optional[FeatureDetails]
