from typing import Optional, List, Tuple

from pydantic import BaseModel, Field

from mlrun.api.schemas import ObjectKind, ObjectMetadata, ObjectSpec, ObjectStatus


class EndpointSpec(ObjectSpec):
    model: Optional[str]
    function: Optional[str]
    model_class: Optional[str]


class Endpoint(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ObjectMetadata
    spec: EndpointSpec
    status: ObjectStatus


class Histogram(BaseModel):
    buckets: List[Tuple[float, float]]
    count: List[int]


class Metric(BaseModel):
    name: str
    start_timestamp: str
    end_timestamp: str
    headers: List[str]
    values: List[Tuple[str, float]]
    min: float
    avg: float
    max: float


class MetricList(BaseModel):
    metrics: List[Metric]


class Features(BaseModel):
    name: str
    weight: float
    # Expected
    expected_min: float
    expected_avg: float
    expected_max: float
    expected_hist: Histogram
    # Actual
    actual_min: Optional[float]
    actual_avg: Optional[float]
    actual_max: Optional[float]
    actual_hist: Histogram


class FeatureList(BaseModel):
    features: List[Features]


class EndpointState(BaseModel):
    endpoint: Endpoint
    first_request: Optional[str]
    last_request: Optional[str]
    accuracy: Optional[float]
    error_count: Optional[int]
    alert_count: Optional[int]
    drift_status: Optional[str]
    metrics: Optional[MetricList] = None
    features: Optional[FeatureList] = None


class EndpointStateList(BaseModel):
    endpoints: List[EndpointState]
