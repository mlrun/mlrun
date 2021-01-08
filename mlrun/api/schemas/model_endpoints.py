from typing import Optional, List, Tuple

from pydantic import BaseModel, Field
from pydantic.main import Extra

from mlrun.api.schemas.object import (
    ObjectKind,
    ObjectSpec,
    ObjectStatus,
)


class ModelEndpointMetadata(BaseModel):
    project: Optional[str]
    tag: Optional[str]
    labels: Optional[dict]

    class Config:
        extra = Extra.allow


class ModelEndpointSpec(ObjectSpec):
    model: Optional[str]
    function: Optional[str]
    model_class: Optional[str]


class ModelEndpoint(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ModelEndpointMetadata
    spec: ModelEndpointSpec
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


class FeatureValues(BaseModel):
    min: float
    avg: float
    max: float
    histogram: Histogram


class Features(BaseModel):
    name: str
    weight: float
    expected: FeatureValues
    actual: Optional[FeatureValues]


class ModelEndpointState(BaseModel):
    endpoint: ModelEndpoint
    first_request: Optional[str]
    last_request: Optional[str]
    accuracy: Optional[float] = None
    error_count: Optional[int]
    alert_count: Optional[int]
    drift_status: Optional[str]
    metrics: Optional[List[Metric]] = None
    features: Optional[List[Features]] = None


class ModelEndpointStateList(BaseModel):
    endpoints: List[ModelEndpointState]
