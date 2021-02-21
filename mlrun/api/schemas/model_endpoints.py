from hashlib import md5
from typing import Optional, List, Tuple, Any, Dict, Union

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
    id: Optional[str] = None

    class Config:
        extra = Extra.allow

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.id = self.create_endpoint_id()

    def create_endpoint_id(self):
        if not self.spec.function or not self.spec.model or not self.metadata.tag:
            raise ValueError(
                "ModelEndpoint.spec.function, ModelEndpoint.spec.model "
                "and ModelEndpoint.metadata.tag must be initalized"
            )

        endpoint_unique_string = (
            f"{self.spec.function}_{self.spec.model}_{self.metadata.tag}"
        )
        md5_str = md5(endpoint_unique_string.encode("utf-8")).hexdigest()
        return f"{self.metadata.project}.{md5_str}"


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
    first_request: Optional[str] = None
    last_request: Optional[str] = None
    accuracy: Optional[float] = None
    error_count: Optional[int] = None
    drift_status: Optional[str] = None
    metrics: Dict[str, Metric] = {}
    features: List[Features] = []


class ModelEndpointStateList(BaseModel):
    endpoints: List[ModelEndpointState]


class GrafanaColumn(BaseModel):
    text: str
    type: str


class GrafanaTable(BaseModel):
    columns: List[GrafanaColumn]
    rows: List[List[Optional[Union[int, float, str]]]]
    type: str = "table"
