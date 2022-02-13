from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic.main import Extra

from mlrun.api.schemas.object import ObjectKind, ObjectSpec, ObjectStatus
from mlrun.utils.model_monitoring import EndpointType, create_model_endpoint_id


class ModelMonitoringStoreKinds:
    ENDPOINTS = "endpoints"
    EVENTS = "events"


class ModelEndpointMetadata(BaseModel):
    project: Optional[str]
    labels: Optional[dict]
    uid: Optional[str]

    class Config:
        extra = Extra.allow


class ModelEndpointSpec(ObjectSpec):
    function_uri: Optional[str]  # <project_name>/<function_name>:<tag>
    model: Optional[str]  # <model_name>:<version>
    model_class: Optional[str]
    model_uri: Optional[str]
    feature_names: Optional[List[str]]
    label_names: Optional[List[str]]
    stream_path: Optional[str]
    algorithm: Optional[str]
    monitor_configuration: Optional[dict]
    active: Optional[bool]


class Metric(BaseModel):
    name: str
    values: List[Tuple[str, float]]


class Histogram(BaseModel):
    buckets: List[float]
    counts: List[int]


class FeatureValues(BaseModel):
    min: float
    mean: float
    max: float
    histogram: Histogram

    @classmethod
    def from_dict(cls, stats: Optional[dict]):
        if stats:
            return FeatureValues(
                min=stats["min"],
                mean=stats["mean"],
                max=stats["max"],
                histogram=Histogram(buckets=stats["hist"][1], counts=stats["hist"][0]),
            )
        else:
            return None


class Features(BaseModel):
    name: str
    weight: float
    expected: Optional[FeatureValues]
    actual: Optional[FeatureValues]

    @classmethod
    def new(
        cls,
        feature_name: str,
        feature_stats: Optional[dict],
        current_stats: Optional[dict],
    ):
        return cls(
            name=feature_name,
            weight=-1.0,
            expected=FeatureValues.from_dict(feature_stats),
            actual=FeatureValues.from_dict(current_stats),
        )


class ModelEndpointStatus(ObjectStatus):
    feature_stats: Optional[dict]
    current_stats: Optional[dict]
    first_request: Optional[str]
    last_request: Optional[str]
    accuracy: Optional[float]
    error_count: Optional[int]
    drift_status: Optional[str]
    drift_measures: Optional[dict]
    metrics: Optional[Dict[str, Metric]]
    features: Optional[List[Features]]
    children: Optional[List[str]]
    children_uids: Optional[List[str]]
    endpoint_type: Optional[EndpointType]

    class Config:
        extra = Extra.allow


class ModelEndpoint(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ModelEndpointMetadata
    spec: ModelEndpointSpec
    status: ModelEndpointStatus

    class Config:
        extra = Extra.allow

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.metadata.uid is None:
            uid = create_model_endpoint_id(
                function_uri=self.spec.function_uri, versioned_model=self.spec.model,
            )
            self.metadata.uid = str(uid)


class ModelEndpointList(BaseModel):
    endpoints: List[ModelEndpoint]


class GrafanaColumn(BaseModel):
    text: str
    type: str


class GrafanaNumberColumn(GrafanaColumn):
    text: str
    type: str = "number"


class GrafanaStringColumn(GrafanaColumn):
    text: str
    type: str = "string"


class GrafanaTable(BaseModel):
    columns: List[GrafanaColumn]
    rows: List[List[Optional[Union[float, int, str]]]] = []
    type: str = "table"

    def add_row(self, *args):
        self.rows.append(list(args))


class GrafanaDataPoint(BaseModel):
    value: float
    timestamp: int  # Unix timestamp in milliseconds


class GrafanaTimeSeriesTarget(BaseModel):
    target: str
    datapoints: List[Tuple[float, int]] = []

    def add_data_point(self, data_point: GrafanaDataPoint):
        self.datapoints.append((data_point.value, data_point.timestamp))
