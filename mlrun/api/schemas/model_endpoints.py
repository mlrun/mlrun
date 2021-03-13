from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic.main import Extra

from mlrun.api.schemas.object import ObjectKind, ObjectSpec, ObjectStatus


class ModelEndpointMetadata(BaseModel):
    project: Optional[str]
    tag: Optional[str]
    labels: Optional[dict]
    uid: Optional[str]

    class Config:
        extra = Extra.allow


class ModelEndpointSpec(ObjectSpec):
    model: Optional[str]
    function: Optional[str]
    model_class: Optional[str]
    model_artifact: Optional[str]
    stream_path: Optional[str]
    monitor_configuration: Optional[dict]


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
    def new(cls, stats: Optional[dict]):
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
            expected=FeatureValues.new(feature_stats),
            actual=FeatureValues.new(current_stats),
        )


class ModelEndpointStatus(BaseModel):
    state: Optional[str]
    active: Optional[bool]
    feature_names: Optional[List[str]]
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
        self.metadata.uid = self.create_endpoint_id(
            project=self.metadata.project,
            function=self.spec.function,
            model=self.spec.model,
            tag=self.metadata.tag,
        )

    @staticmethod
    def create_endpoint_id(
        project: str, function: str, model: str, tag: Optional[str] = None
    ):
        if not project or not function or not model:
            raise ValueError("project, function, model must be initialized")

        endpoint_unique_string = (
            f"{function}_{model}_{tag}" if tag else f"{function}_{model}"
        )

        md5_str = md5(endpoint_unique_string.encode("utf-8")).hexdigest()
        return f"{project}.{md5_str}"


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
    rows: List[List[Optional[Union[int, float, str]]]] = []
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


class GrafanaTimeSeries(BaseModel):
    target_data_points: List[GrafanaTimeSeriesTarget] = []
