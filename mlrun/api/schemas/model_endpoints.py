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
    model_artifact: Optional[str]
    stream_path: Optional[str]
    feature_stats: Optional[dict]
    feature_names: Optional[List[str]]

    class Config:
        extra = Extra.allow


class ModelEndpointSpec(ObjectSpec):
    model: Optional[str]
    function: Optional[str]
    model_class: Optional[str]


class ModelEndpointStatus(ObjectStatus):
    active: bool = True

    class Config:
        extra = Extra.allow


class ModelEndpoint(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ModelEndpointMetadata
    spec: ModelEndpointSpec
    status: ModelEndpointStatus
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

    @classmethod
    def new(
        cls,
        project: str,
        model: str,
        function: str,
        tag: str = "latest",
        model_class: Optional[str] = None,
        labels: Optional[dict] = None,
        model_artifact: Optional[str] = None,
        stream_path: Optional[str] = None,
        feature_stats: Optional[dict] = None,
        feature_names: Optional[List[str]] = None,
        state: Optional[str] = None,
        active: bool = True,
    ) -> "ModelEndpoint":
        """
        A constructor method for better usability

        Parameters for ModelEndpointMetadata
        :param project: The name of the project of which this endpoint belongs to (used for creating endpoint.id)
        :param tag: The tag/version of the model/function (used for creating endpoint.id)
        :param labels: key value pairs of user defined labels
        :param model_artifact: The path to the model artifact containing metadata about the features of the model
        :param stream_path: The path to the output stream of the model server
        :param feature_stats: A dictionary describing the model's features
        :param feature_names: A list of feature names

        Parameters for ModelEndpointSpec
        :param model: The name of the model that is used in the serving function (used for creating endpoint.id)
        :param function: The name of the function that servers the model (used for creating endpoint.id)
        :param model_class: The class of the model

        Parameters for ModelEndpointStatus
        :param state: The current state of the endpoint
        :param active: The "activation" status of the endpoint - True for active / False for not active (default True)
        """
        return ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=project,
                tag=tag,
                labels=labels or {},
                model_artifact=model_artifact,
                stream_path=stream_path,
                feature_stats=feature_stats,
                feature_names=feature_names
            ),
            spec=ModelEndpointSpec(
                model=model, function=function, model_class=model_class,
            ),
            status=ModelEndpointStatus(state=state, active=active),
        )


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
