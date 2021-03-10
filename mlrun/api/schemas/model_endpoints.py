from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic.main import Extra

from mlrun.api.schemas.object import ObjectKind, ObjectSpec, ObjectStatus
from mlrun.errors import MLRunInvalidArgumentError


class ModelEndpointMetadata(BaseModel):
    project: Optional[str]
    tag: Optional[str]
    labels: Optional[dict]
    model_artifact: Optional[str]
    stream_path: Optional[str]
    feature_stats: Optional[dict]
    feature_names: Optional[List[str]]
    monitor_configuration: Optional[dict]

    class Config:
        extra = Extra.allow


class ModelEndpointSpec(ObjectSpec):
    model: Optional[str]
    function: Optional[str]
    model_class: Optional[str]


class ModelEndpointStatus(ObjectStatus):
    active: Optional[bool]

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
        self.id = self.create_endpoint_id(
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

        endpoint_unique_string = f"{function}_{model}"
        if tag:
            endpoint_unique_string += f"_{tag}"

        md5_str = md5(endpoint_unique_string.encode("utf-8")).hexdigest()
        return f"{project}.{md5_str}"

    @classmethod
    def new(
        cls,
        project: str,
        model: str,
        function: str,
        tag: str,
        model_class: Optional[str] = None,
        labels: Optional[dict] = None,
        model_artifact: Optional[str] = None,
        stream_path: Optional[str] = None,
        feature_stats: Optional[dict] = None,
        feature_names: Optional[List[str]] = None,
        monitor_configuration: Optional[dict] = None,
        state: Optional[str] = None,
        active: bool = True,
    ) -> "ModelEndpoint":
        """ A constructor method for better usability

        Parameters for ModelEndpointMetadata
        :param project: The name of the project of which this endpoint belongs to (used for creating endpoint.id)
        :param tag: The tag/version of the model/function (used for creating endpoint.id)
        :param labels: key value pairs of user defined labels
        :param model_artifact: The path to the model artifact containing metadata about the features of the model
        :param stream_path: The path to the output stream of the model server
        :param feature_stats: A dictionary describing the model's features
        :param feature_names: A list of feature names
        :param monitor_configuration: A monitoring related key value configuration

        Parameters for ModelEndpointSpec
        :param model: The name of the model that is used in the serving function (used for creating endpoint.id)
        :param function: The name of the function that servers the model (used for creating endpoint.id)
        :param model_class: The class of the model

        Parameters for ModelEndpointStatus
        :param state: The current state of the endpoint
        :param active: The "activation" status of the endpoint - True for active / False for not active (default True)
        """

        if project is None and model is None and function is None:
            raise MLRunInvalidArgumentError(
                f"All of 'project[={project}]', 'model[={model}]', 'function[={function}]' must be properly initialized"
            )

        return cls(
            metadata=ModelEndpointMetadata(
                project=project,
                tag=tag,
                labels=labels or {},
                model_artifact=model_artifact,
                stream_path=stream_path,
                feature_stats=feature_stats,
                feature_names=feature_names,
                monitor_configuration=monitor_configuration,
            ),
            spec=ModelEndpointSpec(
                model=model, function=function, model_class=model_class,
            ),
            status=ModelEndpointStatus(state=state, active=active),
        )

    @property
    def project(self):
        return self.metadata.project

    @property
    def function(self):
        return self.spec.function

    @property
    def model(self):
        return self.spec.model

    @property
    def tag(self):
        return self.metadata.tag


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


class ModelEndpointState(BaseModel):
    endpoint: ModelEndpoint
    first_request: Optional[str]
    last_request: Optional[str]
    accuracy: Optional[float]
    error_count: Optional[int]
    drift_status: Optional[str]
    drift_measures: Optional[dict]
    current_stats: Optional[dict]
    metrics: Optional[Dict[str, Metric]]
    features: Optional[List[Features]]

    @classmethod
    def new(
        cls,
        endpoint: ModelEndpoint,
        first_request: Optional[str] = None,
        last_request: Optional[str] = None,
        accuracy: Optional[float] = None,
        error_count: Optional[int] = None,
        drift_status: Optional[str] = None,
        drift_measures: Optional[dict] = None,
        current_stats: Optional[dict] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        features: Optional[List[Features]] = None,
    ):
        """ A constructor method for better usability

        :param endpoint: The object representing a model endpoint
        :param first_request: The first request encountered for the given endpoint
        :param last_request: The last request encountered for the given endpoint
        :param accuracy: The accuracy of the model
        :param error_count: The error count of the model
        :param drift_status: The drift status of the model
        :param drift_measures: The drift measures of the model
        :param current_stats: The current statistics of the model
        :param metrics: The collected metrics of the model endpoint
        :param features: The incoming features of the model
        """
        return cls(
            endpoint=endpoint,
            first_request=first_request,
            last_request=last_request,
            accuracy=accuracy,
            error_count=error_count,
            drift_status=drift_status,
            drift_measures=drift_measures,
            current_stats=current_stats,
            metrics=metrics,
            features=features,
        )


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
