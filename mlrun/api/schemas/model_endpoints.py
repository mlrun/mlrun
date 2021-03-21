from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic.main import Extra

from mlrun.api.schemas.object import ObjectKind, ObjectSpec, ObjectStatus
from mlrun.utils import parse_versioned_object_uri


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
    stream_path: Optional[str]
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
            uid = self.create_endpoint_id(
                function_uri=self.spec.function_uri, versioned_model=self.spec.model,
            )
            self.metadata.uid = str(uid)

    @staticmethod
    def create_endpoint_id(function_uri: str, versioned_model: str):
        function_uri = ModelEndpoint.FunctionURI.from_string(function_uri)
        versioned_model = ModelEndpoint.VersionedModel.from_string(versioned_model)

        if (
            not function_uri.project
            or not function_uri.function
            or not versioned_model.model
        ):
            raise ValueError(
                "Both function_uri and versioned_model have to be initialized"
            )

        uid = ModelEndpoint.EndpointUID(
            function_uri.project,
            function_uri.function,
            function_uri.tag,
            function_uri.hash_key,
            versioned_model.model,
            versioned_model.version,
        )

        return uid

    @dataclass
    class FunctionURI:
        project: str
        function: str
        tag: Optional[str] = None
        hash_key: Optional[str] = None

        @classmethod
        def from_string(cls, function_uri):
            project, uri, tag, hash_key = parse_versioned_object_uri(function_uri)
            return cls(
                project=project,
                function=uri,
                tag=tag or None,
                hash_key=hash_key or None,
            )

    @dataclass
    class VersionedModel:
        model: str
        version: Optional[str]

        @classmethod
        def from_string(cls, model):
            try:
                model, version = model.split(":")
            except ValueError:
                model, version = model, None

            return cls(model, version)

    @dataclass
    class EndpointUID:
        project: str
        function: str
        function_tag: str
        function_hash_key: str
        model: str
        model_version: str
        uid: Optional[str] = None

        def __post_init__(self):
            function_ref = f"{self.function}_{self.function_tag or self.function_hash_key or 'N/A'}"
            versioned_model = f"{self.model}_{self.model_version or 'N/A'}"
            unique_string = f"{self.project}_{function_ref}_{versioned_model}"
            self.uid = sha1(unique_string.encode("utf-8")).hexdigest()

        def __str__(self):
            return self.uid


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
