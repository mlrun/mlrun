# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import json
from datetime import datetime
from typing import Any, NamedTuple, Optional, TypeVar

from pydantic.v1 import BaseModel, Field, constr

# TODO: remove the unused import below after `mlrun.datastore` and `mlrun.utils` usage is removed.
# At the moment `make lint` fails if this is removed.
from ..object import ObjectKind, ObjectMetadata, ObjectSpec, ObjectStatus
from . import ModelEndpointSchema
from .constants import (
    FQN_REGEX,
    MODEL_ENDPOINT_ID_PATTERN,
    PROJECT_PATTERN,
    EndpointType,
    ModelEndpointMonitoringMetricType,
    ModelMonitoringMode,
    ResultKindApp,
    ResultStatusApp,
)

Model = TypeVar("Model", bound=BaseModel)


class Histogram(BaseModel):
    buckets: list[float]
    counts: list[int]


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


class ModelEndpointParser(abc.ABC, BaseModel):
    @classmethod
    def json_parse_values(cls) -> list[str]:
        return []

    @classmethod
    def from_flat_dict(
        cls,
        endpoint_dict: dict,
        json_parse_values: Optional[list] = None,
        validate: bool = True,
    ) -> "ModelEndpointParser":
        """Create a `ModelEndpointParser` object from an endpoint dictionary

        :param endpoint_dict:     Model endpoint dictionary.
        :param json_parse_values: List of dictionary keys with a JSON string value that will be parsed into a
                                  dictionary using json.loads().
        :param validate:          Whether to validate the flattened dictionary.
                                  Skip validation to optimize performance when it is safe to do so.
        """
        if json_parse_values is None:
            json_parse_values = cls.json_parse_values()

        return _mapping_attributes(
            model_class=cls,
            flattened_dictionary=endpoint_dict,
            json_parse_values=json_parse_values,
            validate=validate,
        )


class ModelEndpointMetadata(ObjectMetadata, ModelEndpointParser):
    project: constr(regex=PROJECT_PATTERN)
    endpoint_type: EndpointType = EndpointType.NODE_EP
    uid: Optional[constr(regex=MODEL_ENDPOINT_ID_PATTERN)]

    @classmethod
    def mutable_fields(cls):
        return ["labels"]


class ModelEndpointSpec(ObjectSpec, ModelEndpointParser):
    model_class: Optional[str] = ""
    function_name: Optional[str] = ""
    function_tag: Optional[str] = ""
    model_path: Optional[str] = ""
    model_name: Optional[str] = ""
    model_tags: Optional[list[str]] = []
    _model_id: Optional[int] = ""
    feature_names: Optional[list[str]] = []
    label_names: Optional[list[str]] = []
    feature_stats: Optional[dict] = {}
    function_uri: Optional[str] = ""  # <project_name>/<function_hash>
    model_uri: Optional[str] = ""
    children: Optional[list[str]] = []
    children_uids: Optional[list[str]] = []
    monitoring_feature_set_uri: Optional[str] = ""

    @classmethod
    def mutable_fields(cls):
        return [
            "model_path",
            "model_class",
            "feature_names",
            "label_names",
            "children",
            "children_uids",
        ]


class ModelEndpointStatus(ObjectStatus, ModelEndpointParser):
    state: Optional[str] = "unknown"  # will be updated according to the function state
    first_request: Optional[datetime] = None
    monitoring_mode: Optional[ModelMonitoringMode] = ModelMonitoringMode.disabled
    sampling_percentage: Optional[float] = 100

    # operative
    last_request: Optional[datetime] = None
    result_status: Optional[int] = -1
    avg_latency: Optional[float] = None
    error_count: Optional[int] = 0
    current_stats: Optional[dict] = {}
    current_stats_timestamp: Optional[datetime] = None
    drift_measures: Optional[dict] = {}
    drift_measures_timestamp: Optional[datetime] = None

    @classmethod
    def mutable_fields(cls):
        return [
            "monitoring_mode",
            "first_request",
            "last_request",
            "sampling_percentage",
        ]


class ModelEndpoint(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ModelEndpointMetadata
    spec: ModelEndpointSpec
    status: ModelEndpointStatus

    @classmethod
    def mutable_fields(cls):
        return (
            ModelEndpointMetadata.mutable_fields()
            + ModelEndpointSpec.mutable_fields()
            + ModelEndpointStatus.mutable_fields()
        )

    def flat_dict(self) -> dict[str, Any]:
        """Generate a flattened `ModelEndpoint` dictionary. The flattened dictionary result is important for storing
        the model endpoint object in the database.

        :return: Flattened `ModelEndpoint` dictionary.
        """
        # Convert the ModelEndpoint object into a dictionary using BaseModel dict() function
        # In addition, remove the BaseModel kind as it is not required by the DB schema

        model_endpoint_dictionary = self.dict(exclude={"kind"})
        exclude = {
            "tag",
            ModelEndpointSchema.FEATURE_STATS,
            ModelEndpointSchema.CURRENT_STATS,
            ModelEndpointSchema.DRIFT_MEASURES,
            ModelEndpointSchema.FUNCTION_URI,
        }
        # Initialize a flattened dictionary that will be filled with the model endpoint dictionary attributes
        flatten_dict = {}
        for k_object in model_endpoint_dictionary:
            for key in model_endpoint_dictionary[k_object]:
                if key not in exclude:
                    # Extract the value of the current field
                    flatten_dict[key] = model_endpoint_dictionary[k_object][key]

        return flatten_dict

    @classmethod
    def from_flat_dict(
        cls, endpoint_dict: dict, validate: bool = True
    ) -> "ModelEndpoint":
        """Create a `ModelEndpoint` object from an endpoint flattened dictionary. Because the provided dictionary
        is flattened, we pass it as is to the subclasses without splitting the keys into spec, metadata, and status.

        :param endpoint_dict:     Model endpoint dictionary.
        :param validate:          Whether to validate the flattened dictionary.
                                  Skip validation to optimize performance when it is safe to do so.
        """

        return cls(
            metadata=ModelEndpointMetadata.from_flat_dict(
                endpoint_dict=endpoint_dict, validate=validate
            ),
            spec=ModelEndpointSpec.from_flat_dict(
                endpoint_dict=endpoint_dict, validate=validate
            ),
            status=ModelEndpointStatus.from_flat_dict(
                endpoint_dict=endpoint_dict, validate=validate
            ),
        )

    def get(self, field, default=None):
        return (
            getattr(self.metadata, field, None)
            or getattr(self.spec, field, None)
            or getattr(self.status, field, None)
            or default
        )


class ModelEndpointList(BaseModel):
    endpoints: list[ModelEndpoint]


class ModelEndpointMonitoringMetric(BaseModel):
    project: str
    app: str
    type: ModelEndpointMonitoringMetricType
    name: str
    full_name: Optional[str] = None
    kind: Optional[ResultKindApp] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.full_name = compose_full_name(
            project=self.project, app=self.app, name=self.name, type=self.type
        )


def compose_full_name(
    *,
    project: str,
    app: str,
    name: str,
    type: ModelEndpointMonitoringMetricType = ModelEndpointMonitoringMetricType.RESULT,
) -> str:
    return ".".join([project, app, type, name])


def _parse_metric_fqn_to_monitoring_metric(fqn: str) -> ModelEndpointMonitoringMetric:
    match = FQN_REGEX.fullmatch(fqn)
    if match is None:
        raise ValueError("The fully qualified name is not in the expected format")
    return ModelEndpointMonitoringMetric.parse_obj(
        match.groupdict() | {"full_name": fqn}
    )


class _MetricPoint(NamedTuple):
    timestamp: datetime
    value: float


class _ResultPoint(NamedTuple):
    timestamp: datetime
    value: float
    status: ResultStatusApp
    extra_data: Optional[str] = ""


class _ModelEndpointMonitoringMetricValuesBase(BaseModel):
    full_name: str
    type: ModelEndpointMonitoringMetricType
    data: bool


class ModelEndpointMonitoringMetricValues(_ModelEndpointMonitoringMetricValuesBase):
    type: ModelEndpointMonitoringMetricType = ModelEndpointMonitoringMetricType.METRIC
    values: list[_MetricPoint]
    data: bool = True


class ModelEndpointMonitoringResultValues(_ModelEndpointMonitoringMetricValuesBase):
    type: ModelEndpointMonitoringMetricType = ModelEndpointMonitoringMetricType.RESULT
    result_kind: ResultKindApp
    values: list[_ResultPoint]
    data: bool = True


class ModelEndpointMonitoringMetricNoData(_ModelEndpointMonitoringMetricValuesBase):
    full_name: str
    type: ModelEndpointMonitoringMetricType
    data: bool = False


def _mapping_attributes(
    model_class: type[Model],
    flattened_dictionary: dict,
    json_parse_values: list,
    validate: bool = True,
) -> Model:
    """Generate a `BaseModel` object with the provided dictionary attributes.

    :param model_class:          `BaseModel` class (e.g. `ModelEndpointMetadata`).
    :param flattened_dictionary: Flattened dictionary that contains the model endpoint attributes.
    :param json_parse_values:    List of dictionary keys with a JSON string value that will be parsed into a
                                 dictionary using json.loads().
    :param validate:             Whether to validate the flattened dictionary.
                                 Skip validation to optimize performance when it is safe to do so.
    """
    # Get the fields of the provided base model object. These fields will be used to filter to relevant keys
    # from the flattened dictionary.
    wanted_keys = model_class.__fields__.keys()

    # Generate a filtered flattened dictionary that will be parsed into the BaseModel object
    dict_to_parse = {}
    for field_key in wanted_keys:
        if field_key in flattened_dictionary:
            if field_key in json_parse_values:
                # Parse the JSON value into a valid dictionary
                dict_to_parse[field_key] = _json_loads_if_not_none(
                    flattened_dictionary[field_key]
                )
            elif flattened_dictionary[field_key] != "null":
                dict_to_parse[field_key] = flattened_dictionary[field_key]
            else:
                dict_to_parse[field_key] = None

    if validate:
        return model_class.parse_obj(dict_to_parse)

    return model_class.construct(**dict_to_parse)


def _json_loads_if_not_none(field: Any) -> Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )
