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

import enum
import json
from datetime import datetime
from typing import Any, NamedTuple, Optional, TypeVar

from pydantic import BaseModel, Extra, Field, constr, validator

# TODO: remove the unused import below after `mlrun.datastore` and `mlrun.utils` usage is removed.
# At the moment `make lint` fails if this is removed.
import mlrun.common.model_monitoring

from ..object import ObjectKind, ObjectSpec, ObjectStatus
from .constants import (
    FQN_REGEX,
    MODEL_ENDPOINT_ID_PATTERN,
    PROJECT_PATTERN,
    EndpointType,
    EventFieldType,
    EventKeyMetrics,
    EventLiveStats,
    ModelEndpointMonitoringMetricType,
    ModelMonitoringMode,
    ResultKindApp,
    ResultStatusApp,
)

Model = TypeVar("Model", bound=BaseModel)


class ModelMonitoringStoreKinds:
    # TODO: do changes in examples & demos In 1.5.0 remove
    ENDPOINTS = "endpoints"
    EVENTS = "events"


class ModelEndpointMetadata(BaseModel):
    project: constr(regex=PROJECT_PATTERN)
    uid: constr(regex=MODEL_ENDPOINT_ID_PATTERN)
    labels: Optional[dict] = {}

    class Config:
        extra = Extra.allow

    @classmethod
    def from_flat_dict(cls, endpoint_dict: dict, json_parse_values: list = None):
        """Create a `ModelEndpointMetadata` object from an endpoint dictionary

        :param endpoint_dict:     Model endpoint dictionary.
        :param json_parse_values: List of dictionary keys with a JSON string value that will be parsed into a
                                  dictionary using json.loads().
        """
        if json_parse_values is None:
            json_parse_values = [EventFieldType.LABELS]

        return _mapping_attributes(
            model_class=cls,
            flattened_dictionary=endpoint_dict,
            json_parse_values=json_parse_values,
        )


class ModelEndpointSpec(ObjectSpec):
    function_uri: Optional[str] = ""  # <project_name>/<function_name>:<tag>
    model: Optional[str] = ""  # <model_name>:<version>
    model_class: Optional[str] = ""
    model_uri: Optional[str] = ""
    feature_names: Optional[list[str]] = []
    label_names: Optional[list[str]] = []
    stream_path: Optional[str] = ""
    algorithm: Optional[str] = ""
    monitor_configuration: Optional[dict] = {}
    active: Optional[bool] = True
    monitoring_mode: Optional[ModelMonitoringMode] = ModelMonitoringMode.disabled.value

    @classmethod
    def from_flat_dict(cls, endpoint_dict: dict, json_parse_values: list = None):
        """Create a `ModelEndpointSpec` object from an endpoint dictionary

        :param endpoint_dict:     Model endpoint dictionary.
        :param json_parse_values: List of dictionary keys with a JSON string value that will be parsed into a
                                  dictionary using json.loads().
        """
        if json_parse_values is None:
            json_parse_values = [
                EventFieldType.FEATURE_NAMES,
                EventFieldType.LABEL_NAMES,
                EventFieldType.MONITOR_CONFIGURATION,
            ]
        return _mapping_attributes(
            model_class=cls,
            flattened_dictionary=endpoint_dict,
            json_parse_values=json_parse_values,
        )

    @validator("model_uri")
    @classmethod
    def validate_model_uri(cls, model_uri):
        """Validate that the model uri includes the required prefix"""
        prefix, uri = mlrun.datastore.parse_store_uri(model_uri)
        if prefix and prefix != mlrun.utils.helpers.StorePrefix.Model:
            return mlrun.datastore.get_store_uri(
                mlrun.utils.helpers.StorePrefix.Model, uri
            )
        return model_uri


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


class ModelEndpointStatus(ObjectStatus):
    feature_stats: Optional[dict] = {}
    current_stats: Optional[dict] = {}
    first_request: Optional[str] = ""
    last_request: Optional[str] = ""
    error_count: Optional[int] = 0
    drift_status: Optional[str] = ""
    drift_measures: Optional[dict] = {}
    metrics: Optional[dict[str, dict[str, Any]]] = {
        EventKeyMetrics.GENERIC: {
            EventLiveStats.LATENCY_AVG_1H: 0,
            EventLiveStats.PREDICTIONS_PER_SECOND: 0,
        }
    }
    features: Optional[list[Features]] = []
    children: Optional[list[str]] = []
    children_uids: Optional[list[str]] = []
    endpoint_type: Optional[EndpointType] = EndpointType.NODE_EP
    monitoring_feature_set_uri: Optional[str] = ""
    state: Optional[str] = ""

    class Config:
        extra = Extra.allow

    @classmethod
    def from_flat_dict(cls, endpoint_dict: dict, json_parse_values: list = None):
        """Create a `ModelEndpointStatus` object from an endpoint dictionary

        :param endpoint_dict:     Model endpoint dictionary.
        :param json_parse_values: List of dictionary keys with a JSON string value that will be parsed into a
                                  dictionary using json.loads().
        """
        if json_parse_values is None:
            json_parse_values = [
                EventFieldType.FEATURE_STATS,
                EventFieldType.CURRENT_STATS,
                EventFieldType.DRIFT_MEASURES,
                EventFieldType.METRICS,
                EventFieldType.CHILDREN,
                EventFieldType.CHILDREN_UIDS,
                EventFieldType.ENDPOINT_TYPE,
            ]
        return _mapping_attributes(
            model_class=cls,
            flattened_dictionary=endpoint_dict,
            json_parse_values=json_parse_values,
        )


class ModelEndpoint(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ModelEndpointMetadata
    spec: ModelEndpointSpec = ModelEndpointSpec()
    status: ModelEndpointStatus = ModelEndpointStatus()

    class Config:
        extra = Extra.allow

    def flat_dict(self):
        """Generate a flattened `ModelEndpoint` dictionary. The flattened dictionary result is important for storing
        the model endpoint object in the database.

        :return: Flattened `ModelEndpoint` dictionary.
        """
        # Convert the ModelEndpoint object into a dictionary using BaseModel dict() function
        # In addition, remove the BaseModel kind as it is not required by the DB schema
        model_endpoint_dictionary = self.dict(exclude={"kind"})

        # Initialize a flattened dictionary that will be filled with the model endpoint dictionary attributes
        flatten_dict = {}
        for k_object in model_endpoint_dictionary:
            for key in model_endpoint_dictionary[k_object]:
                # Extract the value of the current field
                current_value = model_endpoint_dictionary[k_object][key]

                # If the value is not from type str or bool (e.g. dict), convert it into a JSON string
                # for matching the database required format
                if not isinstance(current_value, (str, bool, int)) or isinstance(
                    current_value, enum.IntEnum
                ):
                    flatten_dict[key] = json.dumps(current_value)
                else:
                    flatten_dict[key] = current_value

        if EventFieldType.METRICS not in flatten_dict:
            # Initialize metrics dictionary
            flatten_dict[EventFieldType.METRICS] = {
                EventKeyMetrics.GENERIC: {
                    EventLiveStats.LATENCY_AVG_1H: 0,
                    EventLiveStats.PREDICTIONS_PER_SECOND: 0,
                }
            }

        # Remove the features from the dictionary as this field will be filled only within the feature analysis process
        flatten_dict.pop(EventFieldType.FEATURES, None)
        return flatten_dict

    @classmethod
    def from_flat_dict(cls, endpoint_dict: dict) -> "ModelEndpoint":
        """Create a `ModelEndpoint` object from an endpoint flattened dictionary. Because the provided dictionary
        is flattened, we pass it as is to the subclasses without splitting the keys into spec, metadata, and status.

        :param endpoint_dict:     Model endpoint dictionary.
        """

        return cls(
            metadata=ModelEndpointMetadata.from_flat_dict(endpoint_dict=endpoint_dict),
            spec=ModelEndpointSpec.from_flat_dict(endpoint_dict=endpoint_dict),
            status=ModelEndpointStatus.from_flat_dict(endpoint_dict=endpoint_dict),
        )


class ModelEndpointList(BaseModel):
    endpoints: list[ModelEndpoint] = []


class ModelEndpointMonitoringMetric(BaseModel):
    project: str
    app: str
    type: ModelEndpointMonitoringMetricType
    name: str
    full_name: str


def _compose_full_name(
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
    model_class: type[Model], flattened_dictionary: dict, json_parse_values: list
) -> Model:
    """Generate a `BaseModel` object with the provided dictionary attributes.

    :param model_class:          `BaseModel` class (e.g. `ModelEndpointMetadata`).
    :param flattened_dictionary: Flattened dictionary that contains the model endpoint attributes.
    :param json_parse_values:    List of dictionary keys with a JSON string value that will be parsed into a
                                 dictionary using json.loads().
    """
    # Get the fields of the provided base model object. These fields will be used to filter to relevent keys
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
            else:
                dict_to_parse[field_key] = flattened_dictionary[field_key]

    return model_class.parse_obj(dict_to_parse)


def _json_loads_if_not_none(field: Any) -> Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )
