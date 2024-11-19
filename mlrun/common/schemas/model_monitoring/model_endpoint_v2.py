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
#
import abc
from typing import Optional

from pydantic.v1 import BaseModel, Field, constr

from ..object import ObjectKind, ObjectMetadata, ObjectSpec, ObjectStatus
from .constants import (
    PROJECT_PATTERN,
    EndpointType,
    ModelMonitoringMode,
)
from .model_endpoints import _mapping_attributes

# TODO : replace ModelEndpoint


class ModelEndpointParser(abc.ABC, BaseModel):
    @classmethod
    def json_parse_values(cls) -> list[str]:
        return []

    @classmethod
    def from_flat_dict(
        cls, endpoint_dict: dict, json_parse_values: Optional[list] = None
    ):
        """Create a `ModelEndpointMetadata` object from an endpoint dictionary

        :param endpoint_dict:     Model endpoint dictionary.
        :param json_parse_values: List of dictionary keys with a JSON string value that will be parsed into a
                                  dictionary using json.loads().
        """
        if json_parse_values is None:
            json_parse_values = cls.json_parse_values()

        return _mapping_attributes(
            model_class=cls,
            flattened_dictionary=endpoint_dict,
            json_parse_values=json_parse_values,
        )


class ModelEndpointV2Metadata(ObjectMetadata, ModelEndpointParser):
    project: constr(regex=PROJECT_PATTERN)
    endpoint_type: Optional[EndpointType] = EndpointType.NODE_EP.value


class ModelEndpointV2Spec(ObjectSpec, ModelEndpointParser):
    model_uid: Optional[str] = ""
    model_name: Optional[str] = ""
    model_tag: Optional[str] = ""
    model_class: Optional[str] = ""
    function_name: Optional[str] = ""
    function_uid: Optional[str] = ""
    feature_names: Optional[list[str]] = []
    label_names: Optional[list[str]] = []


class ModelEndpointV2Status(ObjectStatus, ModelEndpointParser):
    state: Optional[str] = "unknown"  # will be updated according to the function state
    first_request: Optional[str] = ""
    children: Optional[list[str]] = []
    children_uids: Optional[list[str]] = []
    monitoring_feature_set_uri: Optional[str] = ""
    monitoring_mode: Optional[ModelMonitoringMode] = ModelMonitoringMode.disabled.value
    function_uri: Optional[str] = ""  # <project_name>/<function_name>:<tag>
    model_uri: Optional[str] = ""

    # operative
    last_request: Optional[str] = ""
    drift_status: Optional[str] = ""
    avg_latency: Optional[float] = None
    error_count: Optional[int] = 0
    feature_stats: Optional[dict] = {}
    current_stats: Optional[dict] = {}
    drift_measures: Optional[dict] = {}


class ModelEndpointV2(BaseModel):
    kind: ObjectKind = Field(ObjectKind.model_endpoint, const=True)
    metadata: ModelEndpointV2Metadata
    spec: ModelEndpointV2Spec
    status: ModelEndpointV2Status

    def flat_dict(self, exclude: Optional[set] = None):
        """Generate a flattened `ModelEndpoint` dictionary. The flattened dictionary result is important for storing
        the model endpoint object in the database.

        :return: Flattened `ModelEndpoint` dictionary.
        """
        # Convert the ModelEndpoint object into a dictionary using BaseModel dict() function
        # In addition, remove the BaseModel kind as it is not required by the DB schema
        if exclude:
            exclude = exclude | {"kind", "tag"}
        else:
            exclude = {"kind", "tag"}
        model_endpoint_dictionary = self.dict(exclude=exclude)

        # Initialize a flattened dictionary that will be filled with the model endpoint dictionary attributes
        flatten_dict = {}
        for k_object in model_endpoint_dictionary:
            for key in model_endpoint_dictionary[k_object]:
                # Extract the value of the current field
                flatten_dict[key] = model_endpoint_dictionary[k_object][key]

        return flatten_dict

    @classmethod
    def from_flat_dict(cls, endpoint_dict: dict) -> "ModelEndpointV2":
        """Create a `ModelEndpoint` object from an endpoint flattened dictionary. Because the provided dictionary
        is flattened, we pass it as is to the subclasses without splitting the keys into spec, metadata, and status.

        :param endpoint_dict:     Model endpoint dictionary.
        """

        return cls(
            metadata=ModelEndpointV2Metadata.from_flat_dict(
                endpoint_dict=endpoint_dict
            ),
            spec=ModelEndpointV2Spec.from_flat_dict(endpoint_dict=endpoint_dict),
            status=ModelEndpointV2Status.from_flat_dict(endpoint_dict=endpoint_dict),
        )

    @classmethod
    def _operative_data(cls) -> set:
        return {
            "last_request",
            "drift_status",
            "avg_latency",
            "error_count",
            "feature_stats",
            "current_stats",
            "drift_measures",
            "function_uri",
            "model_uri",
        }
