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

import mlrun.common
from mlrun.common.schemas.model_monitoring import (
    EndpointUID,
    FunctionURI,
    VersionedModel,
)


def create_model_endpoint_uid(function_uri: str, versioned_model: str):
    function_uri = FunctionURI.from_string(function_uri)
    versioned_model = VersionedModel.from_string(versioned_model)

    if (
        not function_uri.project
        or not function_uri.function
        or not versioned_model.model
    ):
        raise ValueError("Both function_uri and versioned_model have to be initialized")

    uid = EndpointUID(
        function_uri.project,
        function_uri.function,
        function_uri.tag,
        function_uri.hash_key,
        versioned_model.model,
        versioned_model.version,
    )

    return uid


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = mlrun.platforms.iguazio.parse_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def parse_monitoring_stream_path(stream_uri: str, project: str):
    if stream_uri.startswith("kafka://"):
        if "?topic" in stream_uri:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Custom kafka topic is not allowed"
            )
        # Add topic to stream kafka uri
        stream_uri += f"?topic=monitoring_stream_{project}"

    elif stream_uri.startswith("v3io://") and mlrun.mlconf.is_ce_mode():
        # V3IO is not supported in CE mode, generating a default http stream path
        stream_uri = mlrun.mlconf.model_endpoint_monitoring.default_http_sink
    return stream_uri
