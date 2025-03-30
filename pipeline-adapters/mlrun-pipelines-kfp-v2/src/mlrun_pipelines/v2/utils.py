# Copyright 2024 Iguazio
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
import logging
import tempfile
import typing

import semver

import mlrun_pipelines.client
import mlrun_pipelines.imports


def compile_pipeline(
    pipeline, pipe_file: typing.Optional[str] = None, type_check: bool = False, **kwargs
):
    if not pipe_file:
        pipe_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    mlrun_pipelines.imports.Compiler().compile(
        pipeline, pipe_file, type_check=type_check
    )
    return pipe_file


def get_client(
    url: typing.Optional[str] = None,
    namespace: typing.Optional[str] = None,
) -> mlrun_pipelines.client.Client:
    if url or namespace:
        return mlrun_pipelines.client.Client(host=url, namespace=namespace)
    return mlrun_pipelines.client.Client()


def get_relevant_client_for_kfp_spec(
    pipeline_package_path: str,
    host: typing.Optional[str] = None,
    namespace: str = "mlrun",
) -> mlrun_pipelines.common.client.AbstractClient:
    pipeline_dict = mlrun_pipelines.v2.client.extract_pipeline_yaml(
        pipeline_package_path
    )
    sdk_version = (
        pipeline_dict.get("metadata", {})
        .get("annotations", {})
        .get("pipelines.kubeflow.org/kfp_sdk_version")
    )
    if sdk_version:
        try:
            parsed_version = semver.parse_version_info(sdk_version)
        except (TypeError, ValueError):
            logging.exception("Failed to parse pipeline SDK version annotation.")
            raise
        if parsed_version.major == 1:
            client_klass = mlrun_pipelines.v1.client.Client
        elif parsed_version.major == 2:
            client_klass = mlrun_pipelines.v2.client.Client
        else:
            raise ValueError(f"Unsupported SDK version: {sdk_version}")
        return client_klass(
            host=host,
            namespace=namespace,
        )
    else:
        raise ValueError("Pipeline does not contain SDK version annotation.")
