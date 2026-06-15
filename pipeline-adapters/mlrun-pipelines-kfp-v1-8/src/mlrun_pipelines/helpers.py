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

import typing

from mlrun.config import config
from mlrun.utils import logger
from mlrun_pipelines.imports import PipelineConf


def new_pipe_metadata(
    artifact_path: str | None = None,
    cleanup_ttl: int | None = None,
    op_transformers: list[typing.Callable] | None = None,
):
    def _set_artifact_path(task):
        from kubernetes import client as k8s_client

        task.add_env_variable(
            k8s_client.V1EnvVar(name="MLRUN_ARTIFACT_PATH", value=artifact_path)
        )
        return task

    conf = PipelineConf()
    cleanup_ttl = cleanup_ttl or int(config.kfp_ttl)

    if cleanup_ttl:
        conf.set_ttl_seconds_after_finished(cleanup_ttl)

    try:
        default_timeout = int(config.kfp_default_workflow_timeout)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid kfp_default_workflow_timeout config value, workflow timeout will not be set",
            value=config.kfp_default_workflow_timeout,
        )
        default_timeout = 0
    if default_timeout > 0:
        conf.set_timeout(default_timeout)

    if artifact_path:
        conf.add_op_transformer(_set_artifact_path)
    if op_transformers:
        for op_transformer in op_transformers:
            conf.add_op_transformer(op_transformer)
    return conf
