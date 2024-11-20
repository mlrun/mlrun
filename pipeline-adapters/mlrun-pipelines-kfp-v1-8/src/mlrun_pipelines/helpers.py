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

import typing

from mlrun.config import config
from mlrun_pipelines.imports import PipelineConf


def new_pipe_metadata(
    artifact_path: typing.Optional[str] = None,
    cleanup_ttl: typing.Optional[int] = None,
    op_transformers: typing.Optional[list[typing.Callable]] = None,
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
    if artifact_path:
        conf.add_op_transformer(_set_artifact_path)
    if op_transformers:
        for op_transformer in op_transformers:
            conf.add_op_transformer(op_transformer)
    return conf
