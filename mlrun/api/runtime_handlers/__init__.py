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
from mlrun.api.runtime_handlers.base import BaseRuntimeHandler
from mlrun.api.runtime_handlers.daskjob import DaskRuntimeHandler, get_dask_resource
from mlrun.api.runtime_handlers.kubejob import (
    DatabricksRuntimeHandler,
    KubeRuntimeHandler,
)
from mlrun.api.runtime_handlers.mpijob import (
    MpiRuntimeHandlerContainer,
    resolve_mpijob_crd_version,
)
from mlrun.api.runtime_handlers.remotesparkjob import RemoteSparkRuntimeHandler
from mlrun.api.runtime_handlers.sparkjob import SparkRuntimeHandler
from mlrun.runtimes import RuntimeKinds

runtime_handler_instances_cache = {}
runtime_resources_map = {RuntimeKinds.dask: get_dask_resource()}


def get_runtime_handler(kind: str) -> BaseRuntimeHandler:
    global runtime_handler_instances_cache
    if kind == RuntimeKinds.mpijob:
        mpijob_crd_version = resolve_mpijob_crd_version()
        runtime_handler_class = MpiRuntimeHandlerContainer.handler_selector()
        if not runtime_handler_instances_cache.setdefault(RuntimeKinds.mpijob, {}).get(
            mpijob_crd_version
        ):
            runtime_handler_instances_cache[RuntimeKinds.mpijob][
                mpijob_crd_version
            ] = runtime_handler_class()
        return runtime_handler_instances_cache[RuntimeKinds.mpijob][mpijob_crd_version]

    kind_runtime_handler_map = {
        RuntimeKinds.dask: DaskRuntimeHandler,
        RuntimeKinds.spark: SparkRuntimeHandler,
        RuntimeKinds.remotespark: RemoteSparkRuntimeHandler,
        RuntimeKinds.job: KubeRuntimeHandler,
        RuntimeKinds.databricks: DatabricksRuntimeHandler,
    }
    runtime_handler_class = kind_runtime_handler_map[kind]
    if not runtime_handler_instances_cache.get(kind):
        runtime_handler_instances_cache[kind] = runtime_handler_class()
    return runtime_handler_instances_cache[kind]
