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

__all__ = [
    "BaseRuntime",
    "KubejobRuntime",
    "LocalRuntime",
    "HandlerRuntime",
    "RemoteRuntime",
    "ServingRuntime",
    "DaskCluster",
    "RemoteSparkRuntime",
    "Spark3Runtime",
    "DatabricksRuntime",
    "KubeResource",
    "ApplicationRuntime",
    "MpiRuntimeV1",
    "RuntimeKinds",
]

import typing

from mlrun.runtimes.utils import resolve_spark_operator_version

from ..common.runtimes.constants import MPIJobCRDVersions
from .base import BaseRuntime, RunError, RuntimeClassMode  # noqa
from .constants import RuntimeKinds
from .daskjob import DaskCluster  # noqa
from .databricks_job.databricks_runtime import DatabricksRuntime
from .kubejob import KubejobRuntime, KubeResource  # noqa
from .local import HandlerRuntime, LocalRuntime  # noqa
from .mpijob import MpiRuntimeV1  # noqa
from .nuclio import (
    RemoteRuntime,
    ServingRuntime,
    new_v2_model_server,
    nuclio_init_hook,
)
from .nuclio.api_gateway import APIGateway
from .nuclio.application import ApplicationRuntime
from .nuclio.serving import serving_subkind
from .remotesparkjob import RemoteSparkRuntime
from .sparkjob import Spark3Runtime

# for legacy imports (MLModelServer moved from here to /serving)
from ..serving import MLModelServer, new_v1_model_server  # noqa isort: skip


def new_model_server(
    name,
    model_class: str,
    models: typing.Optional[dict] = None,
    filename="",
    protocol="",
    image="",
    endpoint="",
    explainer=False,
    workers=8,
    canary=None,
    handler=None,
):
    if protocol:
        return new_v2_model_server(
            name,
            model_class,
            models=models,
            filename=filename,
            protocol=protocol,
            image=image,
            endpoint=endpoint,
            workers=workers,
            canary=canary,
        )
    else:
        return new_v1_model_server(
            name,
            model_class,
            models=models,
            filename=filename,
            protocol=protocol,
            image=image,
            endpoint=endpoint,
            workers=workers,
            canary=canary,
        )


def get_runtime_class(kind: str):
    if kind == RuntimeKinds.mpijob:
        return MpiRuntimeV1

    if kind == RuntimeKinds.spark:
        return Spark3Runtime

    kind_runtime_map = {
        RuntimeKinds.remote: RemoteRuntime,
        RuntimeKinds.nuclio: RemoteRuntime,
        RuntimeKinds.serving: ServingRuntime,
        RuntimeKinds.dask: DaskCluster,
        RuntimeKinds.job: KubejobRuntime,
        RuntimeKinds.local: LocalRuntime,
        RuntimeKinds.remotespark: RemoteSparkRuntime,
        RuntimeKinds.databricks: DatabricksRuntime,
        RuntimeKinds.application: ApplicationRuntime,
    }

    return kind_runtime_map[kind]
