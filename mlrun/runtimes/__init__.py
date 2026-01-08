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
from .constants import RuntimeKindsBase
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


class RuntimeKinds(RuntimeKindsBase):
    """
    Public `RuntimeKinds` API exposed from `mlrun.runtimes`.

    We intentionally keep `mlrun/runtimes/constants.py` free of imports from nuclio runtimes
    to avoid import cycles.
    Nuclio resolver helpers therefore live here, where nuclio runtime classes are already imported by this package.
    """

    @staticmethod
    def resolve_nuclio_runtime(kind: str, sub_kind: str):
        kind = kind.split(":")[0]
        if kind not in RuntimeKinds.nuclio_runtimes():
            raise ValueError(
                f"Kind {kind} is not a nuclio runtime, "
                f"available runtimes are {RuntimeKinds.nuclio_runtimes()}"
            )

        # These names are imported at module level below; referenced at call-time (no imports here).
        if sub_kind == serving_subkind:
            return ServingRuntime()

        if kind == RuntimeKinds.application:
            return ApplicationRuntime()

        runtime = RemoteRuntime()
        runtime.spec.function_kind = sub_kind
        return runtime

    @staticmethod
    def resolve_nuclio_sub_kind(kind: str):
        is_nuclio = kind.startswith("nuclio")
        sub_kind = kind[kind.find(":") + 1 :] if is_nuclio and ":" in kind else None
        if kind == RuntimeKinds.serving:
            is_nuclio = True
            sub_kind = serving_subkind
        elif kind == RuntimeKinds.application:
            is_nuclio = True
        return is_nuclio, sub_kind


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
