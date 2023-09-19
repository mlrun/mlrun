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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

__all__ = [
    "BaseRuntime",
    "KubejobRuntime",
    "LocalRuntime",
    "HandlerRuntime",
    "RemoteRuntime",
    "ServingRuntime",
    "DaskCluster",
    "RemoteSparkRuntime",
    "DatabricksRuntime",
]

from mlrun.runtimes.utils import resolve_spark_operator_version

from .base import BaseRuntime, RunError, RuntimeClassMode  # noqa
from .constants import MPIJobCRDVersions
from .daskjob import DaskCluster  # noqa
from .databricks_job.databricks_runtime import DatabricksRuntime
from .function import RemoteRuntime
from .kubejob import KubejobRuntime  # noqa
from .local import HandlerRuntime, LocalRuntime  # noqa
from .mpijob import MpiRuntimeContainer, MpiRuntimeV1, MpiRuntimeV1Alpha1  # noqa
from .nuclio import nuclio_init_hook
from .remotesparkjob import RemoteSparkRuntime
from .serving import ServingRuntime, new_v2_model_server
from .sparkjob import Spark3Runtime

# for legacy imports (MLModelServer moved from here to /serving)
from ..serving import MLModelServer, new_v1_model_server  # noqa isort: skip


def new_model_server(
    name,
    model_class: str,
    models: dict = None,
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


class RuntimeKinds(object):
    remote = "remote"
    nuclio = "nuclio"
    dask = "dask"
    job = "job"
    spark = "spark"
    remotespark = "remote-spark"
    mpijob = "mpijob"
    serving = "serving"
    local = "local"
    handler = "handler"
    databricks = "databricks"

    @staticmethod
    def all():
        return [
            RuntimeKinds.remote,
            RuntimeKinds.nuclio,
            RuntimeKinds.serving,
            RuntimeKinds.dask,
            RuntimeKinds.job,
            RuntimeKinds.spark,
            RuntimeKinds.remotespark,
            RuntimeKinds.mpijob,
            RuntimeKinds.local,
            RuntimeKinds.databricks,
        ]

    @staticmethod
    def runtime_with_handlers():
        return [
            RuntimeKinds.dask,
            RuntimeKinds.job,
            RuntimeKinds.spark,
            RuntimeKinds.remotespark,
            RuntimeKinds.mpijob,
            RuntimeKinds.databricks,
        ]

    @staticmethod
    def abortable_runtimes():
        return [
            RuntimeKinds.job,
            RuntimeKinds.spark,
            RuntimeKinds.remotespark,
            RuntimeKinds.mpijob,
            RuntimeKinds.databricks,
            RuntimeKinds.local,
            RuntimeKinds.handler,
            "",
        ]

    @staticmethod
    def nuclio_runtimes():
        return [
            RuntimeKinds.remote,
            RuntimeKinds.nuclio,
            RuntimeKinds.serving,
        ]

    @staticmethod
    def local_runtimes():
        return [
            RuntimeKinds.local,
            RuntimeKinds.handler,
        ]

    @staticmethod
    def is_log_collectable_runtime(kind: str):
        """
        whether log collector can collect logs for that runtime
        :param kind: kind name
        :return: whether log collector can collect logs for that runtime
        """
        # if local run, the log collector doesn't support it as it is only supports k8s resources
        # when runtime is local the client is responsible for logging the stdout of the run by using `log_std`
        if RuntimeKinds.is_local_runtime(kind):
            return False

        if kind not in [
            # dask implementation is different than other runtimes, because few runs can be run against the same runtime
            # resource, so collecting logs on that runtime resource won't be correct, the way we collect logs for dask
            # is by using `log_std` on client side after we execute the code against the cluster, as submitting the
            # run with the dask client will return the run stdout. for more information head to `DaskCluster._run`
            RuntimeKinds.dask
        ]:
            return True

        return False

    @staticmethod
    def is_local_runtime(kind):
        # "" or None counted as local
        if not kind or kind in RuntimeKinds.local_runtimes():
            return True
        return False

    @staticmethod
    def requires_absolute_artifacts_path(kind):
        """
        Returns True if the runtime kind requires absolute artifacts' path (i.e. is local), False otherwise.
        """
        if RuntimeKinds.is_local_runtime(kind):
            return False

        if kind not in [
            # logging artifacts is done externally to the dask cluster by a client that can either run locally (in which
            # case the path can be relative) or remotely (in which case the path must be absolute and will be passed
            # to another run)
            RuntimeKinds.dask
        ]:
            return True
        return False


def get_runtime_class(kind: str):
    if kind == RuntimeKinds.mpijob:
        return MpiRuntimeContainer.selector()

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
    }

    return kind_runtime_map[kind]
