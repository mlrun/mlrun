# Copyright 2018 Iguazio
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

from .base import RunError, BaseRuntime, BaseRuntimeHandler  # noqa
from .daskjob import DaskCluster, DaskRuntimeHandler, get_dask_resource  # noqa
from .function import RemoteRuntime
from .kubejob import KubejobRuntime, KubeRuntimeHandler  # noqa
from .local import HandlerRuntime, LocalRuntime  # noqa
from .mpijob import (
    MpiRuntimeV1Alpha1,
    MpiRuntimeV1,
    MpiV1RuntimeHandler,
    MpiV1Alpha1RuntimeHandler,
)  # noqa
from .constants import MPIJobCRDVersions
from .nuclio import nuclio_init_hook
from .serving import ServingRuntime, new_v2_model_server

# for legacy imports (MLModelServer moved from here to /serving)
from ..serving import MLModelServer, new_v1_model_server  # noqa
from .sparkjob import SparkRuntime, SparkRuntimeHandler  # noqa
from mlrun.runtimes.utils import resolve_mpijob_crd_version


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
    mpijob = "mpijob"
    serving = "serving"

    @staticmethod
    def all():
        return [
            RuntimeKinds.remote,
            RuntimeKinds.nuclio,
            RuntimeKinds.serving,
            RuntimeKinds.dask,
            RuntimeKinds.job,
            RuntimeKinds.spark,
            RuntimeKinds.mpijob,
        ]

    @staticmethod
    def runtime_with_handlers():
        return [
            RuntimeKinds.dask,
            RuntimeKinds.job,
            RuntimeKinds.spark,
            RuntimeKinds.mpijob,
        ]

    @staticmethod
    def nuclio_runtimes():
        return [
            RuntimeKinds.remote,
            RuntimeKinds.nuclio,
            RuntimeKinds.serving,
        ]


runtime_resources_map = {RuntimeKinds.dask: get_dask_resource()}

runtime_handler_instances_cache = {}


def get_runtime_handler(kind: str) -> BaseRuntimeHandler:
    global runtime_handler_instances_cache
    if kind == RuntimeKinds.mpijob:
        mpijob_crd_version = resolve_mpijob_crd_version()
        crd_version_to_runtime_handler_class = {
            MPIJobCRDVersions.v1alpha1: MpiV1Alpha1RuntimeHandler,
            MPIJobCRDVersions.v1: MpiV1RuntimeHandler,
        }
        runtime_handler_class = crd_version_to_runtime_handler_class[mpijob_crd_version]
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
        RuntimeKinds.job: KubeRuntimeHandler,
    }
    runtime_handler_class = kind_runtime_handler_map[kind]
    if not runtime_handler_instances_cache.get(kind):
        runtime_handler_instances_cache[kind] = runtime_handler_class()
    return runtime_handler_instances_cache[kind]


def get_runtime_class(kind: str):
    if kind == RuntimeKinds.mpijob:
        mpijob_crd_version = resolve_mpijob_crd_version()
        crd_version_to_runtime = {
            MPIJobCRDVersions.v1alpha1: MpiRuntimeV1Alpha1,
            MPIJobCRDVersions.v1: MpiRuntimeV1,
        }
        return crd_version_to_runtime[mpijob_crd_version]

    kind_runtime_map = {
        RuntimeKinds.remote: RemoteRuntime,
        RuntimeKinds.nuclio: RemoteRuntime,
        RuntimeKinds.serving: ServingRuntime,
        RuntimeKinds.dask: DaskCluster,
        RuntimeKinds.job: KubejobRuntime,
        RuntimeKinds.spark: SparkRuntime,
    }

    return kind_runtime_map[kind]
