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

import typing


class RuntimeKindsBase:
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
    application = "application"

    @staticmethod
    def all():
        return [
            RuntimeKindsBase.remote,
            RuntimeKindsBase.nuclio,
            RuntimeKindsBase.serving,
            RuntimeKindsBase.dask,
            RuntimeKindsBase.job,
            RuntimeKindsBase.spark,
            RuntimeKindsBase.remotespark,
            RuntimeKindsBase.mpijob,
            RuntimeKindsBase.local,
            RuntimeKindsBase.databricks,
            RuntimeKindsBase.application,
        ]

    @staticmethod
    def runtime_with_handlers():
        return [
            RuntimeKindsBase.dask,
            RuntimeKindsBase.job,
            RuntimeKindsBase.spark,
            RuntimeKindsBase.remotespark,
            RuntimeKindsBase.mpijob,
            RuntimeKindsBase.databricks,
        ]

    @staticmethod
    def abortable_runtimes():
        return [
            RuntimeKindsBase.job,
            RuntimeKindsBase.spark,
            RuntimeKindsBase.remotespark,
            RuntimeKindsBase.mpijob,
            RuntimeKindsBase.databricks,
            RuntimeKindsBase.local,
            RuntimeKindsBase.handler,
            "",
        ]

    @staticmethod
    def retriable_runtimes():
        return [
            RuntimeKindsBase.job,
        ]

    @staticmethod
    def nuclio_runtimes():
        return [
            RuntimeKindsBase.remote,
            RuntimeKindsBase.nuclio,
            RuntimeKindsBase.serving,
            RuntimeKindsBase.application,
        ]

    @staticmethod
    def pure_nuclio_deployed_runtimes():
        return [
            RuntimeKindsBase.remote,
            RuntimeKindsBase.nuclio,
            RuntimeKindsBase.serving,
        ]

    @staticmethod
    def handlerless_runtimes():
        return [
            RuntimeKindsBase.serving,
            # Application runtime handler is internal reverse proxy
            RuntimeKindsBase.application,
        ]

    @staticmethod
    def local_runtimes():
        return [
            RuntimeKindsBase.local,
            RuntimeKindsBase.handler,
        ]

    @staticmethod
    def is_log_collectable_runtime(kind: typing.Optional[str]):
        """
        whether log collector can collect logs for that runtime
        :param kind: kind name
        :return: whether log collector can collect logs for that runtime
        """
        # if local run, the log collector doesn't support it as it is only supports k8s resources
        # when runtime is local the client is responsible for logging the stdout of the run by using `log_std`
        if RuntimeKindsBase.is_local_runtime(kind):
            return False

        if (
            kind
            not in [
                # dask implementation is different from other runtimes, because few runs can be run against the same
                # runtime resource, so collecting logs on that runtime resource won't be correct, the way we collect
                # logs for dask is by using `log_std` on client side after we execute the code against the cluster,
                # as submitting the run with the dask client will return the run stdout.
                # For more information head to `DaskCluster._run`.
                RuntimeKindsBase.dask
            ]
            + RuntimeKindsBase.nuclio_runtimes()
        ):
            return True

        return False

    @staticmethod
    def is_local_runtime(kind):
        # "" or None counted as local
        if not kind or kind in RuntimeKindsBase.local_runtimes():
            return True
        return False

    @staticmethod
    def requires_k8s_name_validation(kind: str) -> bool:
        """
        Returns True if the runtime kind creates Kubernetes resources that use the function name.

        Function names for k8s-deployed runtimes must conform to DNS-1123 label requirements:
        - Lowercase alphanumeric characters or '-'
        - Start and end with an alphanumeric character
        - Maximum 63 characters

        Local runtimes (local, handler) run on the local machine and don't create k8s resources,
        so they don't require k8s naming validation.

        :param kind: Runtime kind string (job, spark, serving, local, etc.)
        :return: True if function name needs k8s DNS-1123 validation, False otherwise
        """
        return not RuntimeKindsBase.is_local_runtime(kind)

    @staticmethod
    def requires_absolute_artifacts_path(kind):
        """
        Returns True if the runtime kind requires absolute artifacts' path (i.e. is local), False otherwise.
        """
        if RuntimeKindsBase.is_local_runtime(kind):
            return False

        if kind not in [
            # logging artifacts is done externally to the dask cluster by a client that can either run locally (in which
            # case the path can be relative) or remotely (in which case the path must be absolute and will be passed
            # to another run)
            RuntimeKindsBase.dask
        ]:
            return True
        return False

    @staticmethod
    def requires_image_name_for_execution(kind):
        if RuntimeKindsBase.is_local_runtime(kind):
            return False

        # both spark and remote spark uses different mechanism for assigning images
        return kind not in [RuntimeKindsBase.spark, RuntimeKindsBase.remotespark]

    @staticmethod
    def supports_from_notebook(kind):
        return kind not in [RuntimeKindsBase.application]


# Backwards compatibility: keep `mlrun.runtimes.constants.RuntimeKinds` import working.
# The public, extended API remains `mlrun.runtimes.RuntimeKinds`.
RuntimeKinds = RuntimeKindsBase
