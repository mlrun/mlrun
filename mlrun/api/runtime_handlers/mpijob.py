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
from datetime import datetime

from dependency_injector import containers, providers
from sqlalchemy.orm import Session

import mlrun.api.utils.singletons.k8s
import mlrun.k8s_utils
from mlrun.api.db.base import DBInterface
from mlrun.api.runtime_handlers import BaseRuntimeHandler
from mlrun.config import config
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.constants import MPIJobCRDVersions, MPIJobV1Alpha1States, RunStates
from mlrun.runtimes.mpijob import MpiRuntimeV1, MpiRuntimeV1Alpha1


class MpiV1Alpha1RuntimeHandler(BaseRuntimeHandler):
    kind = "mpijob"
    class_modes = {
        RuntimeClassMode.run: "mpijob",
    }

    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> typing.Tuple[bool, typing.Optional[datetime], typing.Optional[str]]:
        """
        https://github.com/kubeflow/mpi-operator/blob/master/pkg/apis/kubeflow/v1alpha1/types.go#L115
        """
        launcher_status = crd_object.get("status", {}).get("launcherStatus", "")
        in_terminal_state = launcher_status in MPIJobV1Alpha1States.terminal_states()
        desired_run_state = MPIJobV1Alpha1States.mpijob_state_to_run_state(
            launcher_status
        )
        completion_time = None
        if in_terminal_state:
            completion_time = datetime.fromisoformat(
                crd_object.get("status", {})
                .get("completionTime")
                .replace("Z", "+00:00")
            )
            desired_run_state = {
                "Succeeded": RunStates.completed,
                "Failed": RunStates.error,
            }[launcher_status]
        return in_terminal_state, completion_time, desired_run_state

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_main_runtime_resource_label_selector() -> str:
        """
        There are some runtimes which might have multiple k8s resources attached to a one runtime, in this case
        we don't want to pull logs from all but rather only for the "driver"/"launcher" etc
        :return: the label selector
        """
        return "mpi_role_type=launcher"

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1Alpha1.crd_group,
            MpiRuntimeV1Alpha1.crd_version,
            MpiRuntimeV1Alpha1.crd_plural,
        )

    @staticmethod
    def _get_crd_object_status(crd_object) -> str:
        return crd_object.get("status", {}).get("launcherStatus", "")


class MpiV1RuntimeHandler(BaseRuntimeHandler):
    kind = "mpijob"
    class_modes = {
        RuntimeClassMode.run: "mpijob",
    }

    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> typing.Tuple[bool, typing.Optional[datetime], typing.Optional[str]]:
        """
        https://github.com/kubeflow/mpi-operator/blob/master/pkg/apis/kubeflow/v1/types.go#L29
        https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L55
        """
        launcher_status = (
            crd_object.get("status", {}).get("replicaStatuses", {}).get("Launcher", {})
        )
        # the launcher status also has running property, but it's empty for
        # short period after the creation, so we're
        # checking terminal state by the completion time existence
        in_terminal_state = (
            crd_object.get("status", {}).get("completionTime", None) is not None
        )
        desired_run_state = RunStates.running
        completion_time = None
        if in_terminal_state:
            completion_time = datetime.fromisoformat(
                crd_object.get("status", {})
                .get("completionTime")
                .replace("Z", "+00:00")
            )
            desired_run_state = (
                RunStates.error
                if launcher_status.get("failed", 0) > 0
                else RunStates.completed
            )
        return in_terminal_state, completion_time, desired_run_state

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_main_runtime_resource_label_selector() -> str:
        """
        There are some runtimes which might have multiple k8s resources attached to a one runtime, in this case
        we don't want to pull logs from all but rather only for the "driver"/"launcher" etc
        :return: the label selector
        """
        return "mpi-job-role=launcher"

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1.crd_group,
            MpiRuntimeV1.crd_version,
            MpiRuntimeV1.crd_plural,
        )


cached_mpijob_crd_version = None


def resolve_mpijob_crd_version():
    """
    Resolve mpijob runtime according to the mpi-operator's supported crd-version.
    If specified on mlrun config set it likewise.
    If not specified, try resolving it according to the mpi-operator, otherwise set to default.
    Since this is a heavy operation (sending requests to k8s/API), and it's unlikely that the crd version
    will change in any context - cache it.
    :return: mpi operator's crd-version
    """
    global cached_mpijob_crd_version
    if not cached_mpijob_crd_version:
        # try to resolve the crd version with K8s API
        # backoff to use default if needed
        mpijob_crd_version = (
            _resolve_mpijob_crd_version_best_effort() or MPIJobCRDVersions.default()
        )

        if mpijob_crd_version not in MPIJobCRDVersions.all():
            raise ValueError(
                f"Unsupported mpijob crd version: {mpijob_crd_version}. "
                f"Supported versions: {MPIJobCRDVersions.all()}"
            )
        cached_mpijob_crd_version = mpijob_crd_version

    return cached_mpijob_crd_version


def _resolve_mpijob_crd_version_best_effort():
    # config overrides everything
    if config.mpijob_crd_version:
        return config.mpijob_crd_version

    if not mlrun.k8s_utils.is_running_inside_kubernetes_cluster():
        return None

    k8s_helper = mlrun.api.utils.singletons.k8s.get_k8s_helper()
    namespace = k8s_helper.resolve_namespace()

    # try resolving according to mpi-operator that's running
    res = k8s_helper.list_pods(namespace=namespace, selector="component=mpi-operator")

    if len(res) == 0:
        return None

    mpi_operator_pod = res[0]
    return mpi_operator_pod.metadata.labels.get("crd-version")


# overrides the way we resolve the mpijob crd version by querying the k8s API
@containers.override(mlrun.runtimes.mpijob.MpiRuntimeContainer)
class MpiRuntimeHandlerContainer(containers.DeclarativeContainer):
    resolver = providers.Callable(
        resolve_mpijob_crd_version,
    )

    handler_selector = providers.Selector(
        resolver,
        v1=providers.Object(MpiV1RuntimeHandler),
        v1alpha1=providers.Object(MpiV1Alpha1RuntimeHandler),
    )
