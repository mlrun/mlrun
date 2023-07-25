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

from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from mlrun.api.runtime_handlers import BaseRuntimeHandler
from mlrun.runtimes.base import RuntimeClassMode
from mlrun.runtimes.constants import MPIJobV1Alpha1States, RunStates
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
