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
import typing
from copy import deepcopy
from datetime import datetime

from kubernetes import client
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.runtimes.base import BaseRuntimeHandler, RunStates
from mlrun.runtimes.constants import MPIJobCRDVersions, MPIJobV1Alpha1States
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime
from mlrun.utils import update_in, get_in


class MpiRuntimeV1Alpha1(AbstractMPIJobRuntime):
    _mpijob_template = {
        "apiVersion": "kubeflow.org/v1alpha1",
        "kind": "MPIJob",
        "metadata": {"name": "", "namespace": "default-tenant"},
        "spec": {
            "replicas": 1,
            "template": {
                "metadata": {},
                "spec": {
                    "containers": [
                        {
                            "image": "mlrun/ml-models",
                            "name": "base",
                            "command": [],
                            "env": [],
                            "volumeMounts": [],
                            "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}},
                            "resources": {"limits": {}},
                        }
                    ],
                    "volumes": [],
                },
            },
        },
    }

    crd_group = "kubeflow.org"
    crd_version = MPIJobCRDVersions.v1alpha1
    crd_plural = "mpijobs"

    def _update_container(self, struct, key, value):
        struct["spec"]["template"]["spec"]["containers"][0][key] = value

    def _generate_mpi_job(
        self, runobj: RunObject, execution: MLClientCtx, meta: client.V1ObjectMeta
    ) -> typing.Dict:
        job = deepcopy(self._mpijob_template)

        pod_labels = deepcopy(meta.labels)
        pod_labels["mlrun/job"] = meta.name
        update_in(job, "metadata", meta.to_dict())
        update_in(job, "spec.template.metadata.labels", pod_labels)
        update_in(job, "spec.replicas", self.spec.replicas or 1)
        if self.spec.image:
            self._update_container(job, "image", self.full_image_path())
        update_in(job, "spec.template.spec.volumes", self.spec.volumes)
        self._update_container(job, "volumeMounts", self.spec.volume_mounts)

        extra_env = self._generate_runtime_env(runobj)
        extra_env = [{"name": k, "value": v} for k, v in extra_env.items()]
        self._update_container(job, "env", extra_env + self.spec.env)
        if self.spec.image_pull_policy:
            self._update_container(job, "imagePullPolicy", self.spec.image_pull_policy)
        if self.spec.resources:
            self._update_container(job, "resources", self.spec.resources)
        if self.spec.workdir:
            self._update_container(job, "workingDir", self.spec.workdir)

        if self.spec.image_pull_secret:
            update_in(
                job,
                "spec.template.spec.imagePullSecrets",
                [{"name": self.spec.image_pull_secret}],
            )

        if self.spec.command:
            self._update_container(
                job, "command", ["mpirun", "python", self.spec.command] + self.spec.args
            )

        return job

    def _get_job_launcher_status(self, resp: typing.List) -> str:
        return get_in(resp, "status.launcherStatus")

    @staticmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        selector = "mlrun/class=mpijob"
        if name:
            selector += ",mpi_job_name={}".format(name)
        if launcher:
            selector += ",mpi_role_type=launcher"

        return selector

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1Alpha1.crd_group,
            MpiRuntimeV1Alpha1.crd_version,
            MpiRuntimeV1Alpha1.crd_plural,
        )


class MpiV1Alpha1RuntimeHandler(BaseRuntimeHandler):
    def _resolve_crd_object_status_info(
        self, db: DBInterface, db_session: Session, crd_object
    ) -> typing.Tuple[bool, typing.Optional[datetime], typing.Optional[str]]:
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
    def _consider_run_on_resources_deletion() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"

    @staticmethod
    def _get_default_label_selector() -> str:
        return "mlrun/class=mpijob"

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
