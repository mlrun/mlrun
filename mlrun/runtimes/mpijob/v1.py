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
import shlex
import typing
from copy import deepcopy
from datetime import datetime

from kubernetes import client
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from mlrun.execution import MLClientCtx
from mlrun.model import RunObject
from mlrun.runtimes.base import BaseRuntimeHandler, RunStates
from mlrun.runtimes.constants import MPIJobCRDVersions
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime
from mlrun.utils import update_in, get_in


class MpiRuntimeV1(AbstractMPIJobRuntime):
    _mpijob_pod_template = {
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
        "metadata": {},
    }

    crd_group = "kubeflow.org"
    crd_version = MPIJobCRDVersions.v1
    crd_plural = "mpijobs"

    def _generate_mpi_job_template(self, launcher_pod_template, worker_pod_template):
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "MPIJob",
            "metadata": {"name": "", "namespace": "default-tenant"},
            "spec": {
                "slotsPerWorker": 1,
                "mpiReplicaSpecs": {
                    "Launcher": {"template": launcher_pod_template},
                    "Worker": {"replicas": 1, "template": worker_pod_template},
                },
            },
        }

    def _update_container(self, struct, key, value):
        struct["spec"]["containers"][0][key] = value

    def _enrich_launcher_configurations(self, launcher_pod_template):
        quoted_args = []
        for arg in self.spec.args:
            quoted_args.append(shlex.quote(arg))
        quoted_mpi_args = []
        for arg in self.spec.mpi_args:
            quoted_mpi_args.append(shlex.quote(arg))
        if self.spec.command:
            self._update_container(
                launcher_pod_template,
                "command",
                [
                    "mpirun",
                    *quoted_mpi_args,
                    "python",
                    shlex.quote(self.spec.command),
                    *quoted_args,
                ],
            )

    def _enrich_worker_configurations(self, worker_pod_template):
        if self.spec.resources:
            self._update_container(
                worker_pod_template, "resources", self.spec.resources
            )

    def _generate_mpi_job(
        self, runobj: RunObject, execution: MLClientCtx, meta: client.V1ObjectMeta,
    ) -> dict:
        pod_labels = deepcopy(meta.labels)
        pod_labels["mlrun/job"] = meta.name

        # Populate mpijob object

        # start by populating pod templates
        launcher_pod_template = deepcopy(self._mpijob_pod_template)
        worker_pod_template = deepcopy(self._mpijob_pod_template)

        # configuration for both launcher and workers
        for pod_template in [launcher_pod_template, worker_pod_template]:
            if self.spec.image:
                self._update_container(pod_template, "image", self.full_image_path())
            self._update_container(
                pod_template, "volumeMounts", self.spec.volume_mounts
            )
            extra_env = self._generate_runtime_env(runobj)
            extra_env = [{"name": k, "value": v} for k, v in extra_env.items()]
            self._update_container(pod_template, "env", extra_env + self.spec.env)
            if self.spec.image_pull_policy:
                self._update_container(
                    pod_template, "imagePullPolicy", self.spec.image_pull_policy,
                )
            if self.spec.workdir:
                self._update_container(pod_template, "workingDir", self.spec.workdir)
            if self.spec.image_pull_secret:
                update_in(
                    pod_template,
                    "spec.imagePullSecrets",
                    [{"name": self.spec.image_pull_secret}],
                )
            update_in(pod_template, "metadata.labels", pod_labels)
            update_in(pod_template, "spec.volumes", self.spec.volumes)

        # configuration for workers only
        # update resources only for workers because the launcher
        # doesn't require special resources (like GPUs, Memory, etc..)
        self._enrich_worker_configurations(worker_pod_template)

        # configuration for launcher only
        self._enrich_launcher_configurations(launcher_pod_template)

        # generate mpi job using both pod templates
        job = self._generate_mpi_job_template(
            launcher_pod_template, worker_pod_template
        )

        # update the replicas only for workers
        update_in(
            job, "spec.mpiReplicaSpecs.Worker.replicas", self.spec.replicas or 1,
        )

        if execution.get_param("slots_per_worker"):
            update_in(
                job, "spec.slotsPerWorker", execution.get_param("slots_per_worker"),
            )

        update_in(job, "metadata", meta.to_dict())

        return job

    def _get_job_launcher_status(self, resp: typing.List) -> str:
        launcher_status = get_in(resp, "status.replicaStatuses.Launcher")
        if launcher_status is None:
            return ""

        for status in ["active", "failed", "succeeded"]:
            if launcher_status.get(status, 0) == 1:
                return status

        return ""

    @staticmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        selector = "mpi-job-name={}".format(name)
        if launcher:
            selector += ",mpi-job-role=launcher"

        return selector

    @staticmethod
    def _get_crd_info() -> typing.Tuple[str, str, str]:
        return (
            MpiRuntimeV1.crd_group,
            MpiRuntimeV1.crd_version,
            MpiRuntimeV1.crd_plural,
        )


class MpiV1RuntimeHandler(BaseRuntimeHandler):
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
            MpiRuntimeV1.crd_group,
            MpiRuntimeV1.crd_version,
            MpiRuntimeV1.crd_plural,
        )
