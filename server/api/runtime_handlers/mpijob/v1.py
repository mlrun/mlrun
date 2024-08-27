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
import copy
import typing
from datetime import datetime

from kubernetes import client

import mlrun.common.constants as mlrun_constants
import mlrun.k8s_utils
import mlrun.utils.helpers
from mlrun import mlconf
from mlrun.common.runtimes.constants import RunStates
from mlrun.runtimes.mpijob import MpiRuntimeV1
from mlrun.utils import update_in
from server.api.runtime_handlers.mpijob.abstract import AbstractMPIJobRuntimeHandler


class MpiV1RuntimeHandler(AbstractMPIJobRuntimeHandler):
    _mpijob_pod_template = {
        "spec": {
            "containers": [
                {
                    "image": "mlrun/mlrun",
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

    def _generate_mpi_job(
        self,
        runtime: MpiRuntimeV1,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
        meta: client.V1ObjectMeta,
    ) -> dict:
        pod_labels = copy.deepcopy(meta.labels)
        pod_labels[mlrun_constants.MLRunInternalLabels.job] = meta.name

        # Populate mpijob object

        # start by populating pod templates
        launcher_pod_template = copy.deepcopy(self._mpijob_pod_template)
        worker_pod_template = copy.deepcopy(self._mpijob_pod_template)
        command, args, extra_env = self._get_cmd_args(runtime, run)

        # configuration for both launcher and workers
        for pod_template in [launcher_pod_template, worker_pod_template]:
            if runtime.spec.image:
                self._update_container(
                    pod_template,
                    "image",
                    runtime.full_image_path(
                        client_version=run.metadata.labels.get(
                            mlrun_constants.MLRunInternalLabels.client_version
                        ),
                        client_python_version=run.metadata.labels.get(
                            mlrun_constants.MLRunInternalLabels.client_python_version
                        ),
                    ),
                )
            self._update_container(
                pod_template, "volumeMounts", runtime.spec.volume_mounts
            )
            self._update_container(pod_template, "env", extra_env + runtime.spec.env)
            if runtime.spec.image_pull_policy:
                self._update_container(
                    pod_template,
                    "imagePullPolicy",
                    runtime.spec.image_pull_policy,
                )
            if runtime.spec.workdir:
                self._update_container(pod_template, "workingDir", runtime.spec.workdir)
            if runtime.spec.image_pull_secret:
                update_in(
                    pod_template,
                    "spec.imagePullSecrets",
                    [{"name": runtime.spec.image_pull_secret}],
                )
            if runtime.spec.security_context:
                update_in(
                    pod_template,
                    "spec.securityContext",
                    mlrun.runtimes.pod.get_sanitized_attribute(
                        runtime.spec, "security_context"
                    ),
                )
            update_in(pod_template, "metadata.labels", pod_labels)
            update_in(pod_template, "spec.volumes", runtime.spec.volumes)
            update_in(pod_template, "spec.nodeName", runtime.spec.node_name)
            update_in(
                pod_template,
                "spec.nodeSelector",
                mlrun.utils.helpers.to_non_empty_values_dict(run.spec.node_selector),
            )
            update_in(
                pod_template,
                "spec.affinity",
                mlrun.runtimes.pod.get_sanitized_attribute(runtime.spec, "affinity"),
            )
            update_in(
                pod_template,
                "spec.tolerations",
                mlrun.runtimes.pod.get_sanitized_attribute(runtime.spec, "tolerations"),
            )
            if runtime.spec.priority_class_name and len(
                mlconf.get_valid_function_priority_class_names()
            ):
                update_in(
                    pod_template,
                    "spec.priorityClassName",
                    runtime.spec.priority_class_name,
                )
            if runtime.spec.service_account:
                update_in(
                    pod_template,
                    "spec.serviceAccountName",
                    runtime.spec.service_account,
                )

        # configuration for workers only
        # update resources only for workers because the launcher
        # doesn't require special resources (like GPUs, Memory, etc..)
        self._enrich_worker_configurations(runtime, worker_pod_template)

        # configuration for launcher only
        self._enrich_launcher_configurations(
            runtime, launcher_pod_template, [command] + args
        )

        # generate mpi job using both pod templates
        job = self._generate_mpi_job_template(
            launcher_pod_template, worker_pod_template
        )

        # update the replicas only for workers
        update_in(
            job,
            "spec.mpiReplicaSpecs.Worker.replicas",
            runtime.spec.replicas or 1,
        )

        update_in(
            job,
            "spec.cleanPodPolicy",
            runtime.spec.clean_pod_policy,
        )

        if execution.get_param("slots_per_worker"):
            update_in(
                job,
                "spec.slotsPerWorker",
                execution.get_param("slots_per_worker"),
            )

        update_in(job, "metadata", meta.to_dict())

        return job

    def _get_job_launcher_status(self, resp: list) -> str:
        launcher_status = mlrun.utils.get_in(resp, "status.replicaStatuses.Launcher")
        if launcher_status is None:
            return ""

        for status in ["active", "failed", "succeeded"]:
            if launcher_status.get(status, 0) == 1:
                return status

        return ""

    @staticmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        selector = f"{mlrun_constants.MLRunInternalLabels.mpi_job_name}={name}"
        if launcher:
            selector += f",{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher"

        return selector

    def _generate_mpi_job_template(self, launcher_pod_template, worker_pod_template):
        # https://github.com/kubeflow/mpi-operator/blob/master/pkg/apis/kubeflow/v1/types.go#L25
        # MPI job consists of Launcher and Worker which both are of type ReplicaSet
        # https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L74
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "MPIJob",
            "metadata": {"name": "", "namespace": "default-tenant"},
            "spec": {
                "slotsPerWorker": 1,
                "cleanPodPolicy": "All",
                "mpiReplicaSpecs": {
                    "Launcher": {"template": launcher_pod_template},
                    "Worker": {"replicas": 1, "template": worker_pod_template},
                },
            },
        }

    @staticmethod
    def _update_container(struct, key, value):
        struct["spec"]["containers"][0][key] = value

    def _enrich_launcher_configurations(
        self, runtime: MpiRuntimeV1, launcher_pod_template, args
    ):
        quoted_args = args or []
        quoted_mpi_args = []
        for arg in runtime.spec.mpi_args:
            quoted_mpi_args.append(arg)
        self._update_container(
            launcher_pod_template,
            "command",
            ["mpirun", *quoted_mpi_args, *quoted_args],
        )
        self._update_container(
            launcher_pod_template,
            "resources",
            mlconf.get_default_function_pod_resources(),
        )

    def _enrich_worker_configurations(self, runtime: MpiRuntimeV1, worker_pod_template):
        if runtime.spec.resources:
            self._update_container(
                worker_pod_template, "resources", runtime.spec.resources
            )

    def _resolve_crd_object_status_info(
        self, crd_object: dict
    ) -> tuple[bool, typing.Optional[datetime], typing.Optional[str]]:
        """
        https://github.com/kubeflow/mpi-operator/blob/v0.3.0/pkg/apis/kubeflow/v1/types.go#L29
        https://github.com/kubeflow/common/blob/master/pkg/apis/common/v1/types.go#L55
        """
        launcher_status = (
            crd_object.get("status", {}).get("replicaStatuses", {}).get("Launcher", {})
        )
        # the launcher status also has running property, but it's empty for
        # short period after the creation, so we're
        # checking terminal state by the completion time existence
        in_terminal_state = self._is_terminal_state(crd_object)
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

    def _resolve_container_error_status(self, crd_object: dict) -> tuple[str, str]:
        # TODO:
        return "", ""

    def _is_terminal_state(self, runtime_resource: dict) -> bool:
        return (
            runtime_resource.get("status", {}).get("completionTime", None) is not None
        )

    @staticmethod
    def are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"{mlrun_constants.MLRunInternalLabels.uid}={object_id}"

    @staticmethod
    def _get_main_runtime_resource_label_selector() -> str:
        """
        There are some runtimes which might have multiple k8s resources attached to a one runtime, in this case
        we don't want to pull logs from all but rather only for the "driver"/"launcher" etc
        :return: the label selector
        """
        return f"{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher"

    @staticmethod
    def _get_crd_info() -> tuple[str, str, str]:
        return (
            MpiRuntimeV1.crd_group,
            MpiRuntimeV1.crd_version,
            MpiRuntimeV1.crd_plural,
        )
