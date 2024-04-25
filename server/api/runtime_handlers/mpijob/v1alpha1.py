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

import mlrun.k8s_utils
import mlrun.utils.helpers
from mlrun import mlconf
from mlrun.runtimes.constants import MPIJobV1Alpha1States, RunStates
from mlrun.runtimes.mpijob import AbstractMPIJobRuntime, MpiRuntimeV1Alpha1
from mlrun.utils import update_in
from server.api.runtime_handlers.mpijob.abstract import AbstractMPIJobRuntimeHandler


class MpiV1Alpha1RuntimeHandler(AbstractMPIJobRuntimeHandler):
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
            },
        },
    }

    @staticmethod
    def _update_container(struct, key, value):
        struct["spec"]["template"]["spec"]["containers"][0][key] = value

    def _generate_mpi_job(
        self,
        runtime: AbstractMPIJobRuntime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
        meta: client.V1ObjectMeta,
    ) -> dict:
        job = copy.deepcopy(self._mpijob_template)

        pod_labels = copy.deepcopy(meta.labels)
        pod_labels["mlrun/job"] = meta.name
        update_in(job, "metadata", meta.to_dict())
        update_in(job, "spec.template.metadata.labels", pod_labels)
        update_in(job, "spec.replicas", runtime.spec.replicas or 1)
        if runtime.spec.image:
            self._update_container(
                job,
                "image",
                runtime.full_image_path(
                    client_version=run.metadata.labels.get("mlrun/client_version"),
                    client_python_version=run.metadata.labels.get(
                        "mlrun/client_python_version"
                    ),
                ),
            )
        update_in(job, "spec.template.spec.volumes", runtime.spec.volumes)
        self._update_container(job, "volumeMounts", runtime.spec.volume_mounts)
        update_in(job, "spec.template.spec.nodeName", runtime.spec.node_name)
        update_in(job, "spec.template.spec.nodeSelector", runtime.spec.node_selector)
        update_in(
            job,
            "spec.template.spec.affinity",
            mlrun.runtimes.pod.get_sanitized_attribute(runtime.spec, "affinity"),
        )
        update_in(
            job,
            "spec.template.spec.tolerations",
            mlrun.runtimes.pod.get_sanitized_attribute(runtime.spec, "tolerations"),
        )
        update_in(
            job,
            "spec.template.spec.securityContext",
            mlrun.runtimes.pod.get_sanitized_attribute(
                runtime.spec, "security_context"
            ),
        )
        if runtime.spec.priority_class_name and len(
            mlconf.get_valid_function_priority_class_names()
        ):
            update_in(
                job,
                "spec.template.spec.priorityClassName",
                runtime.spec.priority_class_name,
            )

        extra_env = runtime.generate_runtime_k8s_env(run)
        self._update_container(job, "env", extra_env + runtime.spec.env)
        if runtime.spec.image_pull_policy:
            self._update_container(
                job, "imagePullPolicy", runtime.spec.image_pull_policy
            )
        if runtime.spec.resources:
            self._update_container(job, "resources", runtime.spec.resources)
        if runtime.spec.workdir:
            self._update_container(job, "workingDir", runtime.spec.workdir)

        if runtime.spec.image_pull_secret:
            update_in(
                job,
                "spec.template.spec.imagePullSecrets",
                [{"name": runtime.spec.image_pull_secret}],
            )

        if runtime.spec.command:
            self._update_container(
                job,
                "command",
                ["mpirun", "python", runtime.spec.command] + runtime.spec.args,
            )

        return job

    def _get_job_launcher_status(self, resp: list) -> str:
        return mlrun.utils.get_in(resp, "status.launcherStatus")

    @staticmethod
    def _generate_pods_selector(name: str, launcher: bool) -> str:
        selector = "mlrun/class=mpijob"
        if name:
            selector += f",mpi_job_name={name}"
        if launcher:
            selector += ",mpi_role_type=launcher"

        return selector

    def _resolve_crd_object_status_info(
        self, crd_object: dict
    ) -> tuple[bool, typing.Optional[datetime], typing.Optional[str]]:
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
    def are_resources_coupled_to_run_object() -> bool:
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
    def _get_crd_info() -> tuple[str, str, str]:
        return (
            MpiRuntimeV1Alpha1.crd_group,
            MpiRuntimeV1Alpha1.crd_version,
            MpiRuntimeV1Alpha1.crd_plural,
        )

    @staticmethod
    def _get_crd_object_status(crd_object) -> str:
        return crd_object.get("status", {}).get("launcherStatus", "")
